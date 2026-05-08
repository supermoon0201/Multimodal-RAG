import logging
from pathlib import Path

import numpy as np
from pymilvus import MilvusClient, DataType
from config import settings

logger = logging.getLogger(__name__)


def _l2_normalize(vec):
    arr = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm == 0:
        return arr.tolist()
    return (arr / norm).tolist()


def _escape_filter_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


class VectorStore:
    def __init__(self):
        self.client_kwargs = self._build_client_kwargs()
        self.client = self._connect()
        self._ensure_collection(settings.collection_name)

    @staticmethod
    def _build_client_kwargs() -> dict:
        uri = settings.milvus_uri
        if "://" not in uri:
            Path(uri).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

        client_kwargs = {"uri": uri}
        if settings.milvus_token:
            client_kwargs["token"] = settings.milvus_token
        return client_kwargs

    def _connect(self) -> MilvusClient:
        return MilvusClient(**self.client_kwargs)

    def _reconnect(self):
        close = getattr(self.client, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                logger.debug("Ignoring Milvus close failure before reconnect", exc_info=True)
        self.client = self._connect()

    @staticmethod
    def _is_retryable_error(exc: Exception) -> bool:
        message = str(exc)
        return any(
            marker in message
            for marker in (
                "GOAWAY",
                "UNAVAILABLE",
                "too_many_pings",
                "ENHANCE_YOUR_CALM",
            )
        )

    def _run_search_with_reconnect(self, operation):
        try:
            return operation()
        except Exception as exc:
            if not self._is_retryable_error(exc):
                raise
            logger.warning("Milvus search connection was closed by gRPC (%s); reconnecting once", exc)
            self._reconnect()
            return operation()

    def _fetch_colqwen2_page_vectors(self, doc_name: str, page_idx: int) -> list[list[float]]:
        safe_doc_name = _escape_filter_value(doc_name)
        rows = self._run_search_with_reconnect(
            lambda: self.client.query(
                settings.collection_name,
                filter=f'doc_name == "{safe_doc_name}" and page_idx == {page_idx}',
                output_fields=["vector"],
                limit=16384,
            )
        )
        return [row["vector"] for row in rows]

    def _rerank_colqwen2_pages(
        self,
        query_vectors: list[list[float]],
        candidates: list[dict],
        top_k: int,
    ) -> list[dict]:
        query_arr = np.asarray(query_vectors, dtype=np.float32)
        reranked = []

        for candidate in candidates:
            page_vectors = self._fetch_colqwen2_page_vectors(
                candidate["doc_name"],
                int(candidate["page_idx"]),
            )
            if not page_vectors:
                continue

            page_arr = np.asarray(page_vectors, dtype=np.float32)
            scores = query_arr @ page_arr.T
            reranked.append({
                "doc_name": candidate["doc_name"],
                "page_idx": candidate["page_idx"],
                "score": float(scores.max(axis=1).sum()),
            })

        reranked.sort(key=lambda item: item["score"], reverse=True)
        return reranked[:top_k]

    def _ensure_collection(self, collection_name: str):
        if self.client.has_collection(collection_name):
            return
        self.create_collection(collection_name)

    def create_collection(self, collection_name: str = None):
        if collection_name is None:
            collection_name = settings.collection_name
        if self.client.has_collection(collection_name):
            self.client.drop_collection(collection_name)

        schema = self.client.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("doc_name", DataType.VARCHAR, max_length=256)
        schema.add_field("page_idx", DataType.INT64)
        if settings.embed_provider == "colqwen2":
            schema.add_field("patch_idx", DataType.INT64)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=settings.embed_dim)

        index = self.client.prepare_index_params()
        if settings.index_type == "IVF_FLAT":
            index.add_index(
                field_name="vector",
                index_type="IVF_FLAT",
                metric_type="IP",
                params={"nlist": settings.ivf_nlist},
            )
        else:
            index.add_index(
                field_name="vector",
                index_type=settings.index_type,
                metric_type="IP",
            )

        self.client.create_collection(collection_name, schema=schema, index_params=index)
        print(f"[INFO] Collection '{collection_name}' created (dim={settings.embed_dim}).")

    def insert_pages(self, doc_name: str, page_vectors) -> int:
        """Insert embeddings into the collection.

        For single-vector providers, stores one row per page.
        For ColQwen2, stores one row per patch vector.
        """
        collection_name = settings.collection_name
        rows = []
        if settings.embed_provider == "colqwen2":
            for page_idx, patch_vectors in enumerate(page_vectors):
                for patch_idx, vec in enumerate(patch_vectors):
                    rows.append({
                        "doc_name": doc_name,
                        "page_idx": page_idx,
                        "patch_idx": patch_idx,
                        "vector": vec,
                    })
        else:
            rows = [
                {
                    "doc_name": doc_name,
                    "page_idx": idx,
                    "vector": _l2_normalize(vec),
                }
                for idx, vec in enumerate(page_vectors)
            ]
        self.client.insert(collection_name, rows)
        self.client.flush(collection_name)
        return len(rows)

    def search(self, query_vector, top_k: int = None,
               doc_name: str = None) -> list[dict]:
        if top_k is None:
            top_k = settings.top_k
        collection_name = settings.collection_name

        filter_expr = None
        if doc_name:
            filter_expr = f'doc_name == "{_escape_filter_value(doc_name)}"'

        search_params = {"metric_type": "IP"}
        if settings.index_type == "IVF_FLAT":
            search_params["params"] = {"nprobe": min(settings.ivf_nprobe, settings.ivf_nlist)}

        if settings.embed_provider == "colqwen2":
            page_query_scores: dict[tuple[str, int], dict[int, float]] = {}

            hits_per_query = self._run_search_with_reconnect(
                lambda: self.client.search(
                    collection_name,
                    data=query_vector,
                    limit=settings.colqwen2_candidate_patches,
                    output_fields=["doc_name", "page_idx"],
                    search_params=search_params,
                    filter=filter_expr,
                )
            )

            for query_idx, hits in enumerate(hits_per_query):
                for h in hits:
                    entity = h["entity"]
                    page_key = (entity["doc_name"], int(entity["page_idx"]))
                    score = float(h["distance"])
                    query_scores = page_query_scores.setdefault(page_key, {})
                    if score > query_scores.get(query_idx, float("-inf")):
                        query_scores[query_idx] = score

            ranked = sorted(
                (
                    {
                        "doc_name": doc_name_e,
                        "page_idx": page_idx,
                        "score": sum(query_scores.values()),
                    }
                    for (doc_name_e, page_idx), query_scores in page_query_scores.items()
                ),
                key=lambda item: item["score"],
                reverse=True,
            )
            candidate_pages = ranked[: max(top_k * 20, 50)]
            return self._rerank_colqwen2_pages(query_vector, candidate_pages, top_k)

        query_vector = _l2_normalize(query_vector)
        hits = self._run_search_with_reconnect(
            lambda: self.client.search(
                collection_name,
                data=[query_vector],
                limit=top_k,
                output_fields=["doc_name", "page_idx"],
                search_params=search_params,
                filter=filter_expr,
            )
        )[0]

        return [
            {
                "doc_name": h["entity"]["doc_name"],
                "page_idx": h["entity"]["page_idx"],
                "score": h["distance"],
            }
            for h in hits
        ]

    def get_collection_stats(self) -> dict:
        return self.client.get_collection_stats(settings.collection_name)

    def drop_collection(self, collection_name: str = None):
        if collection_name is None:
            collection_name = settings.collection_name
        if self.client.has_collection(collection_name):
            self.client.drop_collection(collection_name)
            print(f"[INFO] Collection '{collection_name}' dropped.")
