from pathlib import Path

from pymilvus import MilvusClient, DataType
from config import settings


class VectorStore:
    def __init__(self):
        uri = settings.milvus_uri
        if "://" not in uri:
            Path(uri).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

        client_kwargs = {"uri": uri}
        if settings.milvus_token:
            client_kwargs["token"] = settings.milvus_token

        self.client = MilvusClient(**client_kwargs)
        self._ensure_collection(settings.collection_name)

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
                params={"nlist": 128},
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
                    "vector": vec,
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
            filter_expr = f'doc_name == "{doc_name}"'

        search_params = {"metric_type": "IP"}
        if settings.index_type == "IVF_FLAT":
            search_params["params"] = {"nprobe": 10}

        if settings.embed_provider == "colqwen2":
            page_patch_scores: dict[tuple[str, int], dict[int, float]] = {}
            for qv in query_vector:
                hits = self.client.search(
                    collection_name,
                    data=[qv],
                    limit=settings.colqwen2_candidate_patches,
                    output_fields=["doc_name", "page_idx", "patch_idx"],
                    search_params=search_params,
                    filter=filter_expr,
                )[0]
                for h in hits:
                    entity = h["entity"]
                    page_key = (entity["doc_name"], int(entity["page_idx"]))
                    patch_idx = int(entity["patch_idx"])
                    score = float(h["distance"])
                    patch_scores = page_patch_scores.setdefault(page_key, {})
                    if score > patch_scores.get(patch_idx, float("-inf")):
                        patch_scores[patch_idx] = score

            ranked = sorted(
                (
                    {
                        "doc_name": doc_name_e,
                        "page_idx": page_idx,
                        "score": sum(patch_scores.values()),
                    }
                    for (doc_name_e, page_idx), patch_scores in page_patch_scores.items()
                ),
                key=lambda item: item["score"],
                reverse=True,
            )
            return ranked[:top_k]

        hits = self.client.search(
            collection_name,
            data=[query_vector],
            limit=top_k,
            output_fields=["doc_name", "page_idx"],
            search_params=search_params,
            filter=filter_expr,
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
