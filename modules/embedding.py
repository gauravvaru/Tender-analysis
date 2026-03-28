"""
Embedding generation and FAISS vector search
"""
import logging
from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Generate embeddings using sentence-transformers"""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.vector_size = 384
        logger.info(f"Embedding model loaded: {model_name}")

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings

    def embed_single(self, text: str) -> np.ndarray:
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embedding


class FAISSVectorStore:
    """FAISS-backed vector store with duplicate prevention"""

    def __init__(
        self,
        dimension: int = 384,
        index_dir: str = "data/faiss_index"
    ):
        self.dimension = dimension
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.index_path = self.index_dir / "faiss_index.bin"
        self.metadata_path = self.index_dir / "metadata.pkl"

        self.embedder = EmbeddingService()

        if self.index_path.exists() and self.metadata_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            with open(self.metadata_path, "rb") as f:
                self.chunk_metadata = pickle.load(f)
            logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
        else:
            self.index = faiss.IndexFlatL2(dimension)
            self.chunk_metadata: List[Dict] = []
            logger.info("Created new FAISS index")

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_vectors(self, embeddings: np.ndarray, metadata_list: List[Dict]):
        """
        Add vectors to FAISS while skipping duplicates.

        Duplicate detection uses chunk 'id' if present, otherwise a hash
        of the chunk text.
        """
        embeddings = embeddings.astype(np.float32)

        # Build set of already-indexed identifiers
        existing_ids = set()
        for m in self.chunk_metadata:
            if "id" in m:
                existing_ids.add(m["id"])
            elif "text" in m:
                existing_ids.add(hash(m["text"]))

        new_vectors: List[np.ndarray] = []
        new_metadata: List[Dict] = []

        for vec, meta in zip(embeddings, metadata_list):
            identifier = meta.get("id", hash(meta.get("text", "")))

            if identifier in existing_ids:
                continue

            existing_ids.add(identifier)
            new_vectors.append(vec)
            new_metadata.append(meta)

        if not new_vectors:
            logger.info("No new vectors to add (all duplicates skipped)")
            return

        vectors_array = np.array(new_vectors, dtype=np.float32)
        self.index.add(vectors_array)
        self.chunk_metadata.extend(new_metadata)

        self._persist()
        logger.info(f"Added {len(new_vectors)} new vectors")

    def _persist(self):
        """Save index and metadata to disk."""
        faiss.write_index(self.index, str(self.index_path))
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.chunk_metadata, f)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search(
        self,
        query_text: str,
        k: int = 5,
        filters: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Semantic search: embed query → FAISS search → optional metadata filter.

        Returns up to k results sorted by similarity (highest first).
        Each result:
            {
                "index": int,
                "similarity": float,   # 1 / (1 + L2_distance), higher = more similar
                "distance": float,
                "metadata": dict
            }
        """
        if self.index.ntotal == 0:
            logger.warning("FAISS index is empty — no results returned")
            return []

        try:
            query_embedding = self.embedder.embed_single(query_text)
            query_embedding = query_embedding.astype(np.float32).reshape(1, -1)

            # Fetch extra candidates so we still return k after filtering
            fetch_k = min(k * 3, self.index.ntotal)
            distances, indices = self.index.search(query_embedding, fetch_k)

            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx == -1:
                    continue

                # FIX 7: Guard against index out-of-range (can happen if metadata
                # and index fall out of sync after a partial write)
                if idx >= len(self.chunk_metadata):
                    logger.warning(f"Index {idx} out of range for metadata list")
                    continue

                metadata = self.chunk_metadata[idx]

                # Optional metadata filter
                if filters:
                    if any(metadata.get(k) != v for k, v in filters.items()):
                        continue

                similarity = float(1 / (1 + distance))
                results.append({
                    "index": int(idx),
                    "similarity": similarity,
                    "distance": float(distance),
                    "metadata": metadata,
                })

                if len(results) >= k:
                    break

            logger.info(f"Search returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict:
        return {
            "total_vectors": self.index.ntotal,
            "vector_dimension": self.dimension,
            "total_chunks": len(self.chunk_metadata),
        }

    def reset_index(self):
        """
        Clear the index completely — useful during development/testing
        when you want to re-ingest all documents from scratch.
        """
        self.index = faiss.IndexFlatL2(self.dimension)
        self.chunk_metadata = []
        self._persist()
        logger.info("FAISS index reset")
