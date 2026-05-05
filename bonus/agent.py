from __future__ import annotations

import math
import re
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

try:
    from fastembed import TextEmbedding
except Exception:
    TextEmbedding = None
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams
except Exception:
    QdrantClient = None
try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None

ROOT = Path(__file__).resolve().parent.parent
FEAST_DIR = ROOT / "app" / "feast_repo"
COLLECTION = "bonus_episodic_memory"
EMBED_DIM = 384


class HashEmbedder:
    """Tiny fallback so `python bonus/demo.py` still works without lab deps."""

    def embed(self, texts: list[str]):
        for text in texts:
            vector = [0.0] * EMBED_DIM
            for token in re.findall(r"\w+", text.lower()):
                vector[hash(token) % EMBED_DIM] += 1.0
            norm = math.sqrt(sum(x * x for x in vector)) or 1.0
            yield [x / norm for x in vector]


class HybridMemoryAgent:
    def __init__(self) -> None:
        self.embedder = self._make_embedder()
        self.qdrant = QdrantClient(":memory:") if QdrantClient else None
        if self.qdrant:
            self.qdrant.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
            )
        self.memories: list[dict] = []
        self.vectors: list[list[float]] = []
        self.bm25 = None
        self.next_id = 1
        self.recent_queries: dict[str, deque[str]] = defaultdict(lambda: deque(maxlen=10))

    def remember(self, text: str, user_id: str = "u_001") -> None:
        """Add a new piece of episodic memory for this user."""
        chunks = self._chunk(text)
        vectors = list(self.embedder.embed(chunks))
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        points = []
        for chunk, vector in zip(chunks, vectors):
            memory = {
                "point_id": self.next_id,
                "memory_id": f"mem_{self.next_id:04d}",
                "user_id": user_id,
                "text": chunk,
                "topic_hint": self._infer_topic(chunk),
                "created_at": now,
            }
            self.memories.append(memory)
            self.vectors.append(list(vector))
            if self.qdrant:
                points.append(PointStruct(id=self.next_id, vector=list(vector), payload=memory))
            self.next_id += 1
        if points:
            self.qdrant.upsert(collection_name=COLLECTION, points=points)
        self._rebuild_bm25()

    def recall(self, query: str, user_id: str = "u_001") -> str:
        """Retrieve top-K memories + user profile features, then return assembled context."""
        profile = self._get_profile(user_id)
        memories = self._rrf(
            self._keyword_search(query, user_id, 8),
            self._vector_search(query, user_id, 8),
            top_k=3,
        )
        self.recent_queries[user_id].append(query)
        memory_block = "\n".join(
            f"  {i}. [{m['topic_hint']}] {m['text']}" for i, m in enumerate(memories, 1)
        ) or "  Không tìm thấy memory phù hợp."
        return (
            "User profile:\n"
            f"  user_id: {user_id}\n"
            f"  preferred_language: {profile['preferred_language']}\n"
            f"  reading_speed_wpm: {profile['reading_speed_wpm']}\n"
            f"  topic_affinity: {profile['topic_affinity']}\n"
            "Recent activity:\n"
            f"  queries_last_hour_feature: {profile['queries_last_hour']}\n"
            f"  live_recent_queries: {list(self.recent_queries[user_id])}\n"
            f"Top memories:\n{memory_block}\n"
            f"Assembler note: Trả lời bằng {profile['preferred_language']}, ưu tiên "
            f"{profile['topic_affinity']} nếu query là recommendation."
        )

    @staticmethod
    def _make_embedder():
        if TextEmbedding:
            try:
                return TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
            except Exception:
                pass
        return HashEmbedder()

    @staticmethod
    def _chunk(text: str, max_words: int = 85) -> list[str]:
        parts = [p.strip() for p in re.split(r"\n\s*\n|(?<=[.!?])\s+", text) if p.strip()]
        chunks = []
        for part in parts:
            words = part.split()
            chunks.extend(
                [part] if len(words) <= max_words
                else [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]
            )
        return chunks

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    @staticmethod
    def _infer_topic(text: str) -> str:
        lower = text.lower()
        topics = {
            "cloud": ["kubernetes", "autoscaling", "hạ tầng", "cloud", "lambda"],
            "security": ["security", "oauth", "encryption", "zero-trust", "bảo mật"],
            "ai_ml": ["rag", "embedding", "llm", "hallucination", "bert"],
            "database": ["postgresql", "mongodb", "redis", "transaction"],
            "devops": ["terraform", "ci/cd", "prometheus", "rollback"],
        }
        return next((t for t, keys in topics.items() if any(k in lower for k in keys)), "general")

    def _rebuild_bm25(self) -> None:
        tokenized = [self._tokens(m["text"]) for m in self.memories]
        self.bm25 = BM25Okapi(tokenized) if BM25Okapi and tokenized else None

    def _keyword_search(self, query: str, user_id: str, top_k: int) -> list[dict]:
        q_tokens = self._tokens(query)
        if self.bm25:
            scores = self.bm25.get_scores(q_tokens)
        else:
            q = Counter(q_tokens)
            scores = [sum((Counter(self._tokens(m["text"])) & q).values()) for m in self.memories]
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
        return [self.memories[i] for i in ranked if self.memories[i]["user_id"] == user_id][:top_k]

    def _vector_search(self, query: str, user_id: str, top_k: int) -> list[dict]:
        if not self.memories:
            return []
        q_vec = list(next(self.embedder.embed([query])))
        if self.qdrant:
            result = self.qdrant.query_points(
                collection_name=COLLECTION,
                query=q_vec,
                query_filter=Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]),
                limit=top_k,
            )
            return [dict(p.payload) for p in result.points]
        scored = [
            (sum(a * b for a, b in zip(q_vec, vector)), memory)
            for vector, memory in zip(self.vectors, self.memories)
            if memory["user_id"] == user_id
        ]
        return [m for _, m in sorted(scored, key=lambda kv: -kv[0])[:top_k]]

    @staticmethod
    def _rrf(keyword_hits: list[dict], vector_hits: list[dict], top_k: int, rrf_k: int = 60) -> list[dict]:
        scores, by_id = {}, {}
        for hits in (keyword_hits, vector_hits):
            for rank, hit in enumerate(hits, 1):
                memory_id = hit["memory_id"]
                scores[memory_id] = scores.get(memory_id, 0.0) + 1.0 / (rrf_k + rank)
                by_id.setdefault(memory_id, hit)
        return [by_id[mid] for mid, _ in sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]]

    def _get_profile(self, user_id: str) -> dict:
        try:
            from feast import FeatureStore

            values = FeatureStore(repo_path=str(FEAST_DIR)).get_online_features(
                features=[
                    "user_profile_features:reading_speed_wpm",
                    "user_profile_features:preferred_language",
                    "user_profile_features:topic_affinity",
                    "query_velocity_features:queries_last_hour",
                    "query_velocity_features:distinct_topics_24h",
                ],
                entity_rows=[{"user_id": user_id}],
            ).to_dict()
            return {k: values[k][0] for k in [
                "reading_speed_wpm", "preferred_language", "topic_affinity",
                "queries_last_hour", "distinct_topics_24h",
            ]}
        except Exception:
            return {
                "reading_speed_wpm": 187,
                "preferred_language": "vi",
                "topic_affinity": "cloud",
                "queries_last_hour": len(self.recent_queries[user_id]),
                "distinct_topics_24h": 3,
            }
