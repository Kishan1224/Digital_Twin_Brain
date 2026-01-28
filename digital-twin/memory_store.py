from __future__ import annotations

import math
import json
import os
import sqlite3
import time
import uuid
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

from dt_types import IngestedFacts, MemoryItem


DEFAULT_COLLECTION = "twin_memories_v1"


def _now_iso() -> str:
    # Keep it simple and stable for sorting/display.
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def _tokenize(text: str) -> List[str]:
    # Very lightweight tokenizer; avoids heavy deps.
    text = (text or "").lower()
    buf = []
    word = []
    for ch in text:
        if ch.isalnum() or ch in ("_", "-"):
            word.append(ch)
        else:
            if word:
                buf.append("".join(word))
                word = []
    if word:
        buf.append("".join(word))
    # Drop extremely short tokens that add noise.
    return [t for t in buf if len(t) >= 2]


def _hash_embed(text: str, *, dim: int = 1024) -> List[float]:
    """
    Pure-Python "embedding" using hashing trick + L2 normalization.
    Not as good as SentenceTransformers, but works anywhere and stays local-only.
    """
    vec = [0.0] * dim
    toks = _tokenize(text)
    for tok in toks:
        # Deterministic-ish index per token.
        h = hash(tok)
        idx = h % dim
        sign = 1.0 if (h & 1) == 0 else -1.0
        vec[idx] += sign

    # L2 normalize
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        inv = 1.0 / norm
        vec = [v * inv for v in vec]
    return vec


def _cosine(a: List[float], b: List[float]) -> float:
    # Vectors are normalized, so cosine is just dot.
    return float(sum(x * y for x, y in zip(a, b)))


class MemoryStore:
    """
    Local-only memory:
    - Default backend: SQLite + pure-Python hashed embeddings (portable)
    - Optional backend: Chroma + SentenceTransformers (if installed)
    """

    def __init__(
        self,
        persist_dir: str,
        collection_name: str = DEFAULT_COLLECTION,
        embedding_model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        os.makedirs(persist_dir, exist_ok=True)

        self._persist_dir = persist_dir
        self._collection_name = collection_name
        self._embedding_model_name = embedding_model_name

        self._backend = "sqlite"
        self._dim = 1024

        # Try optional "better" backend if available.
        self._chroma = None
        self._chroma_collection = None
        self._st_model = None
        try:
            import chromadb  # type: ignore
            from chromadb.api.models.Collection import Collection  # type: ignore
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._st_model = SentenceTransformer(embedding_model_name)
            self._chroma = chromadb.PersistentClient(path=persist_dir)
            self._chroma_collection = self._chroma.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            self._backend = "chroma"
        except Exception:
            self._backend = "sqlite"

        # SQLite backend init (always available as fallback).
        self._db_path = os.path.join(persist_dir, "memories.sqlite3")
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
              id TEXT PRIMARY KEY,
              created_at TEXT NOT NULL,
              source TEXT NOT NULL,
              kind TEXT NOT NULL,
              tags_json TEXT NOT NULL,
              meta_json TEXT NOT NULL,
              text TEXT NOT NULL,
              vector_json TEXT NOT NULL
            )
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_kind ON memories(kind)")
        self._conn.commit()

    @property
    def backend(self) -> str:
        return self._backend

    def add_text(
        self,
        text: str,
        *,
        source: str,
        kind: str,
        tags: Optional[List[str]] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        text = (text or "").strip()
        if not text:
            raise ValueError("Cannot add empty text to memory.")

        mem_id = str(uuid.uuid4())
        meta: Dict[str, Any] = {
            "created_at": _now_iso(),
            "source": source,
            "kind": kind,  # e.g., "note", "chat", "preference", ...
            "tags": tags or [],
        }
        if extra_meta:
            meta.update(extra_meta)

        if self._backend == "chroma" and self._chroma_collection is not None and self._st_model is not None:
            emb = self._st_model.encode([text], normalize_embeddings=True).tolist()
            self._chroma_collection.add(
                ids=[mem_id],
                documents=[text],
                metadatas=[meta],
                embeddings=emb,
            )
        else:
            vec = _hash_embed(text, dim=self._dim)
            self._conn.execute(
                """
                INSERT OR REPLACE INTO memories
                (id, created_at, source, kind, tags_json, meta_json, text, vector_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    mem_id,
                    meta["created_at"],
                    source,
                    kind,
                    json.dumps(tags or [], ensure_ascii=False),
                    json.dumps(meta, ensure_ascii=False),
                    text,
                    json.dumps(vec),
                ),
            )
            self._conn.commit()
        return mem_id

    def add_ingested_facts(self, facts: IngestedFacts, *, source: str) -> List[str]:
        ids: List[str] = []
        payload = asdict(facts)
        for kind, items in payload.items():
            for item in items:
                ids.append(
                    self.add_text(
                        item,
                        source=source,
                        kind=kind,
                        tags=["ingested"],
                    )
                )
        # Also store the raw structured extraction for future inspection/audit.
        ids.append(
            self.add_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                source=source,
                kind="ingestion_payload_json",
                tags=["ingested", "structured"],
            )
        )
        return ids

    def search(self, query: str, *, k: int = 8, where: Optional[Dict[str, Any]] = None) -> List[MemoryItem]:
        query = (query or "").strip()
        if not query:
            return []

        if self._backend == "chroma" and self._chroma_collection is not None and self._st_model is not None:
            emb = self._st_model.encode([query], normalize_embeddings=True).tolist()
            res = self._chroma_collection.query(
                query_embeddings=emb,
                n_results=max(1, k),
                where=where,
                include=["documents", "metadatas", "distances", "ids"],
            )
            ids = (res.get("ids") or [[]])[0]
            docs = (res.get("documents") or [[]])[0]
            metas = (res.get("metadatas") or [[]])[0]
            dists = (res.get("distances") or [[]])[0]

            out: List[MemoryItem] = []
            for mid, doc, meta, dist in zip(ids, docs, metas, dists):
                score = float(max(0.0, 1.0 - dist)) if dist is not None else None
                out.append(MemoryItem(id=mid, text=doc, meta=meta or {}, score=score))
            return out

        # SQLite fallback: brute-force cosine over stored vectors.
        qv = _hash_embed(query, dim=self._dim)

        clauses = []
        params: List[Any] = []
        if where and "kind" in where:
            clauses.append("kind = ?")
            params.append(where["kind"])
        where_sql = ("WHERE " + " AND ".join(clauses)) if clauses else ""

        cur = self._conn.execute(
            f"SELECT id, text, meta_json, vector_json FROM memories {where_sql}",
            tuple(params),
        )

        scored: List[Tuple[float, MemoryItem]] = []
        for mid, text, meta_json, vec_json in cur.fetchall():
            meta = json.loads(meta_json) if meta_json else {}
            vec = json.loads(vec_json) if vec_json else []
            score = _cosine(qv, vec) if vec else 0.0
            scored.append((score, MemoryItem(id=mid, text=text, meta=meta, score=score)))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [mi for _, mi in scored[: max(1, k)]]

    def recent(self, *, limit: int = 50) -> List[MemoryItem]:
        if self._backend == "chroma" and self._chroma_collection is not None:
            res = self._chroma_collection.get(
                include=["documents", "metadatas", "ids"],
            )
            ids = res.get("ids") or []
            docs = res.get("documents") or []
            metas = res.get("metadatas") or []

            items: List[MemoryItem] = []
            for mid, doc, meta in zip(ids, docs, metas):
                items.append(MemoryItem(id=mid, text=doc, meta=meta or {}, score=None))

            def key_fn(mi: MemoryItem) -> Tuple[str, str]:
                return (str(mi.meta.get("created_at") or ""), mi.id)

            items.sort(key=key_fn, reverse=True)
            return items[: max(1, limit)]

        cur = self._conn.execute(
            "SELECT id, text, meta_json FROM memories ORDER BY created_at DESC LIMIT ?",
            (max(1, int(limit)),),
        )
        out: List[MemoryItem] = []
        for mid, text, meta_json in cur.fetchall():
            meta = json.loads(meta_json) if meta_json else {}
            out.append(MemoryItem(id=mid, text=text, meta=meta, score=None))
        return out

