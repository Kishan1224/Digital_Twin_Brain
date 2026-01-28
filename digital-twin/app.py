from __future__ import annotations

import os
from typing import Any, Dict, List

import streamlit as st

from dt_types import TwinMode
from memory_store import MemoryStore
from twin_logic import (
    classify_facts_heuristic,
    parse_ingestion_json,
    try_llm_or_fallback,
)


APP_TITLE = "AI Personal Digital Twin (Local-Only)"


def _init_state() -> None:
    st.session_state.setdefault("chat", [])  # list[dict(role, content, mode)]


def _get_store() -> MemoryStore:
    data_dir = os.path.join(os.path.dirname(__file__), "data", "chroma")
    return MemoryStore(persist_dir=data_dir)


def _sidebar() -> Dict[str, Any]:
    st.sidebar.header("Settings")

    mode = st.sidebar.selectbox(
        "Active mode",
        options=[m.value for m in [TwinMode.MIRROR, TwinMode.ADVISOR, TwinMode.PREDICTOR, TwinMode.REFLECTOR]],
        index=0,
    )

    st.sidebar.subheader("Local LLM (Optional)")
    use_ollama = st.sidebar.toggle("Use Ollama (localhost)", value=False)
    ollama_model = st.sidebar.text_input("Ollama model", value="llama3.1", disabled=not use_ollama)
    ollama_base_url = st.sidebar.text_input("Ollama base URL", value="http://localhost:11434", disabled=not use_ollama)

    st.sidebar.subheader("Behavior")
    explain_why = st.sidebar.toggle("Explain WHY (Advisor/Predictor/Reflector)", value=True)
    k = st.sidebar.slider("Memory retrieval: top K", min_value=2, max_value=20, value=8, step=1)

    st.sidebar.divider()
    if st.sidebar.button("Clear chat (UI only)"):
        st.session_state.chat = []

    return {
        "mode": TwinMode(mode),
        "use_ollama": use_ollama,
        "ollama_model": ollama_model.strip() or "llama3.1",
        "ollama_base_url": ollama_base_url.strip() or "http://localhost:11434",
        "explain_why": explain_why,
        "k": int(k),
    }


def _render_chat_tab(store: MemoryStore, settings: Dict[str, Any]) -> None:
    st.subheader("Chat with Twin")
    st.caption("Tip: ingest some notes first (Memory Timeline tab) for better personalization.")

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            if msg.get("mode"):
                st.markdown(f"**MODE:** `{msg['mode']}`")
            st.write(msg["content"])

    user_text = st.chat_input("Message your Twin…")
    if not user_text:
        return

    # Store user message as a memory (local).
    store.add_text(
        user_text,
        source="chat",
        kind="chat_user",
        tags=["chat"],
    )

    st.session_state.chat.append({"role": "user", "content": user_text, "mode": settings["mode"].value})

    memories = store.search(user_text, k=settings["k"])
    reply = try_llm_or_fallback(
        settings["mode"],
        user_text,
        memories,
        use_ollama=settings["use_ollama"],
        ollama_model=settings["ollama_model"],
        ollama_base_url=settings["ollama_base_url"],
        explain_why=settings["explain_why"],
    )

    # Store assistant reply as memory (local).
    store.add_text(
        reply,
        source="chat",
        kind="chat_twin",
        tags=["chat", settings["mode"].value.lower()],
    )

    st.session_state.chat.append({"role": "assistant", "content": reply, "mode": settings["mode"].value})
    st.rerun()


def _render_memory_tab(store: MemoryStore, settings: Dict[str, Any]) -> None:
    st.subheader("Memory Timeline")
    st.caption("All data stays on this machine in `./data/chroma/`.")

    with st.expander("Ingest notes / habits / history (Training the Twin)", expanded=True):
        src = st.selectbox("Source label", options=["notes", "journal", "tasks", "chat_export", "other"], index=0)
        raw = st.text_area("Paste text to ingest", height=180, placeholder="Paste notes, habits, preferences, goals…")

        col1, col2 = st.columns(2)
        with col1:
            use_llm_ingest = st.toggle("Use Ollama to extract categories (optional)", value=False)
        with col2:
            ingest_btn = st.button("Ingest into memory", type="primary", disabled=not raw.strip())

        if ingest_btn:
            # Always store the raw text as-is.
            store.add_text(raw, source=src, kind="raw_ingest", tags=["ingest", "raw"])

            facts = None
            if use_llm_ingest and settings["use_ollama"]:
                from twin_logic import build_prompt, _ollama_generate  # local import to keep top clean

                prompt = build_prompt(TwinMode.MEMORY_INGESTION, raw, [], explain_why=False)
                out = _ollama_generate(prompt, model=settings["ollama_model"], base_url=settings["ollama_base_url"])
                facts = parse_ingestion_json(out)

            if facts is None:
                facts = classify_facts_heuristic(raw)

            ids = store.add_ingested_facts(facts, source=src)
            st.success(f"Stored {len(ids)} memory items (including structured extraction).")

    st.divider()

    st.markdown("**Recent memories**")
    limit = st.slider("Show last N", 10, 200, 50, 10)
    items = store.recent(limit=limit)

    for it in items:
        meta = it.meta or {}
        title = f"{meta.get('created_at', '')} • {meta.get('kind', '')} • {meta.get('source', '')}"
        with st.expander(title, expanded=False):
            st.write(it.text)
            st.json(meta)


def _render_predictor_tab(store: MemoryStore, settings: Dict[str, Any]) -> None:
    st.subheader("Decision Predictor")
    st.caption("Uses your local memory + selected model (optional).")

    q = st.text_area("What decision are you facing?", height=120, placeholder="e.g., Should I switch projects or stick with the current one?")
    if st.button("Predict", type="primary", disabled=not q.strip()):
        memories = store.search(q, k=settings["k"])
        out = try_llm_or_fallback(
            TwinMode.PREDICTOR,
            q,
            memories,
            use_ollama=settings["use_ollama"],
            ollama_model=settings["ollama_model"],
            ollama_base_url=settings["ollama_base_url"],
            explain_why=True,
        )
        st.write(out)


def _render_suggestions_tab(store: MemoryStore, settings: Dict[str, Any]) -> None:
    st.subheader("Daily Suggestions")
    st.caption("Small proactive suggestions derived from your stored goals/habits (no hallucinations).")

    # Pull some "goal-ish" and "habit-ish" memories for gentle suggestions.
    goals = store.search("goal", k=8, where={"kind": "goals"})
    habits = store.search("habit", k=8, where={"kind": "habits"})

    st.markdown("**Today’s suggestions**")
    if not goals and not habits:
        st.info("No goals/habits found yet. Ingest some notes in the Memory Timeline tab.")
        return

    if habits:
        st.markdown("**Habits**")
        for h in habits[:5]:
            st.write(f"- Keep it easy: do 5 minutes of: {h.text}")

    if goals:
        st.markdown("**Goals**")
        for g in goals[:5]:
            st.write(f"- Move this forward with one tiny action: {g.text}")


def _render_privacy_tab() -> None:
    st.subheader("Privacy Dashboard")
    st.markdown(
        """
**Local-only by default**
- Your memories are stored on disk under `./data/chroma/`.
- This app does not upload data anywhere.

**Optional local model**
- If you enable Ollama, prompts are sent to `localhost` only.
- If Ollama is not running, the app falls back to a non-LLM heuristic responder.

**How to verify**
- You can unplug the network and the app still works (memory + heuristic mode).
"""
    )


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    _init_state()
    store = _get_store()
    settings = _sidebar()

    tabs = st.tabs(
        [
            "Chat with Twin",
            "Memory Timeline",
            "Decision Predictor",
            "Daily Suggestions",
            "Privacy Dashboard",
        ]
    )

    with tabs[0]:
        _render_chat_tab(store, settings)
    with tabs[1]:
        _render_memory_tab(store, settings)
    with tabs[2]:
        _render_predictor_tab(store, settings)
    with tabs[3]:
        _render_suggestions_tab(store, settings)
    with tabs[4]:
        _render_privacy_tab()


if __name__ == "__main__":
    main()

