from __future__ import annotations

import json
import re
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import requests

from dt_types import IngestedFacts, MemoryItem, TwinMode


def _bullets(items: List[str]) -> str:
    return "\n".join([f"- {x}" for x in items if str(x).strip()])


def classify_facts_heuristic(text: str) -> IngestedFacts:
    """
    Lightweight (non-LLM) classifier for the ingestion prompt.
    It tries to extract lines into the requested buckets without hallucinating.
    """
    t = (text or "").strip()
    lines = [ln.strip("-• \t") for ln in t.splitlines()]
    lines = [ln for ln in lines if ln and len(ln) > 2]

    def grab(prefixes: Tuple[str, ...]) -> List[str]:
        out: List[str] = []
        for ln in lines:
            low = ln.lower()
            if any(low.startswith(p) for p in prefixes):
                out.append(ln.split(":", 1)[-1].strip() if ":" in ln else ln)
        return out

    # If user doesn't label things, we avoid "inventing" and keep most in goals/values empty.
    preferences = grab(("preference", "likes", "like ", "prefers", "favorite", "favourite"))
    habits = grab(("habit", "usually", "often", "daily", "every day", "each day", "routine"))
    values = grab(("value", "care about", "important", "principle"))
    skills = grab(("skill", "good at", "experienced", "expert", "know ", "can "))
    goals = grab(("goal", "want to", "aim", "plan to", "trying to"))
    communication_style = grab(("tone", "style", "communication", "writes", "speaks"))

    # Fallback: if nothing was labeled, treat the whole thing as generic "notes".
    if not any([preferences, habits, values, skills, goals, communication_style]) and t:
        goals = [t[:5000]]

    # Deduplicate while preserving order.
    def dedup(xs: List[str]) -> List[str]:
        seen = set()
        out = []
        for x in xs:
            x2 = re.sub(r"\s+", " ", x).strip()
            if x2 and x2 not in seen:
                seen.add(x2)
                out.append(x2)
        return out

    return IngestedFacts(
        preferences=dedup(preferences),
        habits=dedup(habits),
        values=dedup(values),
        skills=dedup(skills),
        goals=dedup(goals),
        communication_style=dedup(communication_style),
    )


def _ollama_generate(
    prompt: str,
    *,
    model: str,
    base_url: str = "http://localhost:11434",
    timeout_s: int = 120,
) -> str:
    """
    Minimal Ollama HTTP client (no extra dependency).
    """
    url = f"{base_url.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3},
    }
    r = requests.post(url, json=payload, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    return (data.get("response") or "").strip()


def build_prompt(
    mode: TwinMode,
    user_message: str,
    memories: List[MemoryItem],
    *,
    explain_why: bool,
) -> str:
    parts: List[str] = []
    for i, m in enumerate(memories):
        score = 0.0 if m.score is None else float(m.score)
        parts.append(
            (
                f"[MEMORY {i+1} | score={score:.2f} | kind={m.meta.get('kind')} | "
                f"created_at={m.meta.get('created_at')}]\n{m.text}"
            )
        )
    mem_block = "\n\n".join(parts)

    # Keep it extremely explicit: no hallucinated preferences, use retrieved memories only.
    base = f"""You are an AI Personal Digital Twin.

Rules:
- Do NOT invent user preferences. Only use the provided memories.
- If memories are insufficient, ask a clarifying question.
- Keep privacy: assume everything is local-only.
- Output must start with: MODE: {mode.value}

Relevant memories (may be empty):
{mem_block or "(none)"}

User message:
{user_message}
"""

    if mode == TwinMode.MIRROR:
        base += """
Task:
- Reply as the user would, matching likely tone and decision style inferred ONLY from memories.
"""
    elif mode == TwinMode.ADVISOR:
        base += """
Task:
- Suggest better decisions than the user’s default.
- Keep it proactive but not intrusive.
"""
    elif mode == TwinMode.PREDICTOR:
        base += """
Task:
- Predict the user’s most likely next decision.
- Provide two alternative decisions.
- Recommend the best decision aligned with long-term goals/values found in memories.
Output format:
MODE: PREDICTOR
1) Most likely next decision: ...
2) Alternative A: ...
3) Alternative B: ...
4) Recommendation: ...
"""
    elif mode == TwinMode.REFLECTOR:
        base += """
Task:
- Summarize behavior patterns over time based ONLY on memories.
"""
    elif mode == TwinMode.MEMORY_INGESTION:
        base += """
Task:
- Extract facts into these categories:
  Preferences, Habits, Values, Skills, Goals, Communication style
- Return STRICT JSON with those keys, each value a list of strings.
- Do not include any other keys.
"""

    if explain_why and mode in (TwinMode.ADVISOR, TwinMode.PREDICTOR, TwinMode.REFLECTOR):
        base += "\nAlso include a short WHY section explaining your reasoning, referencing memories.\n"

    return base


def respond_heuristic(
    mode: TwinMode,
    user_message: str,
    memories: List[MemoryItem],
    *,
    explain_why: bool,
) -> str:
    """
    Non-LLM fallback that still:
    - uses retrieved memories (quotes/snippets)
    - avoids inventing preferences
    - follows mode formatting
    """
    top = memories[:5]
    mem_snips = []
    for m in top:
        snip = m.text.strip().replace("\n", " ")
        if len(snip) > 140:
            snip = snip[:140].rstrip() + "…"
        mem_snips.append(f"- ({m.meta.get('kind')}, {m.meta.get('created_at')}) {snip}")

    if mode == TwinMode.PREDICTOR:
        out = [
            "MODE: PREDICTOR",
            f"1) Most likely next decision: Ask for a concrete next step or a runnable command based on: {user_message[:120].strip()}",
            "2) Alternative A: Share more context/constraints (time, tools, success criteria) before acting.",
            "3) Alternative B: Start with a small prototype and iterate based on what breaks.",
            "4) Recommendation: Start with a small prototype, then tighten based on constraints you care about most.",
        ]
        if explain_why:
            out.append("")
            out.append("WHY:")
            out.append("I’m missing strong personal signals in memory; a prototype-first approach reduces risk without assuming preferences.")
        if mem_snips:
            out.append("")
            out.append("Memory signals used:")
            out.append("\n".join(mem_snips))
        return "\n".join(out)

    if mode == TwinMode.ADVISOR:
        out = ["MODE: ADVISOR"]
        if not memories:
            out.append("I don’t have enough personal history to tailor this yet. What’s your top goal for the next 2 weeks (health, money, learning, career, relationships)?")
            return "\n".join(out)
        out.append("Here’s a higher-leverage next step: turn this into one concrete outcome + deadline, then pick the smallest action that proves progress today.")
        if explain_why:
            out.append("")
            out.append("WHY:")
            out.append("This avoids guessing your preferences while still pushing toward measurable progress.")
        out.append("")
        out.append("Memory signals used:")
        out.append("\n".join(mem_snips) if mem_snips else "(none)")
        return "\n".join(out)

    if mode == TwinMode.REFLECTOR:
        out = ["MODE: REFLECTOR"]
        if not memories:
            out.append("No long-term memories yet. Ingest some notes/chats first, then I can summarize patterns.")
            return "\n".join(out)
        out.append("Patterns observed (from stored memories):")
        out.append("\n".join(mem_snips))
        if explain_why:
            out.append("")
            out.append("WHY:")
            out.append("These are directly derived from the most recent/high-scoring memories for your current query.")
        return "\n".join(out)

    # MIRROR default
    out = ["MODE: MIRROR"]
    if not memories:
        out.append("Need a bit more context from you first—what’s the exact outcome you want here?")
    else:
        out.append("Ok. Here’s what I’d do next: keep it simple, start with a minimal version, and expand after it works.")
        out.append("")
        out.append("Relevant context I’m using:")
        out.append("\n".join(mem_snips))
    return "\n".join(out)


def try_llm_or_fallback(
    mode: TwinMode,
    user_message: str,
    memories: List[MemoryItem],
    *,
    use_ollama: bool,
    ollama_model: str,
    ollama_base_url: str,
    explain_why: bool,
) -> str:
    if not use_ollama:
        return respond_heuristic(mode, user_message, memories, explain_why=explain_why)

    prompt = build_prompt(mode, user_message, memories, explain_why=explain_why)
    try:
        return _ollama_generate(prompt, model=ollama_model, base_url=ollama_base_url)
    except Exception:
        return respond_heuristic(mode, user_message, memories, explain_why=explain_why)


def parse_ingestion_json(text: str) -> Optional[IngestedFacts]:
    """
    Parse STRICT JSON from LLM ingestion mode. Returns None if parse fails.
    """
    t = (text or "").strip()
    # Try to extract a JSON object if model wrapped it.
    m = re.search(r"\{[\s\S]*\}", t)
    if m:
        t = m.group(0)
    try:
        data = json.loads(t)
        return IngestedFacts(
            preferences=list(map(str, data.get("Preferences", data.get("preferences", [])))),
            habits=list(map(str, data.get("Habits", data.get("habits", [])))),
            values=list(map(str, data.get("Values", data.get("values", [])))),
            skills=list(map(str, data.get("Skills", data.get("skills", [])))),
            goals=list(map(str, data.get("Goals", data.get("goals", [])))),
            communication_style=list(
                map(str, data.get("Communication style", data.get("communication_style", data.get("communicationStyle", []))))
            ),
        )
    except Exception:
        return None

