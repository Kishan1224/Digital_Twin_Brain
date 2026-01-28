from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class TwinMode(str, Enum):
    MIRROR = "MIRROR"
    ADVISOR = "ADVISOR"
    PREDICTOR = "PREDICTOR"
    REFLECTOR = "REFLECTOR"
    MEMORY_INGESTION = "MEMORY_INGESTION"


@dataclass
class MemoryItem:
    id: str
    text: str
    meta: Dict[str, Any]
    score: Optional[float] = None


@dataclass
class IngestedFacts:
    preferences: List[str]
    habits: List[str]
    values: List[str]
    skills: List[str]
    goals: List[str]
    communication_style: List[str]

