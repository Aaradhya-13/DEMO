"""
edge_nlp/entity_extractor.py
-------------------------------
Turns raw ASR transcript text into a structured distress JSON object:

    {
        "location_query": str | None,
        "headcount": int,
        "urgency_level": "CRITICAL" | "HIGH" | "MEDIUM" | "LOW",
        "need_type": "BOAT_EVACUATION" | "MEDICAL" | "FOOD_WATER" | "RESCUE",
        "triage_score": float,   # 0-100, dynamically computed
        "resolved_coordinates": {"lat": float, "lon": float} | None
    }

Urgency and need classification use a zero-shot pipeline (candidate labels
pulled directly from the `DistressUrgency` / `DistressNeed` enums, so there
is a single source of truth shared with the binary protocol). Headcount and
the raw location phrase are extracted with a dynamic, language-agnostic
heuristic (numeral parsing + preposition-anchored phrase capture) rather
than a hardcoded gazetteer. If the device has no GPS fix, the extracted
location phrase is resolved to (lat, lon) via a Nominatim/OSM geocoder
(configurable to point at a self-hosted / offline-reachable instance).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Optional

from core.config import settings
from core.logger import get_logger
from packet.binary_protocol import DistressUrgency, DistressNeed

logger = get_logger(__name__)

# Word-form numerals so headcount extraction isn't limited to digit strings
# spoken/transcribed as words (works across the small set of EN loanwords
# commonly mixed into transcribed Indic-language distress calls).
_WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
}

# Prepositions that typically anchor a location phrase in transcribed speech.
_LOCATION_ANCHORS = ("near", "at", "in", "behind", "beside", "opposite", "next to")

_HEADCOUNT_HINTS = ("people", "person", "family", "members", "us", "children", "elderly")


@dataclass(frozen=True)
class TriageResult:
    location_query: Optional[str]
    headcount: int
    urgency_level: str
    need_type: str
    triage_score: float
    resolved_coordinates: Optional[dict]

    def to_dict(self) -> dict:
        return asdict(self)


class EntityExtractor:
    def __init__(self, ner_model_name: Optional[str] = None):
        self.model_name = ner_model_name or settings.ner_model_name
        self._zero_shot_pipeline = None

    def _load_pipeline(self):
        if self._zero_shot_pipeline is not None:
            return self._zero_shot_pipeline
        from transformers import pipeline

        logger.info("Loading zero-shot classification pipeline", extra={"context": {"model": self.model_name}})
        self._zero_shot_pipeline = pipeline("zero-shot-classification", model=self.model_name)
        return self._zero_shot_pipeline

    # ---------------- Headcount ----------------

    def _extract_headcount(self, text: str) -> int:
        lowered = text.lower()

        digit_matches = re.findall(r"\b(\d{1,3})\b", lowered)
        for match in digit_matches:
            value = int(match)
            window_start = max(0, lowered.find(match) - 20)
            window_end = min(len(lowered), lowered.find(match) + len(match) + 20)
            window = lowered[window_start:window_end]
            if any(hint in window for hint in _HEADCOUNT_HINTS) or 0 < value <= 50:
                return value

        for word, value in _WORD_NUMBERS.items():
            if re.search(rf"\b{word}\b", lowered):
                window_idx = lowered.find(word)
                window = lowered[max(0, window_idx - 20): window_idx + len(word) + 20]
                if any(hint in window for hint in _HEADCOUNT_HINTS):
                    return value

        return 1  # default: at least the caller themself

    # ---------------- Location phrase ----------------

    def _extract_location_query(self, text: str) -> Optional[str]:
        lowered = text.lower()
        for anchor in _LOCATION_ANCHORS:
            pattern = rf"\b{re.escape(anchor)}\s+([a-zA-Z0-9,\-\s]{{3,60}}?)(?:[.,]|$)"
            match = re.search(pattern, lowered)
            if match:
                phrase = match.group(1).strip()
                if phrase:
                    return phrase
        return None

    # ---------------- Urgency / need (zero-shot) ----------------

    def _classify(self, text: str, labels: list[str]) -> tuple[str, float]:
        clf = self._load_pipeline()
        result = clf(text, candidate_labels=labels, multi_label=False)
        top_label = result["labels"][0]
        top_score = float(result["scores"][0])
        return top_label, top_score

    # ---------------- Triage scoring ----------------

    @staticmethod
    def _compute_triage_score(
        urgency: DistressUrgency,
        need: DistressNeed,
        headcount: int,
        urgency_confidence: float,
    ) -> float:
        """
        Deterministic, dynamically-weighted 0-100 triage score.
        Higher urgency, medical need, and larger group size raise the score;
        classifier confidence tempers how much the urgency component
        contributes (low-confidence classification pulls the score toward
        the middle rather than swinging it to an extreme).
        """
        urgency_weight = {
            DistressUrgency.LOW: 10,
            DistressUrgency.MEDIUM: 40,
            DistressUrgency.HIGH: 70,
            DistressUrgency.CRITICAL: 95,
        }[urgency]

        need_weight = {
            DistressNeed.MEDICAL: 20,
            DistressNeed.RESCUE: 15,
            DistressNeed.BOAT_EVACUATION: 10,
            DistressNeed.FOOD_WATER: 5,
        }[need]

        headcount_weight = min(headcount, 20) * 0.75

        confidence_adjusted_urgency = urgency_weight * urgency_confidence + 30 * (1 - urgency_confidence)

        raw_score = confidence_adjusted_urgency + need_weight + headcount_weight
        return round(min(raw_score, 100.0), 2)

    # ---------------- Geocoding fallback ----------------

    def resolve_location(self, location_query: str) -> Optional[dict]:
        """
        Resolve a spoken/free-text landmark string to (lat, lon) via a
        Nominatim-compatible geocoder. `NOMINATIM_BASE_URL` can point at a
        self-hosted / LAN-local instance for offline field deployments.
        """
        if not location_query:
            return None
        try:
            from geopy.geocoders import Nominatim

            geolocator = Nominatim(
                user_agent=settings.nominatim_user_agent,
                domain=settings.nominatim_base_url.replace("https://", "").replace("http://", ""),
                scheme="https" if settings.nominatim_base_url.startswith("https") else "http",
            )
            location = geolocator.geocode(location_query, timeout=10)
            if location is None:
                logger.warning("Geocoder returned no match", extra={"context": {"query": location_query}})
                return None
            return {"lat": location.latitude, "lon": location.longitude}
        except Exception as exc:
            logger.error(
                "Geocoding failed",
                extra={"context": {"query": location_query, "error": str(exc)}},
            )
            return None

    # ---------------- Public API ----------------

    def extract(self, text: str, resolve_coordinates: bool = True) -> TriageResult:
        if not text or not text.strip():
            raise ValueError("Cannot extract entities from empty transcript text")

        urgency_labels = [level.name for level in DistressUrgency]
        need_labels = [need.name for need in DistressNeed]

        urgency_label, urgency_confidence = self._classify(text, urgency_labels)
        need_label, _ = self._classify(text, need_labels)

        urgency = DistressUrgency[urgency_label]
        need = DistressNeed[need_label]

        headcount = self._extract_headcount(text)
        location_query = self._extract_location_query(text)

        triage_score = self._compute_triage_score(urgency, need, headcount, urgency_confidence)

        resolved_coordinates = None
        if resolve_coordinates and location_query:
            resolved_coordinates = self.resolve_location(location_query)

        result = TriageResult(
            location_query=location_query,
            headcount=headcount,
            urgency_level=urgency.name,
            need_type=need.name,
            triage_score=triage_score,
            resolved_coordinates=resolved_coordinates,
        )
        logger.info("Triage extraction complete", extra={"context": result.to_dict()})
        return result
