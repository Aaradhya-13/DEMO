"""
server/schemas.py
--------------------
Pydantic models shared by the FastAPI REST endpoints and WebSocket streams.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class DistressIngestResponse(BaseModel):
    id: int
    latitude: float
    longitude: float
    urgency: str
    victim_count: int
    need_code: str
    timestamp: int
    device_id_hash: int
    battery_level_0_15: int
    signal_strength_0_15: int
    received_via: str
    received_at: datetime


class ActiveDistressRecord(BaseModel):
    id: int
    latitude: float
    longitude: float
    urgency: str
    victim_count: int
    need_code: str
    timestamp: int
    device_id_hash: int
    battery_level_0_15: int
    signal_strength_0_15: int
    received_via: str
    received_at: datetime
    resolved: bool = False


class ActiveDistressList(BaseModel):
    count: int
    records: list[ActiveDistressRecord]


class FloodMaskRequest(BaseModel):
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    reference_time: Optional[datetime] = None


class FloodMaskResponse(BaseModel):
    geojson: dict
    otsu_threshold_db: float
    aoi_bbox: tuple[float, float, float, float]
    baseline_window: tuple[str, str]
    postflood_window: tuple[str, str]
    water_fraction: float


class RouteRequest(BaseModel):
    origin_lat: float = Field(..., ge=-90.0, le=90.0)
    origin_lon: float = Field(..., ge=-180.0, le=180.0)
    destination_lat: float = Field(..., ge=-90.0, le=90.0)
    destination_lon: float = Field(..., ge=-180.0, le=180.0)
    mode: str = Field(default="pedestrian")
    flood_geojson: dict

    @field_validator("mode")
    @classmethod
    def _validate_mode(cls, v: str) -> str:
        if v not in {"boat", "pedestrian"}:
            raise ValueError("mode must be 'boat' or 'pedestrian'")
        return v


class RouteResponse(BaseModel):
    mode: str
    coordinates: list[tuple[float, float]]
    distance_meters: float
    estimated_duration_minutes: float
    intersects_flood_polygon: bool


class VoiceTriageResponse(BaseModel):
    transcript: str
    language: str
    language_probability: float
    location_query: Optional[str]
    headcount: int
    urgency_level: str
    need_type: str
    triage_score: float
    resolved_coordinates: Optional[dict]


class WebSocketDistressEvent(BaseModel):
    event_type: str = "distress_update"
    payload: ActiveDistressRecord
