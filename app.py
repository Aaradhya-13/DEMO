"""
server/app.py
---------------
FastAPI async backend for the flood rescue ecosystem.

Endpoints
---------
POST /api/v1/sos/ingest_binary     Accept a raw 40-byte-max binary distress
                                    packet (from BLE/LoRa gateways or direct
                                    device upload), validate + persist it,
                                    broadcast it to connected dashboards.
GET  /api/v1/sos/active_distress   List unresolved distress records.
POST /api/v1/sos/{id}/resolve      Mark a distress record resolved.
GET  /api/v1/sar/flood_mask        Run the GEE SAR pipeline for given coords,
                                    return a GeoJSON flood-water mask.
POST /api/v1/route                 Compute a hazard-aware A* route.
POST /api/v1/voice/triage          Upload a WAV recording; transcribe,
                                    extract structured triage JSON.
WS   /ws/distress                  Live push of newly ingested distress events.
"""

from __future__ import annotations

import asyncio
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiosqlite
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from core.config import settings
from core.logger import get_logger
from packet.binary_protocol import BinaryDistressProtocol, PacketValidationError
from server.schemas import (
    ActiveDistressList,
    ActiveDistressRecord,
    DistressIngestResponse,
    FloodMaskRequest,
    FloodMaskResponse,
    RouteRequest,
    RouteResponse,
    VoiceTriageResponse,
)

logger = get_logger(__name__)


def _sqlite_path_from_url(database_url: str) -> str:
    # Accepts URLs like "sqlite+aiosqlite:///./flood_rescue.db"
    if "///" in database_url:
        return database_url.split("///", 1)[1]
    raise ValueError(f"Unsupported DATABASE_URL format: {database_url}")


_DB_PATH = _sqlite_path_from_url(settings.database_url)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS distress_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    urgency TEXT NOT NULL,
    victim_count INTEGER NOT NULL,
    need_code TEXT NOT NULL,
    event_timestamp INTEGER NOT NULL,
    device_id_hash INTEGER NOT NULL,
    battery_level_0_15 INTEGER NOT NULL,
    signal_strength_0_15 INTEGER NOT NULL,
    received_via TEXT NOT NULL,
    received_at TEXT NOT NULL,
    resolved INTEGER NOT NULL DEFAULT 0
);
"""


class ConnectionManager:
    """Tracks active dashboard WebSocket clients and broadcasts JSON events."""

    def __init__(self):
        self._connections: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections.add(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._connections.discard(websocket)

    async def broadcast(self, message: dict) -> None:
        async with self._lock:
            dead = []
            for ws in self._connections:
                try:
                    await ws.send_json(message)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self._connections.discard(ws)


connection_manager = ConnectionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    Path(_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(_DB_PATH) as db:
        await db.execute(_CREATE_TABLE_SQL)
        await db.commit()
    logger.info("Database ready", extra={"context": {"path": _DB_PATH}})
    yield
    logger.info("Server shutting down")


app = FastAPI(
    title="Flood Rescue & Navigation API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _insert_distress_record(decoded: dict, received_via: str) -> ActiveDistressRecord:
    received_at = datetime.now(timezone.utc)
    async with aiosqlite.connect(_DB_PATH) as db:
        cursor = await db.execute(
            """
            INSERT INTO distress_events (
                latitude, longitude, urgency, victim_count, need_code,
                event_timestamp, device_id_hash, battery_level_0_15,
                signal_strength_0_15, received_via, received_at, resolved
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
            """,
            (
                decoded["latitude"],
                decoded["longitude"],
                decoded["urgency"],
                decoded["victim_count"],
                decoded["need_code"],
                decoded["timestamp"],
                decoded["device_id_hash"],
                decoded["battery_level_0_15"],
                decoded["signal_strength_0_15"],
                received_via,
                received_at.isoformat(),
            ),
        )
        await db.commit()
        new_id = cursor.lastrowid

    return ActiveDistressRecord(
        id=new_id,
        latitude=decoded["latitude"],
        longitude=decoded["longitude"],
        urgency=decoded["urgency"],
        victim_count=decoded["victim_count"],
        need_code=decoded["need_code"],
        timestamp=decoded["timestamp"],
        device_id_hash=decoded["device_id_hash"],
        battery_level_0_15=decoded["battery_level_0_15"],
        signal_strength_0_15=decoded["signal_strength_0_15"],
        received_via=received_via,
        received_at=received_at,
        resolved=False,
    )


@app.post("/api/v1/sos/ingest_binary", response_model=DistressIngestResponse)
async def ingest_binary(request: Request):
    """Accept a raw binary distress payload (application/octet-stream body)."""
    raw_bytes = await request.body()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Empty request body; expected raw binary packet.")

    try:
        decoded = BinaryDistressProtocol.unpack_payload(raw_bytes)
    except PacketValidationError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid packet: {exc}") from exc

    received_via = request.headers.get("X-Received-Via", "http")
    record = await _insert_distress_record(decoded, received_via)

    await connection_manager.broadcast(
        {"event_type": "distress_update", "payload": record.model_dump(mode="json")}
    )

    return DistressIngestResponse(**record.model_dump())


@app.get("/api/v1/sos/active_distress", response_model=ActiveDistressList)
async def active_distress(include_resolved: bool = False):
    query = "SELECT * FROM distress_events"
    if not include_resolved:
        query += " WHERE resolved = 0"
    query += " ORDER BY received_at DESC"

    async with aiosqlite.connect(_DB_PATH) as db:
        db.row_factory = sqlite3.Row
        cursor = await db.execute(query)
        rows = await cursor.fetchall()

    records = [
        ActiveDistressRecord(
            id=row["id"],
            latitude=row["latitude"],
            longitude=row["longitude"],
            urgency=row["urgency"],
            victim_count=row["victim_count"],
            need_code=row["need_code"],
            timestamp=row["event_timestamp"],
            device_id_hash=row["device_id_hash"],
            battery_level_0_15=row["battery_level_0_15"],
            signal_strength_0_15=row["signal_strength_0_15"],
            received_via=row["received_via"],
            received_at=datetime.fromisoformat(row["received_at"]),
            resolved=bool(row["resolved"]),
        )
        for row in rows
    ]
    return ActiveDistressList(count=len(records), records=records)


@app.post("/api/v1/sos/{distress_id}/resolve")
async def resolve_distress(distress_id: int):
    async with aiosqlite.connect(_DB_PATH) as db:
        cursor = await db.execute(
            "UPDATE distress_events SET resolved = 1 WHERE id = ?", (distress_id,)
        )
        await db.commit()
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail=f"Distress record {distress_id} not found.")
    return {"id": distress_id, "resolved": True}


@app.get("/api/v1/sar/flood_mask", response_model=FloodMaskResponse)
async def flood_mask(latitude: float, longitude: float, reference_time: Optional[str] = None):
    from geospatial.gee_sar_engine import SARFloodEngine, SARProcessingError, GEEInitializationError

    ref_dt = datetime.fromisoformat(reference_time) if reference_time else None

    try:
        engine = SARFloodEngine()
        result = await asyncio.to_thread(engine.generate_flood_mask, latitude, longitude, ref_dt)
    except GEEInitializationError as exc:
        raise HTTPException(status_code=503, detail=f"Earth Engine unavailable: {exc}") from exc
    except SARProcessingError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return FloodMaskResponse(
        geojson=result.geojson,
        otsu_threshold_db=result.otsu_threshold_db,
        aoi_bbox=result.aoi_bbox,
        baseline_window=result.baseline_window,
        postflood_window=result.postflood_window,
        water_fraction=result.water_fraction,
    )


@app.post("/api/v1/route", response_model=RouteResponse)
async def compute_route(payload: RouteRequest):
    from geospatial.hazard_router import HazardAwareRouter, RouteMode, RoutingError

    router_engine = HazardAwareRouter()
    try:
        result = await asyncio.to_thread(
            router_engine.route,
            payload.origin_lat,
            payload.origin_lon,
            payload.destination_lat,
            payload.destination_lon,
            payload.flood_geojson,
            RouteMode(payload.mode),
        )
    except RoutingError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return RouteResponse(
        mode=result.mode.value,
        coordinates=result.coordinates,
        distance_meters=result.distance_meters,
        estimated_duration_minutes=result.estimated_duration_minutes,
        intersects_flood_polygon=result.intersects_flood_polygon,
    )


@app.post("/api/v1/voice/triage", response_model=VoiceTriageResponse)
async def voice_triage(audio: UploadFile = File(...), language: Optional[str] = None):
    from edge_nlp.transcriber import WhisperTranscriber
    from edge_nlp.entity_extractor import EntityExtractor

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

    transcriber = WhisperTranscriber()
    extractor = EntityExtractor()

    transcription = await asyncio.to_thread(transcriber.transcribe, audio_bytes, language)
    if not transcription.text.strip():
        raise HTTPException(status_code=422, detail="Transcription produced no text; audio may be silent.")

    triage = await asyncio.to_thread(extractor.extract, transcription.text)

    return VoiceTriageResponse(
        transcript=transcription.text,
        language=transcription.language,
        language_probability=transcription.language_probability,
        location_query=triage.location_query,
        headcount=triage.headcount,
        urgency_level=triage.urgency_level,
        need_type=triage.need_type,
        triage_score=triage.triage_score,
        resolved_coordinates=triage.resolved_coordinates,
    )


@app.websocket("/ws/distress")
async def distress_websocket(websocket: WebSocket):
    await connection_manager.connect(websocket)
    try:
        while True:
            # Clients don't need to send anything; this keeps the connection alive
            # and detects disconnects.
            await websocket.receive_text()
    except WebSocketDisconnect:
        await connection_manager.disconnect(websocket)


@app.get("/healthz")
async def health_check():
    return {"status": "ok", "service": settings.service_name, "environment": settings.environment}
