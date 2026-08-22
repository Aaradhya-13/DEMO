# Flood Rescue & Navigation Ecosystem

Offline-resilient, end-to-end system for flood disaster response: voice-based
distress intake, satellite flood detection, hazard-aware routing, and
long-range mesh relay for zero-connectivity zones.

## Architecture

```
flood_rescue_system/
├── core/
│   ├── config.py                 Pydantic v2 BaseSettings — single source of config truth
│   └── logger.py                 Structured JSON logging
├── packet/
│   └── binary_protocol.py        27-byte (≤40B) CRC-16 protected distress packet
├── edge_nlp/
│   ├── audio_recorder.py         Mic capture (sounddevice, PyAudio fallback)
│   ├── transcriber.py            faster-whisper ASR, CUDA→CPU/INT8 fallback
│   └── entity_extractor.py       Zero-shot triage: urgency, need, headcount, location, geocoding
├── geospatial/
│   ├── gee_sar_engine.py         Sentinel-1 SAR ingestion, Lee filter, dynamic Otsu flood mask
│   ├── dem_processor.py          Copernicus/SRTM DEM elevation + flood depth estimation
│   └── hazard_router.py          OSMnx graph + flood-aware A* routing (boat / pedestrian)
├── mesh_radio/
│   ├── ble_service.py            Async BLE store-and-forward mesh relay (bleak)
│   └── lora_gateway.py           Async serial bridge to ESP32/SX1276 LoRa node
├── server/
│   ├── app.py                    FastAPI: ingestion, active SOS, SAR, routing, voice triage, WS
│   └── schemas.py                Pydantic request/response models
├── dashboard/
│   └── app.py                    Streamlit + Folium NDRF-style command map
├── firmware/
│   └── esp32_lora_node.ino       ESP32 + SX1276 firmware (RadioLib), CRC-matched to Python
├── requirements.txt
└── .env.example
```

## Data flow

```
Voice (offline ASR + triage) ──┐
                                ├──> BinaryDistressProtocol.pack_payload() ──> 27-byte packet
Manual dispatch / sensors ─────┘                                                    │
                                                                                      ▼
                                                              ┌── BLE mesh relay (bleak) ──┐
                                                              │                             │
                                                              └── LoRa serial gateway ───────┼──> FastAPI
                                                                  (ESP32 + SX1276)           │    /ingest_binary
                                                                                              ▼
                                                                                   SQLite (aiosqlite)
                                                                                              │
                                                                                              ▼
                                                                              WebSocket push ──> Streamlit dashboard
```

Independently, the dashboard/backend call `gee_sar_engine.py` for the AOI
around any distress point to get a live Sentinel-1 flood polygon, then
`hazard_router.py` computes a flood-aware A* path from a rescue team's
position to the victim, using that same polygon as the obstacle mask.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Fill in at minimum:
#   GEE_SERVICE_ACCOUNT_EMAIL / GEE_SERVICE_ACCOUNT_KEY_PATH  (or GEE_USE_OAUTH=true)
#   LORA_SERIAL_PORT   (if a physical LoRa gateway is attached)

# Terminal 1 — backend
uvicorn server.app:app --host 0.0.0.0 --port 8000

# Terminal 2 — dashboard
streamlit run dashboard/app.py

# Terminal 3 — LoRa gateway daemon (optional, needs physical hardware)
python -c "
import asyncio
from mesh_radio.lora_gateway import LoRaSerialGateway
import httpx

async def forward(raw_bytes, decoded):
    async with httpx.AsyncClient() as client:
        await client.post(
            'http://localhost:8000/api/v1/sos/ingest_binary',
            content=raw_bytes,
            headers={'Content-Type': 'application/octet-stream', 'X-Received-Via': 'lora'},
        )

asyncio.run(LoRaSerialGateway(on_packet_received=forward).run_forever())
"

# Terminal 4 — BLE mesh daemon (optional)
python -c "
import asyncio
from mesh_radio.ble_service import BLEMeshService
import httpx

async def forward(raw_bytes, decoded):
    async with httpx.AsyncClient() as client:
        await client.post(
            'http://localhost:8000/api/v1/sos/ingest_binary',
            content=raw_bytes,
            headers={'Content-Type': 'application/octet-stream', 'X-Received-Via': 'ble'},
        )

asyncio.run(BLEMeshService(on_packet_received=forward).run_forever())
"
```

Flash `firmware/esp32_lora_node.ino` to an ESP32 + SX1276 board via the
Arduino IDE or `arduino-cli` with the `RadioLib` library installed; set
`LORA_FREQUENCY_MHZ` (868/433) as a build flag for your regulatory region.

## Design notes

- **Nothing is hardcoded per-deployment.** Coordinates, bounding boxes, AOI
  buffers, time windows, speeds, ports, and credentials all come from
  `core/config.py` (env-driven). The only "constants" in the codebase are
  physical/protocol constants (byte offsets, enum codes, CRC polynomial)
  that must match across Python and firmware by definition.
- **CRC-16 is shared** between `packet/binary_protocol.py` (uses `crcmod`'s
  predefined `"crc-16"`, poly 0x8005 reflected) and the firmware's
  `crc16Ansi()` — verified to agree via the packed/unpacked round-trip test.
- **Dynamic Otsu thresholding**: the SAR engine computes the water/no-water
  cutoff from each AOI's own backscatter-difference histogram via
  between-class variance maximization — never a fixed dB constant, so it
  adapts to local terrain, incidence angle, and vegetation.
- **Graceful degradation**: Whisper falls back CUDA→CPU/INT8; the DEM
  processor falls back Copernicus GLO-30→SRTM; the audio recorder falls
  back sounddevice→PyAudio; the LoRa gateway auto-reconnects on USB
  disconnect; BLE advertise failures re-queue the packet for the next cycle.
- **Tenant-free but device-scoped**: distress packets carry a `device_id_hash`
  (not raw PII) so the mesh and server never need to handle personally
  identifying information to route rescue effort.
