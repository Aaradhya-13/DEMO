import os
import time
import struct
import requests
import folium
import streamlit as st
from streamlit_folium import st_folium

st.set_page_config(page_title="Flood Rescue Command Dashboard", page_icon="🌊", layout="wide")

# Robust URL cleaning: strips hidden single/double quotes, newlines, and trailing slashes
raw_api = os.getenv("DASHBOARD_API_BASE_URL", "https://demo-w8i9.onrender.com")
API_BASE = raw_api.strip().strip("'\"").rstrip("/")

REFRESH_INTERVAL = 10
DEFAULT_ZOOM = 12

_URGENCY_COLORS = {
    "CRITICAL": "red",
    "HIGH": "orange",
    "MEDIUM": "beige",
    "LOW": "lightgray",
}
_NEED_COLOR_OVERRIDE = {
    "MEDICAL": "red",
    "FOOD_WATER": "yellow",
}


def _api_get(path, **params):
    try:
        url = f"{API_BASE}{path}"
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        st.error(f"API request failed: {path} -> {exc}")
        return None


def _api_post_json(path, payload):
    try:
        url = f"{API_BASE}{path}"
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        st.error(f"API request failed: {path} -> {exc}")
        return None


def _api_post_binary(path, raw_bytes, extra_headers=None):
    headers = {"Content-Type": "application/octet-stream"}
    if extra_headers:
        headers.update(extra_headers)
    try:
        url = f"{API_BASE}{path}"
        resp = requests.post(url, data=raw_bytes, headers=headers, timeout=20)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        st.error(f"API request failed: {path} -> {exc}")
        return None


def _api_post_file(path, filename, file_bytes, mime):
    try:
        url = f"{API_BASE}{path}"
        resp = requests.post(
            url,
            files={"audio": (filename, file_bytes, mime)},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        st.error(f"API request failed: {path} -> {exc}")
        return None


# --------------------------------------------------------------------------
# Session State Initialization
# --------------------------------------------------------------------------

if "flood_mask_geojson" not in st.session_state:
    st.session_state.flood_mask_geojson = None
if "flood_mask_meta" not in st.session_state:
    st.session_state.flood_mask_meta = None
if "active_route" not in st.session_state:
    st.session_state.active_route = None
if "selected_victim" not in st.session_state:
    st.session_state.selected_victim = None
if "map_center" not in st.session_state:
    st.session_state.map_center = None


# --------------------------------------------------------------------------
# Sidebar: Controls
# --------------------------------------------------------------------------

with st.sidebar:
    st.title("🌊 Flood Rescue Command")
    st.caption(f"Backend API: `{API_BASE}`")

    st.subheader("📡 SAR Flood Mask")
    sar_lat = st.number_input("AOI Latitude", format="%.6f", key="sar_lat", value=26.8452)
    sar_lon = st.number_input("AOI Longitude", format="%.6f", key="sar_lon", value=94.2148)

    if st.button("Fetch Sentinel-1 Flood Mask", use_container_width=True):
        with st.spinner("Running SAR pipeline on Google Earth Engine..."):
            result = _api_get(
                "/api/v1/sar/flood_mask", latitude=sar_lat, longitude=sar_lon
            )
        if result and "geojson" in result:
            st.session_state.flood_mask_geojson = result["geojson"]
            st.session_state.flood_mask_meta = result
            st.session_state.map_center = (sar_lat, sar_lon)
            water_frac = result.get("water_fraction", 0.0)
            otsu_db = result.get("otsu_threshold_db", 0.0)
            st.success(f"Flood mask ready — water fraction: {water_frac:.1%}, Otsu threshold: {otsu_db:.2f} dB")

    st.divider()
    st.subheader("🎙️ Voice Distress Intake")
    uploaded_audio = st.file_uploader("Upload / Record Distress Call (WAV)", type=["wav"])
    voice_language = st.text_input("Language code (optional, blank = auto-detect)", value="", key="voice_lang")

    if uploaded_audio is not None and st.button("Run ASR + Triage", use_container_width=True):
        with st.spinner("Transcribing and extracting triage data..."):
            triage = _api_post_file(
                "/api/v1/voice/triage", uploaded_audio.name, uploaded_audio.getvalue(), "audio/wav"
            )
        if triage:
            st.json(triage)
            st.session_state["last_triage"] = triage

    if st.session_state.get("last_triage"):
        triage = st.session_state["last_triage"]
        st.caption("Dispatch this triage as a distress packet:")
        col1, col2 = st.columns(2)
        with col1:
            dispatch_lat = st.number_input(
                "Dispatch Lat",
                value=float((triage.get("resolved_coordinates") or {}).get("lat", 26.8452)),
                format="%.6f",
                key="dispatch_lat",
            )
        with col2:
            dispatch_lon = st.number_input(
                "Dispatch Lon",
                value=float((triage.get("resolved_coordinates") or {}).get("lon", 94.2148)),
                format="%.6f",
                key="dispatch_lon",
            )
        device_id_hash = st.number_input(
            "Device ID hash", min_value=0, value=1, step=1, key="dispatch_device_hash"
        )
        if st.button("📨 Dispatch Distress Packet", use_container_width=True):
            urgency_val = 3 if triage.get("urgency_level") == "CRITICAL" else 2
            need_val = 0 if triage.get("need_type") == "BOAT_EVACUATION" else 1
            count_val = int(triage.get("headcount", 1))
            
            packet = struct.pack(
                ">BffBBBIQBBH",
                0xAA,
                dispatch_lat,
                dispatch_lon,
                urgency_val,
                count_val,
                need_val,
                int(time.time()),
                int(device_id_hash),
                (15 << 4) | 15,
                0,
                0x1234
            )

            response = _api_post_binary(
                "/api/v1/sos/ingest_binary", packet, extra_headers={"X-Received-Via": "dashboard"}
            )
            if response:
                st.success(f"Dispatched distress record #{response.get('id', 'OK')}")

    st.divider()
    st.subheader("🧭 Route Planning")
    route_mode = st.selectbox("Mode", ["boat", "pedestrian"])
    if st.session_state.selected_victim and st.session_state.flood_mask_geojson:
        origin_lat = st.number_input("Rescue team lat", format="%.6f", key="origin_lat", value=26.8400)
        origin_lon = st.number_input("Rescue team lon", format="%.6f", key="origin_lon", value=94.2100)
        if st.button("Compute Safe Route", use_container_width=True):
            victim_lat, victim_lon = st.session_state.selected_victim
            with st.spinner("Computing obstacle-aware path..."):
                route = _api_post_json(
                    "/api/v1/route",
                    {
                        "origin_lat": origin_lat,
                        "origin_lon": origin_lon,
                        "destination_lat": victim_lat,
                        "destination_lon": victim_lon,
                        "mode": route_mode,
                        "flood_geojson": st.session_state.flood_mask_geojson,
                    },
                )
            if route:
                st.session_state.active_route = route
                dist = route.get("distance_meters", 0.0)
                dur = route.get("estimated_duration_minutes", 0.0)
                st.success(f"Route ready: {dist:.0f} m, ~{dur:.1f} min")
    else:
        st.caption("Select a victim pin on the map and fetch a flood mask first.")

    st.divider()
    auto_refresh = st.checkbox(
        "Auto-refresh SOS feed", value=False,
        help=f"Refreshes feed every {REFRESH_INTERVAL}s",
    )


# --------------------------------------------------------------------------
# Main Map View
# --------------------------------------------------------------------------

st.header("🗺️ Live Tactical Command Map")

active = _api_get("/api/v1/sos/active_distress") or {"count": 0, "records": []}
records = active.get("records", [])

if st.session_state.map_center:
    center = st.session_state.map_center
elif records:
    center = (records[0]["latitude"], records[0]["longitude"])
else:
    center = (26.8452, 94.2148)

fmap = folium.Map(location=center, zoom_start=DEFAULT_ZOOM, tiles="OpenStreetMap")

# Render SAR Flood Mask Layer
if st.session_state.flood_mask_geojson:
    folium.GeoJson(
        st.session_state.flood_mask_geojson,
        name="SAR Flood Mask",
        style_function=lambda _: {"fillColor": "#1f77b4", "color": "#1f77b4", "fillOpacity": 0.4},
    ).add_to(fmap)

# Render Distress SOS Pins
for record in records:
    urgency = str(record.get("urgency", "CRITICAL"))
    need = str(record.get("need_code", "BOAT_EVACUATION"))
    color = _NEED_COLOR_OVERRIDE.get(need, _URGENCY_COLORS.get(urgency, "red"))
    popup_html = (
        f"<b>ID:</b> {record.get('id')}<br>"
        f"<b>Urgency:</b> {urgency}<br>"
        f"<b>Need:</b> {need}<br>"
        f"<b>Victims:</b> {record.get('victim_count')}<br>"
        f"<b>Battery:</b> {record.get('battery_level_0_15', 15)}/15<br>"
        f"<b>Received via:</b> {record.get('received_via', 'mesh')}<br>"
        f"<b>Time:</b> {record.get('received_at', '')}"
    )
    marker = folium.Marker(
        location=(record["latitude"], record["longitude"]),
        popup=folium.Popup(popup_html, max_width=300),
        icon=folium.Icon(color=color, icon="exclamation-triangle", prefix="fa"),
    )
    marker.add_to(fmap)

# Render Navigation Route
if st.session_state.active_route and st.session_state.active_route.get("coordinates"):
    folium.PolyLine(
        st.session_state.active_route["coordinates"],
        color="lime",
        weight=5,
        opacity=0.9,
        tooltip=(
            f"{st.session_state.active_route.get('mode', 'boat').title()} route — "
            f"{st.session_state.active_route.get('distance_meters', 0):.0f} m"
        ),
    ).add_to(fmap)

folium.LayerControl().add_to(fmap)

map_state = st_folium(fmap, width=None, height=600, returned_objects=["last_object_clicked"])

if map_state and map_state.get("last_object_clicked"):
    clicked = map_state["last_object_clicked"]
    st.session_state.selected_victim = (clicked["lat"], clicked["lng"])
    st.info(f"Selected point for routing: {clicked['lat']:.5f}, {clicked['lng']:.5f}")


# --------------------------------------------------------------------------
# Active Distress Queue Table
# --------------------------------------------------------------------------

st.header("📋 Active SOS Queue")
if records:
    st.dataframe(
        [
            {
                "ID": r.get("id"),
                "Urgency": r.get("urgency"),
                "Need": r.get("need_code"),
                "Victims": r.get("victim_count"),
                "Lat": r.get("latitude"),
                "Lon": r.get("longitude"),
                "Battery": r.get("battery_level_0_15"),
                "Via": r.get("received_via"),
                "Received": r.get("received_at"),
            }
            for r in records
        ],
        use_container_width=True,
    )
    resolve_id = st.number_input("Resolve distress by ID", min_value=0, step=1, value=0)
    if st.button("Mark Resolved") and resolve_id > 0:
        result = requests.post(f"{API_BASE}/api/v1/sos/{int(resolve_id)}/resolve", timeout=10)
        if result.ok:
            st.success(f"Resolved #{resolve_id}")
            st.rerun()
else:
    st.caption("No active distress signals in queue.")

if auto_refresh:
    time.sleep(REFRESH_INTERVAL)
    st.rerun()
