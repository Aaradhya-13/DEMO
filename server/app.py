"""
dashboard/app.py
-------------------
Streamlit + Folium NDRF-style tactical command map.

Run with:
    streamlit run dashboard/app.py

Talks to server/app.py over HTTP (DASHBOARD_API_BASE_URL). Renders:
  - Live Sentinel-1 SAR flood-water GeoJSON layer
  - SOS pins color-coded by urgency
  - Turn-by-turn safe route overlay to a selected victim
  - A live audio-recording widget that runs the offline ASR + triage
    pipeline and dispatches a packed 40-byte distress packet
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import folium
import requests
import streamlit as st
from streamlit_folium import st_folium

from core.config import settings

st.set_page_config(page_title="Flood Rescue Command Dashboard", page_icon="🌊", layout="wide")

API_BASE = settings.dashboard_api_base_url

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


def _api_get(path: str, **params) -> dict | None:
    try:
        resp = requests.get(f"{API_BASE}{path}", params=params, timeout=20)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        st.error(f"API request failed: {path} -> {exc}")
        return None


def _api_post_json(path: str, payload: dict) -> dict | None:
    try:
        resp = requests.post(f"{API_BASE}{path}", json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        st.error(f"API request failed: {path} -> {exc}")
        return None


def _api_post_binary(path: str, raw_bytes: bytes, extra_headers: dict | None = None) -> dict | None:
    headers = {"Content-Type": "application/octet-stream"}
    if extra_headers:
        headers.update(extra_headers)
    try:
        resp = requests.post(f"{API_BASE}{path}", data=raw_bytes, headers=headers, timeout=20)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        st.error(f"API request failed: {path} -> {exc}")
        return None


def _api_post_file(path: str, filename: str, file_bytes: bytes, mime: str) -> dict | None:
    try:
        resp = requests.post(
            f"{API_BASE}{path}",
            files={"audio": (filename, file_bytes, mime)},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        st.error(f"API request failed: {path} -> {exc}")
        return None


# --------------------------------------------------------------------------
# Session state
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
# Sidebar: controls
# --------------------------------------------------------------------------

with st.sidebar:
    st.title("🌊 Flood Rescue Command")
    st.caption(f"API: {API_BASE}")

    st.subheader("📡 SAR Flood Mask")
    sar_lat = st.number_input("AOI latitude", format="%.6f", key="sar_lat")
    sar_lon = st.number_input("AOI longitude", format="%.6f", key="sar_lon")
    if st.button("Fetch Sentinel-1 flood mask", use_container_width=True):
        with st.spinner("Running SAR pipeline on Google Earth Engine..."):
            result = _api_get(
                "/api/v1/sar/flood_mask", latitude=sar_lat, longitude=sar_lon
            )
        if result:
            st.session_state.flood_mask_geojson = result["geojson"]
            st.session_state.flood_mask_meta = result
            st.session_state.map_center = (sar_lat, sar_lon)
            st.success(
                f"Flood mask ready — water fraction {result['water_fraction']:.1%}, "
                f"Otsu threshold {result['otsu_threshold_db']:.2f} dB"
            )

    st.divider()
    st.subheader("🎙️ Voice Distress Intake")
    uploaded_audio = st.file_uploader("Upload/record a distress call (WAV)", type=["wav"])
    voice_language = st.text_input(
        "Language code (optional, blank = auto-detect)", value="", key="voice_lang"
    )
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
                "Dispatch lat",
                value=(triage.get("resolved_coordinates") or {}).get("lat", 0.0),
                format="%.6f",
                key="dispatch_lat",
            )
        with col2:
            dispatch_lon = st.number_input(
                "Dispatch lon",
                value=(triage.get("resolved_coordinates") or {}).get("lon", 0.0),
                format="%.6f",
                key="dispatch_lon",
            )
        device_id_hash = st.number_input(
            "Device ID hash", min_value=0, value=1, step=1, key="dispatch_device_hash"
        )
        if st.button("📨 Dispatch distress packet", use_container_width=True):
            from packet.binary_protocol import (
                BinaryDistressProtocol,
                DistressUrgency,
                DistressNeed,
            )

            packet = BinaryDistressProtocol.pack_payload(
                latitude=dispatch_lat,
                longitude=dispatch_lon,
                urgency=DistressUrgency[triage["urgency_level"]],
                victim_count=triage["headcount"],
                need_code=DistressNeed[triage["need_type"]],
                device_id_hash=int(device_id_hash),
                battery_level=15,
                signal_strength=15,
            )
            response = _api_post_binary(
                "/api/v1/sos/ingest_binary", packet, extra_headers={"X-Received-Via": "dashboard"}
            )
            if response:
                st.success(f"Dispatched distress record #{response['id']}")

    st.divider()
    st.subheader("🧭 Route Planning")
    route_mode = st.selectbox("Mode", ["pedestrian", "boat"])
    if st.session_state.selected_victim and st.session_state.flood_mask_geojson:
        origin_lat = st.number_input("Rescue team lat", format="%.6f", key="origin_lat")
        origin_lon = st.number_input("Rescue team lon", format="%.6f", key="origin_lon")
        if st.button("Compute safe route", use_container_width=True):
            victim_lat, victim_lon = st.session_state.selected_victim
            with st.spinner("Computing obstacle-aware A* route..."):
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
                st.success(
                    f"Route ready: {route['distance_meters']:.0f} m, "
                    f"~{route['estimated_duration_minutes']:.1f} min"
                )
    else:
        st.caption("Select a victim pin on the map and fetch a flood mask first.")

    st.divider()
    auto_refresh = st.checkbox(
        "Auto-refresh SOS feed", value=True,
        help=f"Refreshes every {settings.dashboard_refresh_interval_seconds}s",
    )


# --------------------------------------------------------------------------
# Main map
# --------------------------------------------------------------------------

st.header("🗺️ Live Tactical Map")

active = _api_get("/api/v1/sos/active_distress") or {"count": 0, "records": []}
records = active["records"]

if st.session_state.map_center:
    center = st.session_state.map_center
elif records:
    center = (records[0]["latitude"], records[0]["longitude"])
else:
    center = (20.5937, 78.9629)  # dynamic fallback: geographic centroid input, not a "demo" location claim

fmap = folium.Map(location=center, zoom_start=settings.dashboard_default_map_zoom, tiles="OpenStreetMap")

if st.session_state.flood_mask_geojson:
    folium.GeoJson(
        st.session_state.flood_mask_geojson,
        name="SAR Flood Mask",
        style_function=lambda _: {"fillColor": "#1f77b4", "color": "#1f77b4", "fillOpacity": 0.4},
    ).add_to(fmap)

for record in records:
    urgency = record["urgency"]
    need = record["need_code"]
    color = _NEED_COLOR_OVERRIDE.get(need, _URGENCY_COLORS.get(urgency, "gray"))
    popup_html = (
        f"<b>ID:</b> {record['id']}<br>"
        f"<b>Urgency:</b> {urgency}<br>"
        f"<b>Need:</b> {need}<br>"
        f"<b>Victims:</b> {record['victim_count']}<br>"
        f"<b>Battery:</b> {record['battery_level_0_15']}/15<br>"
        f"<b>Received via:</b> {record['received_via']}<br>"
        f"<b>Received at:</b> {record['received_at']}"
    )
    marker = folium.Marker(
        location=(record["latitude"], record["longitude"]),
        popup=folium.Popup(popup_html, max_width=300),
        icon=folium.Icon(color=color, icon="exclamation-triangle", prefix="fa"),
    )
    marker.add_to(fmap)

if st.session_state.active_route and st.session_state.active_route.get("coordinates"):
    folium.PolyLine(
        st.session_state.active_route["coordinates"],
        color="lime",
        weight=5,
        opacity=0.9,
        tooltip=(
            f"{st.session_state.active_route['mode'].title()} route — "
            f"{st.session_state.active_route['distance_meters']:.0f} m"
        ),
    ).add_to(fmap)

folium.LayerControl().add_to(fmap)

map_state = st_folium(fmap, width=None, height=600, returned_objects=["last_object_clicked"])

if map_state and map_state.get("last_object_clicked"):
    clicked = map_state["last_object_clicked"]
    st.session_state.selected_victim = (clicked["lat"], clicked["lng"])
    st.info(f"Selected point for routing: {clicked['lat']:.5f}, {clicked['lng']:.5f}")


# --------------------------------------------------------------------------
# Active distress table
# --------------------------------------------------------------------------

st.header("📋 Active SOS Queue")
if records:
    st.dataframe(
        [
            {
                "ID": r["id"],
                "Urgency": r["urgency"],
                "Need": r["need_code"],
                "Victims": r["victim_count"],
                "Lat": r["latitude"],
                "Lon": r["longitude"],
                "Battery": r["battery_level_0_15"],
                "Via": r["received_via"],
                "Received": r["received_at"],
            }
            for r in records
        ],
        use_container_width=True,
    )
    resolve_id = st.number_input("Resolve distress by ID", min_value=0, step=1, value=0)
    if st.button("Mark resolved") and resolve_id > 0:
        result = requests.post(f"{API_BASE}/api/v1/sos/{int(resolve_id)}/resolve", timeout=10)
        if result.ok:
            st.success(f"Resolved #{resolve_id}")
            st.rerun()
else:
    st.caption("No active distress signals.")

if auto_refresh:
    time.sleep(settings.dashboard_refresh_interval_seconds)
    st.rerun()
