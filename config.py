"""
core/config.py
---------------
Centralized, environment-driven configuration for the flood rescue system.
No coordinates, bounding boxes, ports, or credentials are hardcoded anywhere
else in the codebase — every module imports `settings` from here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="",
        extra="ignore",
    )

    # ---------------------------------------------------------------
    # General / deployment
    # ---------------------------------------------------------------
    environment: str = Field(default="development", validation_alias="ENVIRONMENT")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    log_json: bool = Field(default=True, validation_alias="LOG_JSON")
    service_name: str = Field(default="flood-rescue-system", validation_alias="SERVICE_NAME")
    timezone: str = Field(default="UTC", validation_alias="TZ")

    # ---------------------------------------------------------------
    # Google Earth Engine
    # ---------------------------------------------------------------
    gee_service_account_email: Optional[str] = Field(
        default=None, validation_alias="GEE_SERVICE_ACCOUNT_EMAIL"
    )
    gee_service_account_key_path: Optional[Path] = Field(
        default=None, validation_alias="GEE_SERVICE_ACCOUNT_KEY_PATH"
    )
    gee_project_id: Optional[str] = Field(default=None, validation_alias="GEE_PROJECT_ID")
    gee_use_oauth: bool = Field(default=False, validation_alias="GEE_USE_OAUTH")

    # SAR analysis parameters (all dynamic — no magic numbers baked into code)
    sar_aoi_buffer_meters: float = Field(default=5000.0, validation_alias="SAR_AOI_BUFFER_METERS")
    sar_baseline_window_days: int = Field(default=30, validation_alias="SAR_BASELINE_WINDOW_DAYS")
    sar_postflood_window_days: int = Field(default=3, validation_alias="SAR_POSTFLOOD_WINDOW_DAYS")
    sar_speckle_filter_kernel_size: int = Field(
        default=7, validation_alias="SAR_SPECKLE_FILTER_KERNEL_SIZE"
    )
    sar_polarizations: str = Field(default="VV,VH", validation_alias="SAR_POLARIZATIONS")

    @field_validator("sar_speckle_filter_kernel_size")
    @classmethod
    def _kernel_must_be_odd(cls, v: int) -> int:
        if v % 2 == 0:
            raise ValueError("SAR_SPECKLE_FILTER_KERNEL_SIZE must be odd (Lee filter window).")
        return v

    # ---------------------------------------------------------------
    # DEM / elevation
    # ---------------------------------------------------------------
    dem_dataset_id: str = Field(
        default="COPERNICUS/DEM/GLO30", validation_alias="DEM_DATASET_ID"
    )
    dem_fallback_dataset_id: str = Field(
        default="USGS/SRTMGL1_003", validation_alias="DEM_FALLBACK_DATASET_ID"
    )

    # ---------------------------------------------------------------
    # Hazard routing / OSMnx
    # ---------------------------------------------------------------
    routing_network_type: str = Field(default="drive", validation_alias="ROUTING_NETWORK_TYPE")
    routing_graph_buffer_meters: float = Field(
        default=3000.0, validation_alias="ROUTING_GRAPH_BUFFER_METERS"
    )
    routing_boat_speed_kmh: float = Field(default=15.0, validation_alias="ROUTING_BOAT_SPEED_KMH")
    routing_pedestrian_speed_kmh: float = Field(
        default=4.5, validation_alias="ROUTING_PEDESTRIAN_SPEED_KMH"
    )

    # ---------------------------------------------------------------
    # Edge NLP (ASR + entity extraction)
    # ---------------------------------------------------------------
    whisper_model_size: str = Field(default="small", validation_alias="WHISPER_MODEL_SIZE")
    whisper_device: str = Field(default="auto", validation_alias="WHISPER_DEVICE")  # auto|cuda|cpu
    whisper_compute_type: str = Field(default="int8", validation_alias="WHISPER_COMPUTE_TYPE")
    whisper_default_language: Optional[str] = Field(
        default=None, validation_alias="WHISPER_DEFAULT_LANGUAGE"
    )
    audio_sample_rate_hz: int = Field(default=16000, validation_alias="AUDIO_SAMPLE_RATE_HZ")
    audio_channels: int = Field(default=1, validation_alias="AUDIO_CHANNELS")
    audio_max_record_seconds: int = Field(default=30, validation_alias="AUDIO_MAX_RECORD_SECONDS")

    ner_model_name: str = Field(
        default="facebook/bart-large-mnli", validation_alias="NER_MODEL_NAME"
    )
    nominatim_user_agent: str = Field(
        default="flood-rescue-system", validation_alias="NOMINATIM_USER_AGENT"
    )
    nominatim_base_url: str = Field(
        default="https://nominatim.openstreetmap.org",
        validation_alias="NOMINATIM_BASE_URL",
    )

    # ---------------------------------------------------------------
    # Binary distress protocol
    # ---------------------------------------------------------------
    protocol_max_packet_bytes: int = Field(default=40, validation_alias="PROTOCOL_MAX_PACKET_BYTES")
    protocol_header_byte: int = Field(default=0xA5, validation_alias="PROTOCOL_HEADER_BYTE")
    protocol_version: int = Field(default=1, validation_alias="PROTOCOL_VERSION")

    # ---------------------------------------------------------------
    # LoRa serial gateway
    # ---------------------------------------------------------------
    lora_serial_port: Optional[str] = Field(default=None, validation_alias="LORA_SERIAL_PORT")
    lora_baud_rate: int = Field(default=115200, validation_alias="LORA_BAUD_RATE")
    lora_reconnect_delay_seconds: float = Field(
        default=5.0, validation_alias="LORA_RECONNECT_DELAY_SECONDS"
    )

    # ---------------------------------------------------------------
    # BLE mesh
    # ---------------------------------------------------------------
    ble_service_uuid: str = Field(
        default="6e400001-b5a3-f393-e0a9-e50e24dcca9e", validation_alias="BLE_SERVICE_UUID"
    )
    ble_characteristic_uuid: str = Field(
        default="6e400002-b5a3-f393-e0a9-e50e24dcca9e",
        validation_alias="BLE_CHARACTERISTIC_UUID",
    )
    ble_scan_interval_seconds: float = Field(
        default=10.0, validation_alias="BLE_SCAN_INTERVAL_SECONDS"
    )
    ble_relay_queue_max_size: int = Field(default=500, validation_alias="BLE_RELAY_QUEUE_MAX_SIZE")

    # ---------------------------------------------------------------
    # FastAPI server
    # ---------------------------------------------------------------
    server_host: str = Field(default="0.0.0.0", validation_alias="SERVER_HOST")
    server_port: int = Field(default=8000, validation_alias="SERVER_PORT")
    database_url: str = Field(
        default="sqlite+aiosqlite:///./flood_rescue.db", validation_alias="DATABASE_URL"
    )
    cors_allow_origins: str = Field(default="*", validation_alias="CORS_ALLOW_ORIGINS")

    # ---------------------------------------------------------------
    # Dashboard
    # ---------------------------------------------------------------
    dashboard_api_base_url: str = Field(
        default="http://localhost:8000", validation_alias="DASHBOARD_API_BASE_URL"
    )
    dashboard_default_map_zoom: int = Field(
        default=11, validation_alias="DASHBOARD_DEFAULT_MAP_ZOOM"
    )
    dashboard_refresh_interval_seconds: int = Field(
        default=15, validation_alias="DASHBOARD_REFRESH_INTERVAL_SECONDS"
    )

    @property
    def sar_polarization_list(self) -> list[str]:
        return [p.strip().upper() for p in self.sar_polarizations.split(",") if p.strip()]

    @property
    def cors_origin_list(self) -> list[str]:
        if self.cors_allow_origins.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.cors_allow_origins.split(",") if o.strip()]


settings = Settings()
