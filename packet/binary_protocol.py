"""
packet/binary_protocol.py
--------------------------
Compact binary distress-signal protocol designed for LoRa / BLE mesh
transport where every byte is precious.

Wire layout (big-endian, network byte order):

    Offset  Size  Field                Type
    ------  ----  -------------------  ------------------------------
    0       1     header_version       uint8  (high nibble = magic header,
                                                low nibble = protocol version)
    1       4     latitude             float32
    5       4     longitude            float32
    9       1     urgency              uint8  (DistressUrgency)
    10      1     victim_count         uint8
    11      1     need_code            uint8  (DistressNeed)
    12      4     timestamp            uint32 (unix epoch seconds)
    16      8     device_id_hash       uint64
    24      1     battery_signal       uint8  (packed: high nibble battery
                                                0-15 scale, low nibble signal
                                                RSSI bucket 0-15 scale)
    25      2     crc16                uint16 (CRC-16/ANSI over bytes 0..24)
    ------------------------------------------------------------------
    Total: 27 bytes  (<= PROTOCOL_MAX_PACKET_BYTES, dynamically enforced)
"""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass
from enum import IntEnum

import crcmod

from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)

# '>' = big-endian / network order, no padding
_STRUCT_FORMAT = ">BffBBBIQB"
_PAYLOAD_SIZE = struct.calcsize(_STRUCT_FORMAT)  # bytes before CRC
_CRC_SIZE = 2
_TOTAL_SIZE = _PAYLOAD_SIZE + _CRC_SIZE

_crc16_func = crcmod.predefined.mkCrcFun("crc-16")

_HEADER_MAGIC_NIBBLE = (settings.protocol_header_byte >> 4) & 0x0F


class DistressUrgency(IntEnum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3


class DistressNeed(IntEnum):
    RESCUE = 0
    BOAT_EVACUATION = 1
    MEDICAL = 2
    FOOD_WATER = 3


class PacketValidationError(ValueError):
    """Raised when a packet fails length, header, or CRC validation."""


@dataclass(frozen=True)
class DistressPacket:
    protocol_version: int
    latitude: float
    longitude: float
    urgency: DistressUrgency
    victim_count: int
    need_code: DistressNeed
    timestamp: int
    device_id_hash: int
    battery_level: int  # 0-15 scale
    signal_strength: int  # 0-15 scale


class BinaryDistressProtocol:
    """Stateless packer/unpacker for the fixed 40-byte-max distress packet."""

    @staticmethod
    def _encode_header_version(version: int) -> int:
        if not (0 <= version <= 0x0F):
            raise ValueError(f"protocol version must fit in 4 bits (0-15), got {version}")
        return ((_HEADER_MAGIC_NIBBLE & 0x0F) << 4) | (version & 0x0F)

    @staticmethod
    def _decode_header_version(byte_val: int) -> tuple[int, int]:
        magic = (byte_val >> 4) & 0x0F
        version = byte_val & 0x0F
        return magic, version

    @staticmethod
    def _pack_battery_signal(battery_level: int, signal_strength: int) -> int:
        if not (0 <= battery_level <= 15):
            raise ValueError(f"battery_level must be 0-15, got {battery_level}")
        if not (0 <= signal_strength <= 15):
            raise ValueError(f"signal_strength must be 0-15, got {signal_strength}")
        return ((battery_level & 0x0F) << 4) | (signal_strength & 0x0F)

    @staticmethod
    def _unpack_battery_signal(byte_val: int) -> tuple[int, int]:
        battery_level = (byte_val >> 4) & 0x0F
        signal_strength = byte_val & 0x0F
        return battery_level, signal_strength

    @classmethod
    def pack_payload(
        cls,
        latitude: float,
        longitude: float,
        urgency: DistressUrgency,
        victim_count: int,
        need_code: DistressNeed,
        device_id_hash: int,
        battery_level: int,
        signal_strength: int,
        timestamp: int | None = None,
        protocol_version: int | None = None,
    ) -> bytes:
        """Serialize a distress event into a CRC-protected binary payload."""
        if not (-90.0 <= latitude <= 90.0):
            raise ValueError(f"latitude out of range: {latitude}")
        if not (-180.0 <= longitude <= 180.0):
            raise ValueError(f"longitude out of range: {longitude}")
        if not (0 <= victim_count <= 255):
            raise ValueError(f"victim_count out of range: {victim_count}")
        if device_id_hash < 0 or device_id_hash > 0xFFFFFFFFFFFFFFFF:
            raise ValueError("device_id_hash must fit in an unsigned 64-bit integer")

        version = protocol_version if protocol_version is not None else settings.protocol_version
        ts = timestamp if timestamp is not None else int(time.time())
        if ts < 0 or ts > 0xFFFFFFFF:
            raise ValueError("timestamp must fit in an unsigned 32-bit integer")

        header_version = cls._encode_header_version(version)
        battery_signal = cls._pack_battery_signal(battery_level, signal_strength)

        body = struct.pack(
            _STRUCT_FORMAT,
            header_version,
            float(latitude),
            float(longitude),
            int(urgency),
            int(victim_count),
            int(need_code),
            int(ts) & 0xFFFFFFFF,
            int(device_id_hash) & 0xFFFFFFFFFFFFFFFF,
            battery_signal,
        )

        crc = _crc16_func(body)
        packet = body + struct.pack(">H", crc)

        if len(packet) > settings.protocol_max_packet_bytes:
            raise PacketValidationError(
                f"Packed payload of {len(packet)} bytes exceeds "
                f"PROTOCOL_MAX_PACKET_BYTES={settings.protocol_max_packet_bytes}"
            )

        logger.debug(
            "Packed distress payload",
            extra={"context": {"size_bytes": len(packet), "urgency": urgency.name}},
        )
        return packet

    @classmethod
    def unpack_payload(cls, raw_bytes: bytes) -> dict:
        """
        Validate and decode a raw binary payload into a plain dict.
        Raises PacketValidationError on length, header-magic, or CRC failure.
        """
        if len(raw_bytes) != _TOTAL_SIZE:
            raise PacketValidationError(
                f"Expected exactly {_TOTAL_SIZE} bytes, got {len(raw_bytes)}"
            )

        body, crc_bytes = raw_bytes[:_PAYLOAD_SIZE], raw_bytes[_PAYLOAD_SIZE:]
        (received_crc,) = struct.unpack(">H", crc_bytes)
        computed_crc = _crc16_func(body)
        if received_crc != computed_crc:
            raise PacketValidationError(
                f"CRC mismatch: received=0x{received_crc:04X} computed=0x{computed_crc:04X}"
            )

        (
            header_version,
            latitude,
            longitude,
            urgency_raw,
            victim_count,
            need_code_raw,
            ts,
            device_id_hash,
            battery_signal,
        ) = struct.unpack(_STRUCT_FORMAT, body)

        magic, version = cls._decode_header_version(header_version)
        if magic != _HEADER_MAGIC_NIBBLE:
            raise PacketValidationError(
                f"Header magic mismatch: expected 0x{_HEADER_MAGIC_NIBBLE:X}, got 0x{magic:X}"
            )

        try:
            urgency = DistressUrgency(urgency_raw)
        except ValueError as exc:
            raise PacketValidationError(f"Unknown urgency code: {urgency_raw}") from exc

        try:
            need_code = DistressNeed(need_code_raw)
        except ValueError as exc:
            raise PacketValidationError(f"Unknown need code: {need_code_raw}") from exc

        battery_level, signal_strength = cls._unpack_battery_signal(battery_signal)

        packet = DistressPacket(
            protocol_version=version,
            latitude=latitude,
            longitude=longitude,
            urgency=urgency,
            victim_count=victim_count,
            need_code=need_code,
            timestamp=ts,
            device_id_hash=device_id_hash,
            battery_level=battery_level,
            signal_strength=signal_strength,
        )

        return {
            "protocol_version": packet.protocol_version,
            "latitude": packet.latitude,
            "longitude": packet.longitude,
            "urgency": packet.urgency.name,
            "victim_count": packet.victim_count,
            "need_code": packet.need_code.name,
            "timestamp": packet.timestamp,
            "device_id_hash": packet.device_id_hash,
            "battery_level_0_15": packet.battery_level,
            "signal_strength_0_15": packet.signal_strength,
        }

    @staticmethod
    def packet_size_bytes() -> int:
        return _TOTAL_SIZE
