"""
mesh_radio/lora_gateway.py
-----------------------------
Async serial interface to an ESP32 + SX1276 LoRa transceiver
(see firmware/esp32_lora_node.ino for the paired firmware).

The firmware echoes every received-over-the-air packet to Serial USB as a
hex-encoded line (`RX:<hex>\n`). This module listens on the configured
serial port, decodes those lines back into raw bytes, validates the
binary distress protocol CRC, and forwards well-formed packets to a
central async queue (e.g. consumed by server/app.py's ingestion endpoint).

Reconnects automatically if the USB serial device is unplugged/replugged,
using a configurable backoff — never crashes the whole gateway process.
"""

from __future__ import annotations

import asyncio
import binascii
from typing import Awaitable, Callable, Optional

from core.config import settings
from core.logger import get_logger
from packet.binary_protocol import BinaryDistressProtocol, PacketValidationError

logger = get_logger(__name__)

PacketHandler = Callable[[bytes, dict], Awaitable[None]]

_LINE_PREFIX = "RX:"


class LoRaGatewayError(RuntimeError):
    """Raised for unrecoverable configuration errors (e.g. no port configured)."""


class LoRaSerialGateway:
    """
    Async serial listener for an ESP32/SX1276 LoRa node bridged over USB.

    Usage:
        gateway = LoRaSerialGateway(on_packet_received=my_async_handler)
        await gateway.run_forever()
    """

    def __init__(
        self,
        serial_port: Optional[str] = None,
        baud_rate: Optional[int] = None,
        on_packet_received: Optional[PacketHandler] = None,
    ):
        self.serial_port = serial_port or settings.lora_serial_port
        self.baud_rate = baud_rate or settings.lora_baud_rate
        self.on_packet_received = on_packet_received
        self._stop_event = asyncio.Event()

        if not self.serial_port:
            raise LoRaGatewayError(
                "No LoRa serial port configured. Set LORA_SERIAL_PORT in the environment "
                "(e.g. /dev/ttyUSB0 or COM5), or pass serial_port explicitly."
            )

    async def run_forever(self) -> None:
        """Maintain a persistent connection to the LoRa gateway, reconnecting on failure."""
        import serial_asyncio

        logger.info(
            "Starting LoRa serial gateway",
            extra={"context": {"port": self.serial_port, "baud": self.baud_rate}},
        )

        while not self._stop_event.is_set():
            try:
                reader, writer = await serial_asyncio.open_serial_connection(
                    url=self.serial_port, baudrate=self.baud_rate
                )
                logger.info("LoRa serial connection established", extra={"context": {"port": self.serial_port}})
                await self._read_loop(reader)
            except (FileNotFoundError, OSError) as exc:
                logger.warning(
                    "LoRa serial port unavailable, retrying",
                    extra={"context": {"port": self.serial_port, "error": str(exc)}},
                )
            except Exception as exc:
                logger.error("Unexpected LoRa gateway error", extra={"context": {"error": str(exc)}})
            finally:
                await asyncio.sleep(settings.lora_reconnect_delay_seconds)

    async def _read_loop(self, reader: "asyncio.StreamReader") -> None:
        while not self._stop_event.is_set():
            line_bytes = await reader.readline()
            if not line_bytes:
                logger.warning("LoRa serial stream closed by device")
                return

            try:
                line = line_bytes.decode("ascii", errors="ignore").strip()
            except Exception:
                continue

            if not line.startswith(_LINE_PREFIX):
                logger.debug("Ignoring non-packet serial line", extra={"context": {"line": line}})
                continue

            hex_payload = line[len(_LINE_PREFIX):].strip()
            await self._handle_hex_payload(hex_payload)

    async def _handle_hex_payload(self, hex_payload: str) -> None:
        try:
            raw_bytes = binascii.unhexlify(hex_payload)
        except (binascii.Error, ValueError) as exc:
            logger.warning(
                "Malformed hex payload from LoRa node",
                extra={"context": {"payload": hex_payload, "error": str(exc)}},
            )
            return

        try:
            decoded = BinaryDistressProtocol.unpack_payload(raw_bytes)
        except PacketValidationError as exc:
            logger.warning(
                "Dropped LoRa packet failing CRC/validation",
                extra={"context": {"error": str(exc)}},
            )
            return

        logger.info("Received valid distress packet over LoRa", extra={"context": decoded})

        if self.on_packet_received is not None:
            await self.on_packet_received(raw_bytes, decoded)

    async def send_packet(self, packet: bytes) -> None:
        """Transmit a packet outbound to the LoRa node for over-the-air broadcast."""
        import serial_asyncio

        reader, writer = await serial_asyncio.open_serial_connection(
            url=self.serial_port, baudrate=self.baud_rate
        )
        try:
            hex_line = f"TX:{packet.hex()}\n".encode("ascii")
            writer.write(hex_line)
            await writer.drain()
            logger.info("Sent packet to LoRa node for TX", extra={"context": {"size_bytes": len(packet)}})
        finally:
            writer.close()

    def stop(self) -> None:
        self._stop_event.set()
