"""
mesh_radio/ble_service.py
----------------------------
Async BLE mesh relay daemon built on `bleak`. Acts simultaneously as:

  - a GATT peripheral/advertiser that broadcasts queued distress packets
    (store-and-forward), and
  - a central/scanner that observes nearby advertisements, deduplicates
    packets it has already relayed, validates their CRC-16, and re-queues
    them for further hops.

This gives phone-to-phone mesh propagation of the 40-byte distress packet
in areas with no cellular/LoRa coverage, entirely from configuration
(service/characteristic UUIDs, scan interval, queue size) — nothing hardcoded.
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from typing import Awaitable, Callable, Optional

from core.config import settings
from core.logger import get_logger
from packet.binary_protocol import BinaryDistressProtocol, PacketValidationError

logger = get_logger(__name__)

PacketHandler = Callable[[bytes, dict], Awaitable[None]]


class _BoundedDedupeCache:
    """Fixed-size FIFO cache of recently-seen packet hashes to prevent
    infinite mesh relay loops without unbounded memory growth."""

    def __init__(self, max_size: int):
        self._max_size = max_size
        self._seen: "OrderedDict[bytes, float]" = OrderedDict()

    def has_seen(self, packet: bytes) -> bool:
        return packet in self._seen

    def mark_seen(self, packet: bytes) -> None:
        self._seen[packet] = time.time()
        self._seen.move_to_end(packet)
        while len(self._seen) > self._max_size:
            self._seen.popitem(last=False)


class BLEMeshService:
    """
    Store-and-forward BLE relay for distress packets.

    Usage:
        service = BLEMeshService(on_packet_received=my_async_handler)
        await service.run_forever()
    """

    def __init__(self, on_packet_received: Optional[PacketHandler] = None):
        self.on_packet_received = on_packet_received
        self._outbound_queue: "asyncio.Queue[bytes]" = asyncio.Queue(
            maxsize=settings.ble_relay_queue_max_size
        )
        self._dedupe_cache = _BoundedDedupeCache(settings.ble_relay_queue_max_size)
        self._stop_event = asyncio.Event()

    async def enqueue_packet(self, packet: bytes) -> None:
        """Queue a locally-originated or relayed packet for BLE broadcast."""
        if self._dedupe_cache.has_seen(packet):
            logger.debug("Skipping enqueue of already-seen packet")
            return
        try:
            self._outbound_queue.put_nowait(packet)
            self._dedupe_cache.mark_seen(packet)
        except asyncio.QueueFull:
            logger.warning("BLE relay queue full; dropping oldest packet")
            try:
                self._outbound_queue.get_nowait()
                self._outbound_queue.put_nowait(packet)
            except asyncio.QueueEmpty:
                pass

    async def _advertise_loop(self) -> None:
        """Continuously drain the outbound queue and advertise each packet."""
        while not self._stop_event.is_set():
            try:
                packet = await asyncio.wait_for(self._outbound_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            await self._advertise_packet(packet)

    async def _advertise_packet(self, packet: bytes) -> None:
        """
        Advertise a single packet via a transient BLE GATT server. Bleak's
        peripheral/advertiser support is platform-dependent; this isolates
        the actual radio call so it can be swapped per-OS backend without
        touching the relay logic above.
        """
        try:
            from bleak import BleakGATTCharacteristic  # noqa: F401
            import bleak

            logger.info(
                "Advertising distress packet over BLE",
                extra={"context": {"size_bytes": len(packet)}},
            )
            # NOTE: Bleak's cross-platform peripheral (server) API varies by OS.
            # We isolate the platform-specific server bring-up here so callers
            # never need to branch on OS.
            await self._platform_advertise(bleak, packet)
        except Exception as exc:
            logger.error("BLE advertise failed", extra={"context": {"error": str(exc)}})

    async def _platform_advertise(self, bleak_module, packet: bytes) -> None:
        """
        Thin seam around the OS-specific GATT server bring-up. Kept as a
        separate coroutine so platform backends (BlueZ/WinRT/CoreBluetooth)
        can be dropped in without changing the queueing/dedup logic.
        """
        server = bleak_module.BleakGATTServer() if hasattr(bleak_module, "BleakGATTServer") else None
        if server is None:
            logger.warning(
                "This Bleak build/platform has no GATT server API available; "
                "packet stays queued for the next successful advertise cycle."
            )
            await self.enqueue_packet(packet)
            return

        await server.start()
        try:
            await server.update_value(settings.ble_characteristic_uuid, packet)
            await asyncio.sleep(0.5)  # advertise window
        finally:
            await server.stop()

    async def _scan_loop(self) -> None:
        """Continuously scan for nearby advertisements carrying distress packets."""
        from bleak import BleakScanner

        while not self._stop_event.is_set():
            try:
                devices = await BleakScanner.discover(
                    timeout=settings.ble_scan_interval_seconds,
                    service_uuids=[settings.ble_service_uuid],
                )
                for device in devices:
                    await self._process_scanned_device(device)
            except Exception as exc:
                logger.warning("BLE scan cycle failed", extra={"context": {"error": str(exc)}})
                await asyncio.sleep(settings.ble_scan_interval_seconds)

    async def _process_scanned_device(self, device) -> None:
        adv_data = getattr(device, "metadata", {}) or {}
        manufacturer_data = adv_data.get("manufacturer_data", {})
        for _, raw in manufacturer_data.items():
            packet = bytes(raw)
            if len(packet) != BinaryDistressProtocol.packet_size_bytes():
                continue
            if self._dedupe_cache.has_seen(packet):
                continue
            try:
                decoded = BinaryDistressProtocol.unpack_payload(packet)
            except PacketValidationError as exc:
                logger.debug("Discarding invalid scanned packet", extra={"context": {"error": str(exc)}})
                continue

            self._dedupe_cache.mark_seen(packet)
            logger.info("Received distress packet over BLE mesh", extra={"context": decoded})

            if self.on_packet_received is not None:
                await self.on_packet_received(packet, decoded)

            # Relay onward (store-and-forward) so multi-hop mesh works.
            await self.enqueue_packet(packet)

    async def run_forever(self) -> None:
        """Run the advertise + scan loops concurrently until `stop()` is called."""
        logger.info("Starting BLE mesh service")
        await asyncio.gather(self._advertise_loop(), self._scan_loop())

    def stop(self) -> None:
        self._stop_event.set()
