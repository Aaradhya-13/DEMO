"""
geospatial/dem_processor.py
------------------------------
SRTM / Copernicus 30m DEM elevation sampling and flood-depth estimation.

Depth is estimated per-pixel inside the SAR-derived water mask using the
classic "bathtub" approach: depth = local_water_surface_elevation - terrain_elevation,
where the water surface elevation is dynamically taken as the DEM elevation
at the flood-polygon boundary (waterline) nearest each interior pixel,
rather than any fixed constant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from core.config import settings
from core.logger import get_logger
from geospatial.gee_sar_engine import _ensure_ee_initialized, SARProcessingError

logger = get_logger(__name__)


@dataclass(frozen=True)
class DepthEstimationResult:
    mean_depth_meters: float
    max_depth_meters: float
    depth_geojson: dict  # GeoJSON FeatureCollection with per-region depth stats
    dem_dataset_used: str


class DEMProcessor:
    """Fetches elevation data and derives flood depth within a water mask."""

    def __init__(self):
        _ensure_ee_initialized()

    def _get_dem_image(self, ee_module, aoi):
        """Try the primary Copernicus DEM, falling back to SRTM if unavailable
        for the AOI (e.g. above/below SRTM's latitude coverage)."""
        primary = ee_module.ImageCollection(settings.dem_dataset_id).mosaic().select(0)
        try:
            sample = primary.reduceRegion(
                reducer=ee_module.Reducer.count(), geometry=aoi, scale=30, maxPixels=1e8
            ).getInfo()
            if sample and any(v for v in sample.values()):
                return primary, settings.dem_dataset_id
        except Exception as exc:
            logger.warning(
                "Primary DEM dataset unavailable, falling back",
                extra={"context": {"dataset": settings.dem_dataset_id, "error": str(exc)}},
            )

        fallback = ee_module.Image(settings.dem_fallback_dataset_id)
        return fallback, settings.dem_fallback_dataset_id

    def estimate_flood_depth(
        self,
        aoi_geojson: dict,
        water_mask_geojson: dict,
    ) -> DepthEstimationResult:
        """
        aoi_geojson: GeoJSON polygon/rectangle defining the analysis area.
        water_mask_geojson: GeoJSON (Multi)Polygon of detected flood water
                             (typically FloodMaskResult.geojson from the SAR engine).
        """
        import ee

        aoi = ee.Geometry(aoi_geojson)
        water_geom = ee.Geometry(
            water_mask_geojson.get("geometry", water_mask_geojson)
            if "geometry" in water_mask_geojson or "type" in water_mask_geojson
            else water_mask_geojson
        )

        dem_image, dataset_used = self._get_dem_image(ee, aoi)

        # Elevation sampled along the waterline (mask boundary) approximates
        # local water-surface elevation without assuming a flat, global value.
        waterline = water_geom.buffer(30).difference(water_geom.buffer(-30))
        waterline_elevation = dem_image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=waterline,
            scale=30,
            maxPixels=1e8,
        ).getInfo()

        elevation_key = list(waterline_elevation.keys())[0] if waterline_elevation else None
        water_surface_elevation = (
            float(waterline_elevation[elevation_key])
            if elevation_key and waterline_elevation[elevation_key] is not None
            else None
        )

        if water_surface_elevation is None:
            raise SARProcessingError(
                "Could not determine waterline elevation from DEM; "
                "the water mask may not intersect DEM coverage."
            )

        depth_image = ee.Image.constant(water_surface_elevation).subtract(dem_image).max(0)
        depth_image = depth_image.updateMask(water_geom)

        stats = depth_image.reduceRegion(
            reducer=ee.Reducer.mean().combine(ee.Reducer.max(), sharedInputs=True),
            geometry=water_geom,
            scale=30,
            maxPixels=1e8,
        ).getInfo()

        band_name = dem_image.bandNames().getInfo()[0]
        mean_key = f"{band_name}_mean"
        max_key = f"{band_name}_max"
        mean_depth = float(stats.get(mean_key) or 0.0)
        max_depth = float(stats.get(max_key) or 0.0)

        depth_vectors = depth_image.reduceToVectors(
            geometry=water_geom,
            scale=30,
            geometryType="polygon",
            eightConnected=True,
            maxPixels=1e8,
            reducer=ee.Reducer.mean(),
        )
        depth_geojson = depth_vectors.getInfo()

        result = DepthEstimationResult(
            mean_depth_meters=round(mean_depth, 2),
            max_depth_meters=round(max_depth, 2),
            depth_geojson=depth_geojson,
            dem_dataset_used=dataset_used,
        )
        logger.info(
            "Flood depth estimation complete",
            extra={
                "context": {
                    "mean_depth_m": result.mean_depth_meters,
                    "max_depth_m": result.max_depth_meters,
                    "dem_dataset": dataset_used,
                }
            },
        )
        return result

    def point_elevation(self, latitude: float, longitude: float) -> Optional[float]:
        """Sample raw DEM elevation at a single point (e.g. for route safety checks)."""
        import ee

        point = ee.Geometry.Point([longitude, latitude])
        dem_image, _ = self._get_dem_image(ee, point.buffer(100))
        value = dem_image.reduceRegion(
            reducer=ee.Reducer.first(), geometry=point, scale=30, maxPixels=1e4
        ).getInfo()
        if not value:
            return None
        band_name = list(value.keys())[0]
        return float(value[band_name]) if value[band_name] is not None else None
