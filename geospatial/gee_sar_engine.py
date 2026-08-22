"""
geospatial/gee_sar_engine.py
-------------------------------
Google Earth Engine Sentinel-1 SAR ingestion and dynamic Otsu-thresholded
flood-water mask generation.

Pipeline (all parameters sourced from `core.config.settings`, nothing
hardcoded per-request except the incoming distress coordinates themselves):

  1. Initialize GEE via service account JSON key or OAuth token.
  2. Build an AOI buffer around the distress coordinates.
  3. Fetch a pre-flood baseline Sentinel-1 GRD composite and a post-flood
     composite over parameterized time windows.
  4. Apply a Lee speckle filter to both composites.
  5. Compute the VV/VH backscatter difference (post - baseline).
  6. Sample the difference image's histogram and compute the Otsu threshold
     dynamically (no fixed dB cutoff).
  7. Threshold to a binary water mask, vectorize to a GeoJSON MultiPolygon.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np

from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)


class GEEInitializationError(RuntimeError):
    """Raised when Earth Engine cannot be authenticated/initialized."""


class SARProcessingError(RuntimeError):
    """Raised when SAR retrieval or processing fails."""


@dataclass(frozen=True)
class FloodMaskResult:
    geojson: dict  # GeoJSON MultiPolygon Feature
    otsu_threshold_db: float
    aoi_bbox: tuple[float, float, float, float]  # (minLon, minLat, maxLon, maxLat)
    baseline_window: tuple[str, str]
    postflood_window: tuple[str, str]
    pixel_count_water: int
    pixel_count_total: int

    @property
    def water_fraction(self) -> float:
        if self.pixel_count_total == 0:
            return 0.0
        return self.pixel_count_water / self.pixel_count_total


_ee_initialized = False


def _ensure_ee_initialized() -> None:
    """Idempotently authenticate + initialize the Earth Engine Python API."""
    global _ee_initialized
    if _ee_initialized:
        return

    import ee

    try:
        if settings.gee_use_oauth:
            ee.Initialize(project=settings.gee_project_id)
        else:
            if not settings.gee_service_account_email or not settings.gee_service_account_key_path:
                raise GEEInitializationError(
                    "GEE_SERVICE_ACCOUNT_EMAIL and GEE_SERVICE_ACCOUNT_KEY_PATH must be set "
                    "when GEE_USE_OAUTH is false."
                )
            credentials = ee.ServiceAccountCredentials(
                settings.gee_service_account_email,
                str(settings.gee_service_account_key_path),
            )
            ee.Initialize(credentials, project=settings.gee_project_id)
        _ee_initialized = True
        logger.info("Google Earth Engine initialized")
    except Exception as exc:
        raise GEEInitializationError(f"Failed to initialize Earth Engine: {exc}") from exc


def _build_aoi(ee_module, latitude: float, longitude: float):
    """Buffer the distress point into an AOI geometry, sized via config."""
    point = ee_module.Geometry.Point([longitude, latitude])
    return point.buffer(settings.sar_aoi_buffer_meters).bounds()


def _lee_filter(ee_module, image, kernel_size: int):
    """
    Apply a Lee speckle filter to a SAR image band using a moving-window
    mean/variance ratio, entirely in Earth Engine's server-side raster algebra
    (kernel size is dynamic/configurable, not fixed).
    """
    kernel = ee_module.Kernel.square(kernel_size // 2)
    mean = image.reduceNeighborhood(
        reducer=ee_module.Reducer.mean(), kernel=kernel
    )
    variance = image.reduceNeighborhood(
        reducer=ee_module.Reducer.variance(), kernel=kernel
    )
    # Lee filter: out = mean + weight * (pixel - mean), weight from local
    # coefficient of variation vs. the image-wide noise variance.
    overall_variance = image.reduceRegion(
        reducer=ee_module.Reducer.variance(), scale=10, maxPixels=1e9
    )
    noise_var = ee_module.Number(overall_variance.values().get(0))
    weight = variance.subtract(noise_var).divide(variance).clamp(0, 1)
    filtered = mean.add(image.subtract(mean).multiply(weight))
    return filtered.rename(image.bandNames())


def _sentinel1_composite(ee_module, aoi, start: str, end: str, polarizations: list[str]):
    collection = (
        ee_module.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee_module.Filter.eq("instrumentMode", "IW"))
    )
    for pol in polarizations:
        collection = collection.filter(
            ee_module.Filter.listContains("transmitterReceiverPolarisation", pol)
        )
    if collection.size().getInfo() == 0:
        raise SARProcessingError(
            f"No Sentinel-1 GRD scenes found for AOI between {start} and {end} "
            f"(polarizations={polarizations}). Try widening SAR_*_WINDOW_DAYS."
        )
    composite = collection.select(polarizations).median()
    return _lee_filter(ee_module, composite, settings.sar_speckle_filter_kernel_size)


def _compute_otsu_threshold(histogram_dict: dict) -> float:
    """
    Compute the Otsu threshold from an Earth Engine histogram
    (bucketMeans / histogram counts) using between-class variance
    maximization — no arbitrary fixed dB cutoff.
    """
    counts = np.array(histogram_dict["histogram"], dtype=np.float64)
    bucket_means = np.array(histogram_dict["bucketMeans"], dtype=np.float64)

    total = counts.sum()
    if total == 0:
        raise SARProcessingError("Empty histogram; cannot compute Otsu threshold.")

    sum_total = (counts * bucket_means).sum()
    sum_background, weight_background, max_variance, threshold = 0.0, 0.0, 0.0, bucket_means[0]

    for count, mean in zip(counts, bucket_means):
        weight_background += count
        if weight_background == 0:
            continue
        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break
        sum_background += mean * count
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        between_class_variance = (
            weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        )
        if between_class_variance > max_variance:
            max_variance = between_class_variance
            threshold = mean

    return float(threshold)


class SARFloodEngine:
    """High-level entrypoint: distress coordinates -> flood-water GeoJSON mask."""

    def __init__(self):
        _ensure_ee_initialized()

    def generate_flood_mask(
        self,
        latitude: float,
        longitude: float,
        reference_time: Optional[datetime] = None,
    ) -> FloodMaskResult:
        import ee

        reference_time = reference_time or datetime.now(timezone.utc)

        postflood_end = reference_time
        postflood_start = reference_time - timedelta(days=settings.sar_postflood_window_days)
        baseline_end = postflood_start
        baseline_start = baseline_end - timedelta(days=settings.sar_baseline_window_days)

        fmt = "%Y-%m-%d"
        postflood_window = (postflood_start.strftime(fmt), postflood_end.strftime(fmt))
        baseline_window = (baseline_start.strftime(fmt), baseline_end.strftime(fmt))

        aoi = _build_aoi(ee, latitude, longitude)
        polarizations = settings.sar_polarization_list

        logger.info(
            "Fetching Sentinel-1 composites",
            extra={
                "context": {
                    "baseline_window": baseline_window,
                    "postflood_window": postflood_window,
                    "polarizations": polarizations,
                }
            },
        )

        baseline_img = _sentinel1_composite(ee, aoi, *baseline_window, polarizations)
        postflood_img = _sentinel1_composite(ee, aoi, *postflood_window, polarizations)

        # Backscatter drop (dB) indicates new open water (specular reflection).
        difference = baseline_img.subtract(postflood_img).select(polarizations[0])

        histogram = difference.reduceRegion(
            reducer=ee.Reducer.histogram(maxBuckets=256),
            geometry=aoi,
            scale=10,
            maxPixels=1e9,
        ).get(polarizations[0])

        histogram_info = ee.Dictionary(histogram).getInfo()
        if not histogram_info or "histogram" not in histogram_info:
            raise SARProcessingError("Earth Engine returned an empty histogram for the AOI difference image.")

        otsu_threshold = _compute_otsu_threshold(histogram_info)

        water_mask = difference.gt(otsu_threshold).selfMask()

        vectors = water_mask.reduceToVectors(
            geometry=aoi,
            scale=10,
            geometryType="polygon",
            eightConnected=True,
            maxPixels=1e9,
        )

        geojson = vectors.getInfo()

        pixel_stats = water_mask.reduceRegion(
            reducer=ee.Reducer.count(), geometry=aoi, scale=10, maxPixels=1e9
        ).getInfo()
        total_stats = difference.reduceRegion(
            reducer=ee.Reducer.count(), geometry=aoi, scale=10, maxPixels=1e9
        ).getInfo()

        pixel_count_water = int(pixel_stats.get(polarizations[0], 0) or 0)
        pixel_count_total = int(total_stats.get(polarizations[0], 0) or 0)

        bbox_coords = aoi.bounds().getInfo()["coordinates"][0]
        lons = [c[0] for c in bbox_coords]
        lats = [c[1] for c in bbox_coords]
        bbox = (min(lons), min(lats), max(lons), max(lats))

        result = FloodMaskResult(
            geojson=geojson,
            otsu_threshold_db=otsu_threshold,
            aoi_bbox=bbox,
            baseline_window=baseline_window,
            postflood_window=postflood_window,
            pixel_count_water=pixel_count_water,
            pixel_count_total=pixel_count_total,
        )
        logger.info(
            "Flood mask generated",
            extra={
                "context": {
                    "otsu_threshold_db": otsu_threshold,
                    "water_fraction": result.water_fraction,
                }
            },
        )
        return result
