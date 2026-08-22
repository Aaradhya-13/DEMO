"""
geospatial/hazard_router.py
------------------------------
Builds a live OSMnx/NetworkX street graph for the region around a distress
point, masks out every edge that intersects the dynamically-generated SAR
flood polygon, and computes obstacle-free A* shortest paths for rescue
boats (routed *through* flooded, non-structural terrain) and pedestrian /
dry-land evacuation (routed *around* flooded terrain).

Nothing about the graph extent, obstacle polygon, or speed profile is
hardcoded — all derive from the distress coordinates and `core.config.settings`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)


class RouteMode(str, Enum):
    BOAT = "boat"
    PEDESTRIAN = "pedestrian"


class RoutingError(RuntimeError):
    """Raised when a graph can't be built or no path exists between nodes."""


@dataclass(frozen=True)
class RouteResult:
    mode: RouteMode
    coordinates: list[tuple[float, float]]  # [(lat, lon), ...] in travel order
    distance_meters: float
    estimated_duration_minutes: float
    intersects_flood_polygon: bool


class HazardAwareRouter:
    """
    Builds (and caches per-AOI) an OSMnx graph, then computes A* shortest
    paths that respect a dynamic flood-polygon obstacle mask.
    """

    def __init__(self):
        self._graph_cache: dict[tuple[float, float], "object"] = {}

    # ---------------- Graph construction ----------------

    def _graph_cache_key(self, latitude: float, longitude: float) -> tuple[float, float]:
        # Round to ~100m grid so nearby distress points reuse the same cached graph.
        return (round(latitude, 3), round(longitude, 3))

    def _build_graph(self, latitude: float, longitude: float):
        import osmnx as ox

        cache_key = self._graph_cache_key(latitude, longitude)
        if cache_key in self._graph_cache:
            return self._graph_cache[cache_key]

        logger.info(
            "Building OSMnx graph",
            extra={
                "context": {
                    "center": (latitude, longitude),
                    "buffer_m": settings.routing_graph_buffer_meters,
                    "network_type": settings.routing_network_type,
                }
            },
        )
        try:
            graph = ox.graph_from_point(
                (latitude, longitude),
                dist=settings.routing_graph_buffer_meters,
                network_type=settings.routing_network_type,
                simplify=True,
            )
        except Exception as exc:
            raise RoutingError(f"Failed to build street graph from OSM: {exc}") from exc

        graph = ox.add_edge_speeds(graph)
        graph = ox.add_edge_travel_times(graph)

        self._graph_cache[cache_key] = graph
        return graph

    # ---------------- Flood-aware edge masking ----------------

    def _mask_flooded_edges(self, graph, flood_polygon_geojson: dict, mode: RouteMode):
        """
        Returns a copy of `graph` with edges intersecting the flood polygon
        removed (pedestrian mode) or with travel time re-weighted for water
        travel using the configured boat speed (boat mode).
        """
        from shapely.geometry import LineString, shape
        import copy

        flood_geom = shape(
            flood_polygon_geojson.get("geometry", flood_polygon_geojson)
        )

        working_graph = copy.deepcopy(graph)
        edges_to_drop = []

        for u, v, key, data in working_graph.edges(keys=True, data=True):
            geom = data.get("geometry")
            if geom is None:
                u_point = (working_graph.nodes[u]["x"], working_graph.nodes[u]["y"])
                v_point = (working_graph.nodes[v]["x"], working_graph.nodes[v]["y"])
                geom = LineString([u_point, v_point])

            intersects = geom.intersects(flood_geom)

            if mode == RouteMode.PEDESTRIAN and intersects:
                edges_to_drop.append((u, v, key))
            elif mode == RouteMode.BOAT and intersects:
                # Boats travel *through* flooded segments — rescale travel
                # time to the configured boat speed instead of road speed.
                length_m = data.get("length", geom.length)
                boat_speed_mps = (settings.routing_boat_speed_kmh * 1000) / 3600
                data["travel_time"] = length_m / boat_speed_mps if boat_speed_mps > 0 else math.inf
            elif mode == RouteMode.BOAT and not intersects:
                # Dry-land segments are impassable to a boat.
                edges_to_drop.append((u, v, key))

        working_graph.remove_edges_from(edges_to_drop)
        return working_graph

    # ---------------- A* pathfinding ----------------

    def route(
        self,
        origin_lat: float,
        origin_lon: float,
        destination_lat: float,
        destination_lon: float,
        flood_polygon_geojson: dict,
        mode: RouteMode = RouteMode.PEDESTRIAN,
    ) -> RouteResult:
        import networkx as ox_nx
        import osmnx as ox

        center_lat = (origin_lat + destination_lat) / 2
        center_lon = (origin_lon + destination_lon) / 2
        base_graph = self._build_graph(center_lat, center_lon)
        masked_graph = self._mask_flooded_edges(base_graph, flood_polygon_geojson, mode)

        try:
            origin_node = ox.nearest_nodes(masked_graph, origin_lon, origin_lat)
            dest_node = ox.nearest_nodes(masked_graph, destination_lon, destination_lat)
        except Exception as exc:
            raise RoutingError(f"Could not snap origin/destination to graph: {exc}") from exc

        def heuristic(n1, n2):
            y1, x1 = masked_graph.nodes[n1]["y"], masked_graph.nodes[n1]["x"]
            y2, x2 = masked_graph.nodes[n2]["y"], masked_graph.nodes[n2]["x"]
            return _haversine_meters(y1, x1, y2, x2)

        try:
            path = ox_nx.astar_path(
                masked_graph, origin_node, dest_node, heuristic=heuristic, weight="travel_time"
            )
        except ox_nx.NetworkXNoPath as exc:
            raise RoutingError(
                f"No obstacle-free {mode.value} path exists between the given points; "
                "the flood polygon may fully sever the graph."
            ) from exc

        coordinates = [
            (masked_graph.nodes[n]["y"], masked_graph.nodes[n]["x"]) for n in path
        ]

        total_distance = 0.0
        total_time_s = 0.0
        for u, v in zip(path[:-1], path[1:]):
            edge_data = min(
                masked_graph.get_edge_data(u, v).values(),
                key=lambda d: d.get("travel_time", math.inf),
            )
            total_distance += edge_data.get("length", 0.0)
            total_time_s += edge_data.get("travel_time", 0.0)

        from shapely.geometry import LineString, shape
        flood_geom = shape(flood_polygon_geojson.get("geometry", flood_polygon_geojson))
        route_line = LineString([(lon, lat) for lat, lon in coordinates])
        intersects = route_line.intersects(flood_geom)

        result = RouteResult(
            mode=mode,
            coordinates=coordinates,
            distance_meters=round(total_distance, 1),
            estimated_duration_minutes=round(total_time_s / 60, 1),
            intersects_flood_polygon=intersects,
        )
        logger.info(
            "Route computed",
            extra={
                "context": {
                    "mode": mode.value,
                    "distance_m": result.distance_meters,
                    "duration_min": result.estimated_duration_minutes,
                }
            },
        )
        return result


def _haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))
