import type {
  DestinationNeed,
  ParkingSpotState,
  RecommendationResult,
  RankedSpot,
} from "../domain/parking";
import {
  PARKING_GEOMETRY,
  SPOT_GEOMETRY_BY_ID,
  distanceBetween,
  type ParkingGeometry,
} from "../geometry/parkingGeometry";
import { LANE_GRAPH, type LaneGraph } from "../routing/laneGraph";
import { findVehicleRoute } from "../routing/routeEngine";

const METERS_PER_MAP_UNIT = 0.12;
const WALKING_METERS_PER_MINUTE = 75;

const REASONS: Record<DestinationNeed, string> = {
  shopping: "Gần lối vào khu mua sắm",
  services: "Thuận tiện đến khu dịch vụ",
  entertainment: "Gần khu giải trí",
};

export interface RecommendationOptions {
  geometry?: ParkingGeometry;
  graph?: LaneGraph;
  calculatedAt?: string;
}

export function getEligibleSpots(spots: readonly ParkingSpotState[]): ParkingSpotState[] {
  return spots.filter((spot) => spot.status === "empty");
}

export function rankParkingSpots(
  spots: readonly ParkingSpotState[],
  need: DestinationNeed,
  options: RecommendationOptions = {},
): RankedSpot[] {
  const geometry = options.geometry ?? PARKING_GEOMETRY;
  const graph = options.graph ?? LANE_GRAPH;
  const anchor = geometry.anchors[need];

  return getEligibleSpots(spots)
    .map((spot): RankedSpot | null => {
      const spotGeometry = SPOT_GEOMETRY_BY_ID.get(spot.id);
      const route = findVehicleRoute(graph, spot.id);
      if (!spotGeometry || !route) return null;

      const drivingDistance = route.distance * METERS_PER_MAP_UNIT;
      const walkingDistance =
        distanceBetween(
          { x: spotGeometry.x + spotGeometry.width / 2, y: spotGeometry.y + spotGeometry.height / 2 },
          anchor,
        ) * METERS_PER_MAP_UNIT;
      const totalScore = drivingDistance * 0.35 + walkingDistance * 0.65;

      return {
        spotId: spot.id,
        zone: spot.zone,
        totalScore: Number(totalScore.toFixed(2)),
        drivingDistance: Math.round(drivingDistance),
        walkingDistance: Math.round(walkingDistance),
        estimatedWalkingMinutes: Math.max(1, Math.ceil(walkingDistance / WALKING_METERS_PER_MINUTE)),
        reason: REASONS[need],
      };
    })
    .filter((spot): spot is RankedSpot => spot !== null)
    .sort((a, b) => a.totalScore - b.totalScore || a.spotId.localeCompare(b.spotId));
}

export function recommendParkingSpots(
  spots: readonly ParkingSpotState[],
  need: DestinationNeed,
  options: RecommendationOptions = {},
): RecommendationResult | null {
  const ranked = rankParkingSpots(spots, need, options);
  const best = ranked[0];
  if (!best) return null;
  return {
    need,
    best,
    alternatives: ranked.slice(1, 3),
    calculatedAt: options.calculatedAt ?? new Date().toISOString(),
  };
}

export function promoteRecommendation(
  result: RecommendationResult,
  spotId: RankedSpot["spotId"],
): RecommendationResult {
  const selected = [result.best, ...result.alternatives].find((spot) => spot.spotId === spotId);
  if (!selected || selected.spotId === result.best.spotId) return result;
  return {
    ...result,
    best: selected,
    alternatives: [result.best, ...result.alternatives.filter((spot) => spot.spotId !== spotId)].slice(0, 2),
  };
}
