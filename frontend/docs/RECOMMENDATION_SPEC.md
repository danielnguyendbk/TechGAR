# Recommendation Specification

## Destination needs
Use only:
```ts
export type DestinationNeed = "shopping" | "services" | "entertainment";
```

Display labels:
- shopping → `Shopping`
- services → `Dịch vụ`
- entertainment → `Giải trí`

Do not include sub-destinations in this phase.

## Access anchors
Represent each need by one or more abstract pedestrian-access anchors around the parking lot:
- Shopping: favor lower/central mall access, configurable in data.
- Dịch vụ: favor central/easy-access anchor.
- Giải trí: favor upper mall access near zone E.

Exact coordinates are mock product data, not real-world mall coordinates.

## Eligibility
Candidate if and only if:
```ts
spot.status === "empty"
```

Exclude transitioning, occupied, and unknown.

## Score
Use a transparent deterministic score:
```ts
totalScore = drivingDistance * 0.35 + walkingDistance * 0.65 + congestionPenalty;
```

Where:
- drivingDistance is route distance from entrance to spot lane node;
- walkingDistance is lane/pedestrian distance from spot to need anchor;
- congestionPenalty is deterministic and may be zero in the initial build.

Lower score is better.

## Output
```ts
export interface RecommendationResult {
  need: DestinationNeed;
  best: RankedSpot;
  alternatives: [RankedSpot, RankedSpot] | RankedSpot[];
  calculatedAt: string;
}
```

Each ranked spot includes:
- spotId
- zone
- totalScore
- drivingDistance
- walkingDistance
- estimatedWalkingMinutes
- reason

## UI behavior
- Calculate after the user chooses a need.
- Highlight best spot and alternatives.
- Do not draw route yet.
- Draw route only after explicit confirmation.
- If best becomes invalid before confirmation, recalculate automatically.
- If confirmed spot becomes invalid, do not auto-switch; ask the user.
