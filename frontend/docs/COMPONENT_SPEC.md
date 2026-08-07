# Component and State Specification

## Core components
- `AppShell`
- `SmartParkingHeader`
- `SummaryCards`
- `CameraHealthIndicator`
- `ParkingMap`
- `ParkingSpotShape`
- `ParkingLegend`
- `MapControls`
- `EntryChoiceSheet`
- `BrowseToolbar`
- `DestinationNeedSelector`
- `RecommendationPanel`
- `AlternativeSpotList`
- `SpotDetailSheet`
- `NavigationStatusBar`
- `InvalidSpotWarningSheet`
- `MockControlPanel` (development only)

## Stores
### Parking store
Owns:
- canonical spots by ID;
- camera health;
- last event time;
- stale revision protection;
- derived counts.

### Driver-flow store
Owns:
- mode;
- active need;
- current recommendation result;
- manually inspected spot;
- confirmed spot;
- warning state;
- browse filter.

## Suggested folder structure
```text
src/
  app/
  components/
  domain/
  geometry/
  mocks/
  recommendation/
  routing/
  stores/
  styles/
  tests/
```

## Responsive layout
### Mobile
- compact header;
- map is primary viewport;
- bottom sheets for entry, spot details, recommendation, and warnings;
- horizontal summary cards or compact grid;
- sticky primary action when appropriate.

### Desktop
- header + summary row;
- map centered and large;
- compact side/floating panel for recommendation or spot details;
- avoid permanent oversized sidebar.
