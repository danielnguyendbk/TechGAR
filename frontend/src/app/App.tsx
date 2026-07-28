import { useEffect, useMemo } from "react";
import { BrowseToolbar } from "../components/BrowseToolbar";
import { EntryChoiceSheet } from "../components/EntryChoiceSheet";
import { InvalidSpotWarningSheet } from "../components/InvalidSpotWarningSheet";
import { MockControlPanel } from "../components/MockControlPanel";
import { NavigationStatusBar } from "../components/NavigationStatusBar";
import { ParkingLegend } from "../components/ParkingLegend";
import { ParkingMap } from "../components/ParkingMap";
import { RecommendationPanel } from "../components/RecommendationPanel";
import { SmartParkingHeader } from "../components/SmartParkingHeader";
import { SpotDetailSheet } from "../components/SpotDetailSheet";
import { SummaryCards } from "../components/SummaryCards";
import type { DestinationNeed, ParkingSpotState, ParkingStatus, SpotId } from "../domain/parking";
import { mockParkingDataSource } from "../mocks/MockParkingDataSource";
import { recommendParkingSpots } from "../recommendation/recommendationEngine";
import { LANE_GRAPH } from "../routing/laneGraph";
import { findVehicleRoute } from "../routing/routeEngine";
import { useDriverFlowStore } from "../stores/driverFlowStore";
import { deriveParkingCounts, useParkingStore } from "../stores/parkingStore";

const NON_EMPTY_STATUSES: ReadonlySet<ParkingStatus> = new Set(["transitioning", "occupied", "unknown"]);

export function App() {
  const spotsById = useParkingStore((state) => state.spots);
  const cameras = useParkingStore((state) => state.cameras);
  const lastEventTime = useParkingStore((state) => state.lastEventTime);
  const applySnapshot = useParkingStore((state) => state.applySnapshot);
  const applyEvent = useParkingStore((state) => state.applyEvent);

  const mode = useDriverFlowStore((state) => state.mode);
  const browseFilter = useDriverFlowStore((state) => state.browseFilter);
  const activeNeed = useDriverFlowStore((state) => state.activeNeed);
  const recommendation = useDriverFlowStore((state) => state.recommendation);
  const inspectedSpotId = useDriverFlowStore((state) => state.inspectedSpotId);
  const confirmedSpotId = useDriverFlowStore((state) => state.confirmedSpotId);
  const warning = useDriverFlowStore((state) => state.warning);
  const enterBrowse = useDriverFlowStore((state) => state.enterBrowse);
  const startRecommendation = useDriverFlowStore((state) => state.startRecommendation);
  const chooseNeed = useDriverFlowStore((state) => state.chooseNeed);
  const setRecommendation = useDriverFlowStore((state) => state.setRecommendation);
  const chooseRecommendedSpot = useDriverFlowStore((state) => state.chooseRecommendedSpot);
  const inspectSpot = useDriverFlowStore((state) => state.inspectSpot);
  const confirmSpot = useDriverFlowStore((state) => state.confirmSpot);
  const setBrowseFilter = useDriverFlowStore((state) => state.setBrowseFilter);
  const showInvalidSpotWarning = useDriverFlowStore((state) => state.showInvalidSpotWarning);
  const cancelNavigation = useDriverFlowStore((state) => state.cancelNavigation);

  const spots = useMemo(
    () => Object.values(spotsById).filter((spot): spot is ParkingSpotState => spot !== undefined),
    [spotsById],
  );
  const counts = useMemo(() => deriveParkingCounts(spots), [spots]);
  const inspectedSpot = inspectedSpotId ? spotsById[inspectedSpotId] : undefined;
  const confirmedSpot = confirmedSpotId ? spotsById[confirmedSpotId] : undefined;

  useEffect(() => {
    let active = true;
    void mockParkingDataSource.getSnapshot().then((snapshot) => {
      if (active) applySnapshot(snapshot);
    });
    const unsubscribe = mockParkingDataSource.subscribe((event) => applyEvent(event));
    mockParkingDataSource.start();
    return () => {
      active = false;
      unsubscribe();
      mockParkingDataSource.stop();
    };
  }, [applyEvent, applySnapshot]);

  useEffect(() => {
    if (mode !== "recommendation" || !activeNeed || spots.length === 0) return;
    const result = recommendParkingSpots(spots, activeNeed, {
      calculatedAt: lastEventTime ?? "2026-07-25T08:00:00.000Z",
    });
    setRecommendation(result ?? undefined);
  }, [activeNeed, lastEventTime, mode, setRecommendation, spots]);

  useEffect(() => {
    if (mode !== "navigation" || !confirmedSpot || warning || !NON_EMPTY_STATUSES.has(confirmedSpot.status)) return;
    const need: DestinationNeed = activeNeed ?? "services";
    const alternatives = recommendParkingSpots(spots, need, {
      calculatedAt: lastEventTime ?? "2026-07-25T08:00:00.000Z",
    });
    const alternativeSpotId = [alternatives?.best, ...(alternatives?.alternatives ?? [])]
      .find((candidate) => candidate && candidate.spotId !== confirmedSpot.id)?.spotId;
    showInvalidSpotWarning({
      spotId: confirmedSpot.id,
      status: confirmedSpot.status as Exclude<ParkingStatus, "empty">,
      alternativeSpotId,
    });
  }, [activeNeed, confirmedSpot, lastEventTime, mode, showInvalidSpotWarning, spots, warning]);

  const route = useMemo(() => {
    if (mode !== "navigation" || !confirmedSpotId || !confirmedSpot || confirmedSpot.status !== "empty" || warning) return null;
    return findVehicleRoute(LANE_GRAPH, confirmedSpotId);
  }, [confirmedSpot, confirmedSpotId, mode, warning]);

  const handleSpotClick = (spotId: SpotId): void => {
    if (mode === "browse") {
      inspectSpot(spotId);
      return;
    }
    if (mode === "recommendation" && recommendation) {
      const isCandidate = [recommendation.best, ...recommendation.alternatives].some((spot) => spot.spotId === spotId);
      if (isCandidate) chooseRecommendedSpot(spotId);
    }
  };

  const handleNeedChange = (need: DestinationNeed): void => chooseNeed(need);

  const switchAlternative = (): void => {
    if (!warning?.alternativeSpotId) return;
    const alternative = spotsById[warning.alternativeSpotId];
    if (alternative?.status === "empty") confirmSpot(alternative.id);
  };

  return (
    <div className="app-shell">
      <SmartParkingHeader mode={mode} cameras={cameras} lastUpdated={lastEventTime} />
      <main>
        <SummaryCards counts={counts} cameras={cameras} />
        {mode === "browse" && (
          <BrowseToolbar filter={browseFilter} onFilterChange={setBrowseFilter} onFindSpot={startRecommendation} />
        )}
        {mode === "navigation" && confirmedSpot && (
          <NavigationStatusBar spotId={confirmedSpot.id} zone={confirmedSpot.zone} paused={Boolean(warning)} onCancel={cancelNavigation} />
        )}
        <div className={`parking-workspace parking-workspace--${mode}`}>
          <div className="map-column">
            <ParkingMap
              spots={spots}
              cameras={cameras}
              filter={mode === "browse" ? browseFilter : "all"}
              recommendation={mode === "recommendation" ? recommendation : undefined}
              inspectedSpotId={inspectedSpotId}
              confirmedSpotId={confirmedSpotId}
              activeNeed={activeNeed}
              route={route}
              routePaused={Boolean(warning)}
              onSpotClick={handleSpotClick}
            />
            <ParkingLegend />
          </div>

          {mode === "recommendation" && (
            <RecommendationPanel
              need={activeNeed}
              result={recommendation}
              onNeedChange={handleNeedChange}
              onChooseAlternative={chooseRecommendedSpot}
              onConfirm={confirmSpot}
              onAbandon={() => enterBrowse("all")}
            />
          )}
          {mode === "browse" && inspectedSpot && (
            <SpotDetailSheet
              spot={inspectedSpot}
              onClose={() => inspectSpot(undefined)}
              onNavigate={() => confirmedSpotId !== inspectedSpot.id && confirmSpot(inspectedSpot.id)}
            />
          )}
        </div>
      </main>

      {mode === "entry" && (
        <EntryChoiceSheet
          onRecommend={startRecommendation}
          onEmptyOnly={() => enterBrowse("empty")}
          onSkip={() => enterBrowse("all")}
        />
      )}
      {warning && (
        <InvalidSpotWarningSheet warning={warning} onSwitch={switchAlternative} onContinueMap={cancelNavigation} />
      )}
      {import.meta.env.DEV && (
        <MockControlPanel
          source={mockParkingDataSource}
          recommendedSpotId={recommendation?.best.spotId}
          selectedSpotId={confirmedSpotId}
        />
      )}
    </div>
  );
}
