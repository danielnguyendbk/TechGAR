import { create } from "zustand";
import type {
  BrowseFilter,
  DestinationNeed,
  DriverMode,
  InvalidSpotWarning,
  RecommendationResult,
  SpotId,
} from "../domain/parking";
import { promoteRecommendation } from "../recommendation/recommendationEngine";

export interface DriverFlowState {
  mode: DriverMode;
  browseFilter: BrowseFilter;
  activeNeed?: DestinationNeed;
  recommendation?: RecommendationResult;
  inspectedSpotId?: SpotId;
  confirmedSpotId?: SpotId;
  warning?: InvalidSpotWarning;
  enterBrowse: (filter?: BrowseFilter) => void;
  startRecommendation: () => void;
  chooseNeed: (need: DestinationNeed) => void;
  setRecommendation: (result?: RecommendationResult) => void;
  chooseRecommendedSpot: (spotId: SpotId) => void;
  inspectSpot: (spotId?: SpotId) => void;
  confirmSpot: (spotId: SpotId) => void;
  setBrowseFilter: (filter: BrowseFilter) => void;
  showInvalidSpotWarning: (warning: InvalidSpotWarning) => void;
  clearWarning: () => void;
  cancelNavigation: () => void;
  reset: () => void;
}

const initialState = {
  mode: "entry" as DriverMode,
  browseFilter: "all" as BrowseFilter,
  activeNeed: undefined,
  recommendation: undefined,
  inspectedSpotId: undefined,
  confirmedSpotId: undefined,
  warning: undefined,
};

export const useDriverFlowStore = create<DriverFlowState>((set) => ({
  ...initialState,
  enterBrowse: (filter = "all") =>
    set({
      mode: "browse",
      browseFilter: filter,
      activeNeed: undefined,
      recommendation: undefined,
      inspectedSpotId: undefined,
      confirmedSpotId: undefined,
      warning: undefined,
    }),
  startRecommendation: () =>
    set({
      mode: "recommendation",
      activeNeed: undefined,
      recommendation: undefined,
      inspectedSpotId: undefined,
      confirmedSpotId: undefined,
      warning: undefined,
    }),
  chooseNeed: (need) => set({ activeNeed: need, recommendation: undefined }),
  setRecommendation: (recommendation) => set({ recommendation }),
  chooseRecommendedSpot: (spotId) =>
    set((state) => ({
      recommendation: state.recommendation
        ? promoteRecommendation(state.recommendation, spotId)
        : state.recommendation,
    })),
  inspectSpot: (inspectedSpotId) => set({ inspectedSpotId }),
  confirmSpot: (confirmedSpotId) =>
    set({
      mode: "navigation",
      confirmedSpotId,
      inspectedSpotId: undefined,
      warning: undefined,
    }),
  setBrowseFilter: (browseFilter) => set({ browseFilter }),
  showInvalidSpotWarning: (warning) => set({ warning }),
  clearWarning: () => set({ warning: undefined }),
  cancelNavigation: () =>
    set({
      mode: "browse",
      activeNeed: undefined,
      recommendation: undefined,
      confirmedSpotId: undefined,
      warning: undefined,
    }),
  reset: () => set(initialState),
}));
