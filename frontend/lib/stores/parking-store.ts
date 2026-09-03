'use client';

import { useSyncExternalStore } from 'react';

import { makeDemoSnapshot } from '@/lib/demo-data';
import type {
  ConnectionStatus,
  RuntimeSnapshot,
  TrackingSource,
} from '@/lib/domain/types';

export interface ParkingState {
  readonly snapshot: RuntimeSnapshot;
  readonly connection: ConnectionStatus;
  readonly trackingSource: TrackingSource;
}

type Listener = () => void;

export class ParkingStore {
  private state: ParkingState;
  private readonly listeners = new Set<Listener>();

  constructor(snapshot: RuntimeSnapshot = makeDemoSnapshot()) {
    this.state = {
      snapshot,
      connection: {
        state: 'connecting',
        lastPublishedAt: snapshot.published_at,
        fetchError: null,
        failures: 0,
      },
      trackingSource: 'demo',
    };
  }

  getState = (): ParkingState => this.state;

  subscribe = (listener: Listener): (() => void) => {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  };

  private emit(): void {
    for (const listener of this.listeners) listener();
  }

  commit(snapshot: RuntimeSnapshot, now = Date.now() / 1_000): void {
    const stale = now - snapshot.published_at > 5;
    this.state = {
      ...this.state,
      snapshot,
      connection: {
        state: stale ? 'stale' : 'live',
        lastPublishedAt: snapshot.published_at,
        fetchError: null,
        failures: 0,
      },
      trackingSource: this.state.trackingSource === 'replay' ? 'replay' : 'live',
    };
    this.emit();
  }

  fail(error: unknown): void {
    const message = error instanceof Error ? error.message : 'Không thể kết nối Runtime API';
    this.state = {
      ...this.state,
      connection: {
        ...this.state.connection,
        state: 'error',
        fetchError: message,
        failures: this.state.connection.failures + 1,
      },
    };
    this.emit();
  }

  setSource(source: TrackingSource): void {
    this.state = { ...this.state, trackingSource: source };
    this.emit();
  }

  resetLocal(): void {
    this.state = {
      ...this.state,
      snapshot: {
        ...this.state.snapshot,
        frame_index: this.state.snapshot.frame_index + 1,
        vehicles: [],
      },
    };
    this.emit();
  }
}

export const parkingStore = new ParkingStore();

export function useParkingState(): ParkingState {
  return useSyncExternalStore(
    parkingStore.subscribe,
    parkingStore.getState,
    parkingStore.getState,
  );
}

