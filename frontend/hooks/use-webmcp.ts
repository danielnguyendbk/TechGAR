'use client';

import { useEffect } from 'react';

interface WebMcpTool {
  readonly name: string;
  readonly title?: string;
  readonly description: string;
  readonly inputSchema: Record<string, unknown>;
  readonly annotations?: {
    readonly readOnlyHint?: boolean;
    readonly untrustedContentHint?: boolean;
  };
  execute(input: unknown): unknown;
}

interface ModelContext {
  registerTool(tool: WebMcpTool, options?: { signal?: AbortSignal }): void | Promise<void>;
}

declare global {
  interface Document {
    readonly modelContext?: ModelContext;
  }
}

export function useWebMcp(tools: readonly WebMcpTool[]): void {
  useEffect(() => {
    const context = document.modelContext;
    if (!context?.registerTool) return;
    const controller = new AbortController();
    for (const tool of tools) {
      try {
        void Promise.resolve(context.registerTool(tool, { signal: controller.signal })).catch(() => undefined);
      } catch {
        // Optional browser capability: visible UI remains the source of truth.
      }
    }
    return () => controller.abort();
  }, [tools]);
}
