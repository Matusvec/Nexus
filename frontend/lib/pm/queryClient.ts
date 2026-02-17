// ============================================
// TanStack Query Configuration — Strategy §8
// ============================================

"use client";

import { QueryClient } from "@tanstack/react-query";

let queryClient: QueryClient | null = null;

export function getQueryClient(): QueryClient {
  if (!queryClient) {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: {
          staleTime: 30_000, // 30 seconds before refetch
          gcTime: 5 * 60_000, // 5 minutes garbage collection
          retry: 1, // One retry on failure
          refetchOnWindowFocus: false, // PM tool, not real-time
        },
        mutations: {
          retry: 0, // No retry on mutations
        },
      },
    });
  }
  return queryClient;
}
