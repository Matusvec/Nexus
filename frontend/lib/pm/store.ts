import { create } from "zustand";
import type { JobStatusResponse } from "./types";

// ── Active jobs tracker (global — visible across PipelineIndicator, Dashboard, etc.) ──
interface JobsState {
  activeJobs: Map<string, JobStatusResponse>;
  setJob: (id: string, job: JobStatusResponse) => void;
  removeJob: (id: string) => void;
}

export const useJobsStore = create<JobsState>((set) => ({
  activeJobs: new Map(),
  setJob: (id, job) =>
    set((s) => {
      const next = new Map(s.activeJobs);
      next.set(id, job);
      return { activeJobs: next };
    }),
  removeJob: (id) =>
    set((s) => {
      const next = new Map(s.activeJobs);
      next.delete(id);
      return { activeJobs: next };
    }),
}));

// ── Problem / Evidence filters (synced with URL params) ──
// Usage pattern:
//   On page load, initialize from URL search params.
//   On filter change, update both store and URL via router.push().
//   This makes filter states bookmarkable and shareable.
interface FilterState {
  severity: string;
  persona: string;
  tag: string;
  search: string;
  setSeverity: (v: string) => void;
  setPersona: (v: string) => void;
  setTag: (v: string) => void;
  setSearch: (v: string) => void;
  clearFilters: () => void;
  initFromParams: (params: Record<string, string>) => void;
}

export const useFilterStore = create<FilterState>((set) => ({
  severity: "",
  persona: "",
  tag: "",
  search: "",
  setSeverity: (severity) => set({ severity }),
  setPersona: (persona) => set({ persona }),
  setTag: (tag) => set({ tag }),
  setSearch: (search) => set({ search }),
  clearFilters: () => set({ severity: "", persona: "", tag: "", search: "" }),
  initFromParams: (params) =>
    set({
      severity: params.severity ?? "",
      persona: params.persona ?? "",
      tag: params.tag ?? "",
      search: params.search ?? "",
    }),
}));

