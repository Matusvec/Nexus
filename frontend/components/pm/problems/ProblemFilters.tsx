// ============================================
// ProblemFilters — Strategy §4.5
// ============================================
// Filter bar with dropdowns + search for problems page.

"use client";

import { useCallback, useEffect, useRef } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { X, Search } from "lucide-react";
import { useFilterStore } from "@/lib/pm/store";
import { FilterDropdown } from "@/components/pm/shared/FilterDropdown";

const severityOptions = [
  { value: "critical", label: "Critical" },
  { value: "high", label: "High" },
  { value: "medium", label: "Medium" },
  { value: "low", label: "Low" },
];

const personaOptions = [
  { value: "PM", label: "PM" },
  { value: "Admin", label: "Admin" },
  { value: "Developer", label: "Developer" },
  { value: "Analyst", label: "Analyst" },
  { value: "User", label: "User" },
];

export function ProblemFilters() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const searchRef = useRef<HTMLInputElement>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout>>(undefined);

  const {
    severity,
    persona,
    search,
    setSeverity,
    setPersona,
    setSearch,
    clearFilters,
    initFromParams,
  } = useFilterStore();

  // Init from URL on mount
  useEffect(() => {
    initFromParams({
      severity: searchParams.get("severity") ?? "",
      persona: searchParams.get("persona") ?? "",
      search: searchParams.get("search") ?? "",
    });
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Sync to URL on filter change
  const syncToUrl = useCallback(
    (params: Record<string, string>) => {
      const qs = new URLSearchParams();
      Object.entries(params).forEach(([k, v]) => {
        if (v) qs.set(k, v);
      });
      qs.set("page", "1"); // Reset to page 1 on filter change
      router.push(`/pm/problems?${qs.toString()}`);
    },
    [router]
  );

  const handleSeverityChange = (v: string) => {
    setSeverity(v);
    syncToUrl({ severity: v, persona, search });
  };

  const handlePersonaChange = (v: string) => {
    setPersona(v);
    syncToUrl({ severity, persona: v, search });
  };

  const handleSearchChange = (v: string) => {
    setSearch(v);
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      syncToUrl({ severity, persona, search: v });
    }, 300);
  };

  const handleClear = () => {
    clearFilters();
    router.push("/pm/problems");
  };

  const hasFilters = severity || persona || search;

  return (
    <div className="flex flex-wrap items-end gap-4">
      <FilterDropdown
        label="Severity"
        value={severity}
        options={severityOptions}
        onChange={handleSeverityChange}
      />

      <FilterDropdown
        label="Persona"
        value={persona}
        options={personaOptions}
        onChange={handlePersonaChange}
      />

      {/* Search */}
      <div className="flex flex-col gap-1">
        <label className="text-[11px] font-medium text-muted-foreground uppercase tracking-[0.1em]">
          Search
        </label>
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <input
            ref={searchRef}
            type="text"
            value={search}
            onChange={(e) => handleSearchChange(e.target.value)}
            placeholder="Search problems…"
            className="h-9 w-[240px] rounded-xl border border-input bg-background pl-9 pr-3 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
          />
        </div>
      </div>

      {/* Clear filters */}
      {hasFilters && (
        <button
          onClick={handleClear}
          className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors duration-150 pb-1.5"
        >
          <X className="h-3.5 w-3.5" />
          Clear filters
        </button>
      )}
    </div>
  );
}
