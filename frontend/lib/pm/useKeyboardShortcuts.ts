// ============================================
// useKeyboardShortcuts — Global navigation shortcuts — Strategy §7
// ============================================
// Number keys 1-7 navigate to pipeline pages.
// "/" focuses search field, Esc closes panels.

"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

const NAV_KEYS: Record<string, string> = {
  "1": "/pm",
  "2": "/pm/evidence",
  "3": "/pm/problems",
  "4": "/pm/clusters",
  "5": "/pm/proposals",
  "6": "/pm/tasks",
  "7": "/pm/roadmap",
};

export function useKeyboardShortcuts() {
  const router = useRouter();

  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      const tag = (e.target as HTMLElement)?.tagName;
      // Don't intercept when user is typing in an input, textarea, or contenteditable
      if (
        tag === "INPUT" ||
        tag === "TEXTAREA" ||
        tag === "SELECT" ||
        (e.target as HTMLElement)?.isContentEditable
      ) {
        return;
      }

      // Number key navigation (1-7)
      if (NAV_KEYS[e.key] && !e.metaKey && !e.ctrlKey && !e.altKey) {
        e.preventDefault();
        router.push(NAV_KEYS[e.key]);
        return;
      }

      // "/" to focus search field
      if (e.key === "/" && !e.metaKey && !e.ctrlKey) {
        const searchInput = document.querySelector<HTMLInputElement>(
          '[data-search-field="true"]'
        );
        if (searchInput) {
          e.preventDefault();
          searchInput.focus();
        }
        return;
      }
    }

    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [router]);
}
