// ============================================
// KeyboardShortcutsProvider — Mounts global keyboard shortcuts — Strategy §7
// ============================================

"use client";

import { useKeyboardShortcuts } from "@/lib/pm/useKeyboardShortcuts";

export function KeyboardShortcutsProvider() {
  useKeyboardShortcuts();
  return null;
}
