// ============================================
// NEXUS DESIGN SYSTEM TOKENS
// ============================================
// Single source of truth for all design tokens.
// Import these constants instead of hardcoding values.

// ─── Color Palette ───────────────────────────
export const colors = {
  // Core brand
  nexus: {
    blue: "#3B82F6",
    purple: "#8B5CF6",
    cyan: "#06B6D4",
    orange: "#F97316",
    green: "#10B981",
    pink: "#EC4899",
  },

  // Persona colors
  persona: {
    max: "#F97316",
    elena: "#8B5CF6",
    byte: "#10B981",
    stacy: "#3B82F6",
  },

  // Semantic colors
  semantic: {
    success: "#10B981",
    warning: "#F59E0B",
    error: "#EF4444",
    info: "#3B82F6",
  },

  // Background shades (dark theme)
  bg: {
    primary: "hsl(222, 47%, 5%)",
    card: "hsl(222, 47%, 7%)",
    muted: "hsl(217, 33%, 17%)",
    elevated: "hsl(222, 47%, 9%)",
  },

  // Text
  text: {
    primary: "hsl(210, 40%, 98%)",
    secondary: "hsl(215, 20%, 65%)",
    muted: "hsl(215, 20%, 45%)",
  },
} as const;

// ─── Typography ──────────────────────────────
export const typography = {
  fontFamily: {
    sans: 'Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
    mono: 'ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace',
  },

  fontSize: {
    xs: "0.75rem",    // 12px
    sm: "0.875rem",   // 14px
    base: "1rem",     // 16px
    lg: "1.125rem",   // 18px
    xl: "1.25rem",    // 20px
    "2xl": "1.5rem",  // 24px
    "3xl": "1.875rem",// 30px
    "4xl": "2.25rem", // 36px
    "5xl": "3rem",    // 48px
    "6xl": "3.75rem", // 60px
    "7xl": "4.5rem",  // 72px
  },

  fontWeight: {
    normal: "400",
    medium: "500",
    semibold: "600",
    bold: "700",
  },

  lineHeight: {
    tight: "1.25",
    snug: "1.375",
    normal: "1.5",
    relaxed: "1.625",
  },
} as const;

// ─── Spacing Scale ───────────────────────────
export const spacing = {
  0: "0",
  0.5: "0.125rem",   // 2px
  1: "0.25rem",      // 4px
  1.5: "0.375rem",   // 6px
  2: "0.5rem",       // 8px
  3: "0.75rem",      // 12px
  4: "1rem",         // 16px
  5: "1.25rem",      // 20px
  6: "1.5rem",       // 24px
  8: "2rem",         // 32px
  10: "2.5rem",      // 40px
  12: "3rem",        // 48px
  16: "4rem",        // 64px
  20: "5rem",        // 80px
  24: "6rem",        // 96px
} as const;

// ─── Border Radius ───────────────────────────
export const radii = {
  none: "0",
  sm: "0.375rem",    // 6px
  md: "0.5rem",      // 8px
  lg: "0.75rem",     // 12px
  xl: "1rem",        // 16px
  "2xl": "1.5rem",   // 24px
  "3xl": "2rem",     // 32px
  full: "9999px",
} as const;

// ─── Shadows ─────────────────────────────────
export const shadows = {
  sm: "0 1px 2px 0 rgb(0 0 0 / 0.05)",
  md: "0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1)",
  lg: "0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1)",
  xl: "0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1)",
  glow: {
    blue: "0 0 20px rgba(59, 130, 246, 0.3), 0 0 40px rgba(59, 130, 246, 0.2)",
    purple: "0 0 20px rgba(139, 92, 246, 0.3), 0 0 40px rgba(139, 92, 246, 0.2)",
    cyan: "0 0 20px rgba(6, 182, 212, 0.3), 0 0 40px rgba(6, 182, 212, 0.2)",
  },
} as const;

// ─── Motion / Animation ──────────────────────
export const motion = {
  duration: {
    fast: "150ms",
    normal: "200ms",
    slow: "300ms",
    slower: "500ms",
  },
  easing: {
    ease: "cubic-bezier(0.4, 0, 0.2, 1)",
    easeIn: "cubic-bezier(0.4, 0, 1, 1)",
    easeOut: "cubic-bezier(0, 0, 0.2, 1)",
    spring: "cubic-bezier(0.34, 1.56, 0.64, 1)",
  },
} as const;

// ─── Breakpoints ─────────────────────────────
export const breakpoints = {
  sm: "640px",
  md: "768px",
  lg: "1024px",
  xl: "1280px",
  "2xl": "1400px",
} as const;

// ─── Z-Index Scale ───────────────────────────
export const zIndex = {
  base: 0,
  dropdown: 10,
  sticky: 20,
  overlay: 30,
  modal: 40,
  popover: 50,
  toast: 60,
  tooltip: 70,
} as const;
