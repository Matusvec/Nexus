# Nexus PM — Frontend Strategy & Implementation Blueprint

> **Evidence to Roadmap. Signal to Strategy. Noise to Clarity.**
>
> A precision-engineered pipeline interface that transforms raw customer evidence into prioritized, dev-ready product decisions — with full provenance at every step.

---

## Table of Contents

1. [Design Philosophy](#1-design-philosophy)
2. [Brand & Visual Identity System](#2-brand--visual-identity-system)
3. [Layout System & Design Architecture](#3-layout-system--design-architecture)
4. [Page-Level Strategy](#4-page-level-strategy)
5. [Component Architecture](#5-component-architecture)
6. [Interaction & Animation System](#6-interaction--animation-system)
7. [User Experience & Navigation Strategy](#7-user-experience--navigation-strategy)
8. [State Management & Data Layer](#8-state-management--data-layer)
9. [Technical Implementation Strategy](#9-technical-implementation-strategy)
10. [Engagement Strategy](#10-engagement-strategy)
11. [Quality Bar](#11-quality-bar)
12. [Development Phases](#12-development-phases)

---

## 1. Design Philosophy

### Emotional Tone

Nexus PM should feel like a **trusted analyst** — confident enough to surface hard truths, intelligent enough to find the signal, fast enough to never block you, and polished enough that you trust every pixel.

| Attribute | Expression |
|---|---|
| **Confident** | Bold typography hierarchy, decisive color usage, no wishy-washy "maybe" UI. Every element has a reason |
| **Intelligent** | Data density without clutter. Severity distributions, priority scores, and citation links feel like an analyst's desk, not a reporting dashboard |
| **Fast** | Instant navigation via client-side transitions. Skeleton states instead of blank screens. Optimistic updates on mutations. Perceived performance > actual performance |
| **Premium** | Micro-shadows on cards, consistent 4px radius increments, system-level font rendering, precise spacing grid. The kind of fit-and-finish that makes PMs screenshot their own tool |

### UX Principles

1. **Clarity over clutter** — Show one clear hierarchy per screen. If a user can't identify the primary action within 2 seconds, the page needs a redesign.
2. **Guided interaction** — Every page implies a next step. Evidence → "Extract Problems." Cluster → "Generate Proposal." Proposal → "Approve & Generate Tasks." The pipeline is the UX.
3. **Minimal friction** — Upload should be < 3 clicks. Navigation should never exceed 2 levels deep. Filters persist across sessions. Back buttons always work.
4. **Evidence is king** — Every data point traces back to a verbatim quote. Every quote traces to a source document. Citation links are first-class citizens, not footnotes.
5. **Async-aware, never blocked** — LLM operations take 5–30 seconds. The UI shows progress bars with estimated completion, toast notifications on finish, and lets users navigate freely while jobs run.
6. **Density over decoration** — PMs scan 50+ rows of data. Tables are the primary pattern. Cards are for overview screens. White space serves readability, not aesthetics.

### How the Frontend Supports Product Value

The core value proposition is: **"From raw signal to ranked roadmap in minutes, not weeks."**

The frontend makes this real by:
- Making upload instant (paste text, drop file, click submit — done)
- Showing extraction results within the evidence context (problems appear where they were found)
- Visualizing clusters as pain magnitude (bigger cluster card = bigger pain = higher priority)
- Rendering proposals as structured specs with clickable citations (every claim verified)
- Presenting the roadmap as a transparent priority formula (users can adjust weights and understand why Feature A outranks Feature B)

---

## 2. Brand & Visual Identity System

### Color Strategy

Nexus PM uses a **warm-neutral light theme** with strategic yellow and blue accents. Yellow represents **insight and energy** (the product's output). Blue represents **trust and depth** (the product's intelligence). They are used sparingly — as signals, not wallpaper.

#### Primary Palette

| Token | Hex | HSL | Usage |
|---|---|---|---|
| `--nexus-blue` | `#0E7490` | `190 82% 31%` | Primary CTAs, active sidebar items, links, pipeline "complete" indicators |
| `--nexus-yellow` | `#E88C0A` | `36 90% 47%` | Accent highlights, "running" status, notification badges, score accents |
| `--nexus-amber` | `#F59E0B` | `38 92% 50%` | Secondary accent, hover state lift on yellow elements |

#### Neutral Palette

| Token | Hex | HSL | Usage |
|---|---|---|---|
| `--surface-0` | `#FAFAF6` | `60 22% 97%` | Page background (warm off-white, avoids clinical white) |
| `--surface-1` | `#F5F3EE` | `40 24% 95%` | Card backgrounds, sidebar bg |
| `--surface-2` | `#EBE8E0` | `38 22% 90%` | Input backgrounds, muted section fills |
| `--surface-3` | `#DDD9CE` | `38 18% 84%` | Borders, dividers |
| `--ink-primary` | `#1A2332` | `215 30% 15%` | Headings, primary text |
| `--ink-secondary` | `#4A5568` | `215 15% 35%` | Body text, descriptions |
| `--ink-muted` | `#8A9AB5` | `215 20% 62%` | Captions, timestamps, placeholders |

#### Semantic Colors

| Token | Hex | Usage |
|---|---|---|
| `--severity-critical` | `#DC2626` | Critical severity badges, destructive actions |
| `--severity-high` | `#EA580C` | High severity |
| `--severity-medium` | `#D97706` | Medium severity (warm amber, not yellow — avoids brand conflict) |
| `--severity-low` | `#16A34A` | Low severity, success states |
| `--status-draft` | `#3B82F6` | Draft proposals |
| `--status-approved` | `#16A34A` | Approved proposals |
| `--status-rejected` | `#9CA3AF` | Rejected proposals |
| `--status-running` | `#E88C0A` | Active jobs, processing |
| `--status-error` | `#DC2626` | Failed jobs, errors |

#### Color Usage Rules

1. **Yellow never as background fill** — Only as text accents, small badges, progress bar fills, and score highlights. Full yellow backgrounds look cheap.
2. **Blue as the primary action color** — All primary buttons, links, and interactive elements use `--nexus-blue`. Hover state shifts to `#0C6577` (10% darker).
3. **Neutral backgrounds create depth** — `surface-0` for the page, `surface-1` for cards, `surface-2` for nested containers. This 3-layer system creates visual hierarchy without borders.
4. **Severity colors are status-only** — Never use severity reds/oranges for branding or decorative purposes. They are reserved exclusively for data meaning.
5. **Gradients: subtle and purposeful** — The background gradient on the PM layout uses `radial-gradient` with `nexus-blue` at 8% opacity and `nexus-yellow` at 6% opacity. This creates warmth without distraction.

#### CSS Variable Implementation (globals.css `.pm-root` override)

```css
.pm-root {
  --background: 48 22% 97%;          /* surface-0 */
  --foreground: 215 30% 15%;         /* ink-primary */
  --card: 40 24% 95%;                /* surface-1 */
  --card-foreground: 215 30% 15%;
  --popover: 40 24% 95%;
  --popover-foreground: 215 30% 15%;
  --primary: 190 82% 31%;            /* nexus-blue */
  --primary-foreground: 0 0% 100%;
  --secondary: 40 24% 91%;           /* surface-2-ish */
  --secondary-foreground: 215 30% 15%;
  --muted: 38 18% 90%;               /* surface-2 */
  --muted-foreground: 215 15% 40%;   /* ink-secondary */
  --accent: 36 90% 47%;              /* nexus-yellow */
  --accent-foreground: 0 0% 100%;
  --destructive: 0 72% 51%;
  --destructive-foreground: 0 0% 100%;
  --border: 38 18% 84%;              /* surface-3 */
  --input: 38 22% 90%;
  --ring: 190 82% 31%;
}
```

### Typography

#### Font Pairing

| Role | Font | Weight | Why |
|---|---|---|---|
| **Display** (h1, h2) | `Fraunces` (variable, optical size) | 600–700 | High-personality serif. Signals editorial authority — appropriate for a tool that synthesizes documents. Google Fonts variable, already loaded. |
| **Body / UI** | `IBM Plex Sans` | 400, 500, 600 | Engineered for data-dense interfaces. Excellent legibility at 13–14px. Variable width numbers for tables. Already loaded. |
| **Code / Monospace** | `IBM Plex Mono` | 400 | Technical contexts (API keys, job IDs, JSON export). Same family as body for cohesion. |

#### Type Scale (based on 16px root)

| Level | Font | Size | Weight | Line Height | Letter Spacing | Usage |
|---|---|---|---|---|---|---|
| **Page Title** | Fraunces | 30px / `1.875rem` | 600 | 1.2 | −0.02em | One per page: "Evidence", "Roadmap", etc. |
| **Section Head** | Fraunces | 22px / `1.375rem` | 600 | 1.3 | −0.015em | Card group titles, detail page sections |
| **Card Title** | IBM Plex Sans | 16px / `1rem` | 600 | 1.4 | −0.01em | Evidence title in table, cluster card label |
| **Body** | IBM Plex Sans | 14px / `0.875rem` | 400 | 1.6 | 0 | Descriptions, rationale text, summaries |
| **Body Strong** | IBM Plex Sans | 14px / `0.875rem` | 500 | 1.6 | 0 | Severity labels, score values, inline emphasis |
| **Caption** | IBM Plex Sans | 12px / `0.75rem` | 400 | 1.5 | 0 | Timestamps, source attribution, meta info |
| **Overline** | IBM Plex Sans | 11px / `0.6875rem` | 500 | 1.0 | 0.15em | "PM PIPELINE", "EVIDENCE", section labels (uppercase tracking) |
| **Badge** | IBM Plex Sans | 11px / `0.6875rem` | 600 | 1.0 | 0.02em | Severity badges, status pills, effort tags |

#### Spacing Rhythm

All vertical spacing between type elements follows a **4px base grid**:
- Overline to Title: 8px (`mt-2`)
- Title to Description: 8px (`mt-2`)
- Section to Section: 32px (`mt-8`)
- Card internal padding: 20px (`p-5`)
- Table row height: 48px
- Form field spacing: 16px (`space-y-4`)

### Iconography

| Aspect | Decision |
|---|---|
| **Library** | Lucide React (already installed, tree-shakeable, 1500+ icons) |
| **Style** | Outline only, `strokeWidth={1.75}` (slightly lighter than default 2 for elegance). Never filled. |
| **Size** | 16px for inline/nav, 20px for page headers, 24px for empty states |
| **Color** | Icons inherit text color. Never colored independently unless they represent status (severity dot, job status dot) |
| **Custom icons** | None. Lucide covers all pipeline concepts. Use `FileText` for evidence, `AlertTriangle` for problems, `Layers` for clusters, `Sparkles` for proposals, `ListChecks` for tasks, `Map` for roadmap. |

### Illustration Strategy

- **No illustrations.** This is a data tool. Empty states use the pipeline icon + 2-line text message + primary CTA button. Never clipart, never abstract blobs.
- **Data is the visual.** Severity distribution bars, pipeline status dots, and score breakdowns ARE the visual language.

---

## 3. Layout System & Design Architecture

### Grid System

| Property | Value |
|---|---|
| **Container max-width** | `1400px` (2xl breakpoint, already configured in Tailwind) |
| **Grid columns** | 12-column CSS Grid via Tailwind `grid-cols-12` |
| **Gutter** | 24px (`gap-6`) |
| **Page horizontal padding** | 32px (`px-8`) |
| **Sidebar width** | 256px fixed (`w-64`) |
| **Content area** | `calc(100vw - 256px)` or flex-1 |

### Spacing Scale (Tailwind native, referenced explicitly)

```
4px   = space-1    → Icon-to-text gap
8px   = space-2    → Tight grouping (badge padding, overline-to-title)
12px  = space-3    → Card internal element spacing
16px  = space-4    → Form field spacing, list item gap
20px  = space-5    → Card padding
24px  = space-6    → Section gap, grid gutter
32px  = space-8    → Major section break
48px  = space-12   → Page-level vertical rhythm
```

### Breakpoints

| Name | Min Width | Layout Behavior |
|---|---|---|
| **Desktop XL** | 1400px | Full layout, 3-column cluster grid |
| **Desktop** | 1024px | Full sidebar + content. 2-column cluster grid |
| **Tablet** | 768px | Sidebar collapses to icon-only (56px). Content fills. 2-column grid |
| **Mobile** | < 768px | Sidebar hidden (hamburger toggle). Single column. Tables become stacked cards. **Not a priority** — PMs work on desktop |

### Card Pattern

Cards are the atomic container for grouped information. All cards follow this structure:

```
┌─ Card ─────────────────────────────────────────────────┐
│  rounded-2xl                                            │
│  border border-border                                   │
│  bg-card (surface-1)                                    │
│  p-5                                                    │
│  shadow-none (default) → shadow-sm (hover, if actionable)│
│  transition-shadow duration-200                         │
│                                                         │
│  [Overline]           ← 11px, uppercase, tracking-wide  │
│  [Title]              ← 16px, font-semibold             │
│  [Description/Body]   ← 14px, text-muted-foreground     │
│  [Metadata row]       ← 12px, flex justify-between      │
│  [Action area]        ← buttons aligned right or full-w  │
└─────────────────────────────────────────────────────────┘
```

**Rules:**
- Cards are NEVER nested inside cards
- Interactive cards (cluster grid, evidence rows) gain `shadow-sm` on hover + `cursor-pointer`
- Static cards (stat cards, detail sections) have no hover effect
- Card border radius: `16px` (`rounded-2xl`)
- Card background: `bg-card` (one shade lighter than page background)

### Page Structure Template

Every page in the PM pipeline follows this structure:

```
┌─ Page ──────────────────────────────────────────────────────────┐
│                                                                  │
│  ┌─ PageHeader ──────────────────────────────────────────────┐  │
│  │  [Overline: "PM PIPELINE"]                                 │  │
│  │  [Page Title: "Evidence"]            [Action Button(s)]   │  │
│  │  [Description: 1-line page purpose]                       │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌─ Stats Bar (optional) ────────────────────────────────────┐  │
│  │  Severity distribution badges  |  Total count  |  Filter  │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌─ Filter Bar (optional) ───────────────────────────────────┐  │
│  │  [Persona ▼] [Severity ▼] [Tags ▼] [Search_________]     │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌─ Primary Content ─────────────────────────────────────────┐  │
│  │  Table  OR  Card Grid  OR  Detail View                     │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌─ Pagination (if table) ───────────────────────────────────┐  │
│  │  Page 1 of 3  [← Prev] [Next →]    Showing 1-20 of 47    │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 4. Page-Level Strategy

### 4.1 Dashboard (`/pm`)

**Primary user goal:** Understand pipeline status at a glance and know what to do next.

**Layout:**
```
┌─────────────────────────────────────────────────────────────────┐
│  PageHeader: "Dashboard" + "Pipeline overview and next steps"   │
├───────────┬───────────┬───────────┬───────────┬────────────────┤
│  Evidence │ Problems  │ Clusters  │ Proposals │  Roadmap       │
│  12 docs  │ 47 items  │ 8 groups  │ 5 specs   │  5 ranked      │
│  ✅ done  │ ✅ done   │ ✅ done   │ 3 draft   │  3 approved    │
├───────────┴───────────┴───────────┴───────────┴────────────────┤
│                                                                 │
│  Next Best Actions                                              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 🟡 2 proposals awaiting review → [Review Proposals →]  │   │
│  │ 🟡 3 approved proposals need tasks → [Generate Tasks →]│   │
│  │ 🔵 Upload more evidence → [Upload Evidence →]          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Recent Jobs                                Active Jobs         │
│  ┌────────────────────────────┐  ┌──────────────────────────┐  │
│  │ Extract problems  ✅ 12s  │  │ Clustering  ████░░ 60%   │  │
│  │ Embed problems    ✅ 8s   │  │ ETA: ~15s remaining      │  │
│  │ Upload evidence   ✅ 2s   │  └──────────────────────────┘  │
│  └────────────────────────────┘                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Key components:**
- `StatCard` × 5 — one per pipeline stage. Shows count + sub-status. Clickable → navigates to page.
- `NextActionsPanel` — Computed from pipeline state. If problems exist but no clusters, show "Run Clustering." If clusters exist but no proposals, show "Generate Proposals." Always one dominant action.
- `RecentJobsList` — Last 10 jobs with type, status badge, duration. Fetched from `/llm/calls`.
- `ActiveJobsPanel` — Active jobs with polling progress bar. Uses `useJobsStore`.

**Data requirements:**
```
GET /api/v1/evidence?page=1&per_page=1       → total count
GET /api/v1/problems/stats                     → total + severity breakdown
GET /api/v1/clusters?page=1&per_page=1        → total count
GET /api/v1/roadmap                            → proposal count, statuses
GET /api/v1/llm/calls                          → recent job history
```

**States:**
| State | Behavior |
|---|---|
| **Empty** (first visit) | Single large CTA card: "Start by uploading your first piece of evidence." Upload button centered. |
| **Loading** | 5 skeleton stat cards (shimmer animation). Jobs list shows 3 skeleton rows. |
| **Error** | Inline error banner per failed fetch. Individual stat cards show "—" with retry link. |
| **Populated** | Full layout as described. |

---

### 4.2 Evidence List (`/pm/evidence`)

**Primary user goal:** Browse uploaded evidence and track extraction status.

**Layout:**

```
┌──────────────────────────────────────────────────────────────────┐
│  PageHeader: "Evidence" + "Upload and manage source material"    │
│                                            [+ Upload Evidence]   │
├──────────────────────────────────────────────────────────────────┤
│  [Filter: Source type ▼] [Filter: Persona ▼] [Search _______]   │
├──────────────────────────────────────────────────────────────────┤
│  ┌───────────────────────────────┬──────┬────────┬─────┬──────┐ │
│  │ Title                         │ Type │Persona │Chunks│Status│ │
│  ├───────────────────────────────┼──────┼────────┼─────┼──────┤ │
│  │ Customer Interview - Acme     │ 🎙️  │ PM     │ 12  │ ✅   │ │
│  │ Support Ticket Batch Q4       │ 🎫  │ Admin  │ 34  │ ✅   │ │
│  │ Sales Call Notes - BigCo      │ 📞  │ PM     │ 8   │ ⏳   │ │
│  └───────────────────────────────┴──────┴────────┴─────┴──────┘ │
│  Showing 1-20 of 34          [← Prev] [1] [2] [Next →]          │
├──────────────────────────────────────────────────────────────────┤
│  Empty state: "No evidence uploaded yet. Start by uploading a    │
│  customer interview, support ticket, or sales note."             │
│  [Upload Evidence →]                                             │
└──────────────────────────────────────────────────────────────────┘
```

**Key interactions:**
- Click row → navigate to `/pm/evidence/[id]`
- "Upload Evidence" button → navigate to `/pm/evidence/upload`
- Source type filter: dropdown with `interview`, `support_ticket`, `sales_note`, `survey`, `other`
- Processing status column shows: ✅ extracted, ⏳ extracting (pulse animation), ❌ failed (with retry)
- Sortable columns: Title (alpha), Type, Created (date), Chunks (numeric)

**Data requirements:**
```
GET /api/v1/evidence?page={n}&per_page=20&source_type={filter}&persona={filter}
```

**Component:** `EvidenceTable` using TanStack Table with column definitions, sorting state, and pagination.

---

### 4.3 Evidence Upload (`/pm/evidence/upload`)

**Primary user goal:** Get evidence into the system in under 30 seconds.

**Layout:**
```
┌──────────────────────────────────────────────────────────────────┐
│  PageHeader: "Upload Evidence"                   [← Back]        │
├──────────────────────────────────────────────────────────────────┤
│  ┌─ Dropzone ────────────────────────────────────────────────┐  │
│  │                                                            │  │
│  │   📄  Drop a file here, or click to browse                │  │
│  │       Accepts: .txt, .pdf, .csv, .md, .docx               │  │
│  │                                                            │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ── OR paste text directly ──                                    │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ [Multi-line text area, 8 rows, monospace hint]             │  │
│  │                                                            │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  Title *        [Customer Interview - Acme Corp              ]   │
│  Source Type *  [Interview ▼]                                    │
│  Persona        [Product Manager                             ]   │
│  Segment        [Enterprise                                  ]   │
│  Date           [2026-01-15                                  ]   │
│                                                                  │
│                              [Cancel]  [Upload & Extract →]      │
└──────────────────────────────────────────────────────────────────┘
```

**Key interactions:**
- Drag-and-drop zone via `react-dropzone`. Highlights with blue dashed border on drag enter.
- File upload reads content client-side (text files) or sends as multipart (PDF/DOCX — future backend support).
- Form validation via `react-hook-form` + `zod`: title required, source_type required, raw_text required (from file or paste).
- On submit: calls `createEvidence()` → calls `extractProblems()` → shows toast "Processing started" → redirects to evidence list.
- If extraction is still running when user navigates away, `useJobsStore` tracks it. PipelineIndicator and RecentJobs show status.

**States:**
| State | Behavior |
|---|---|
| **Default** | Empty form, dropzone ready |
| **File dropped** | File name shown below dropzone, text area auto-populated (if text file) |
| **Submitting** | Button shows spinner + "Uploading…" → "Extracting…" (two-phase) |
| **Success** | Toast: "Evidence uploaded. Problem extraction started." Redirect. |
| **Error** | Inline error below form. Button re-enabled. |

---

### 4.4 Evidence Detail (`/pm/evidence/[id]`)

**Primary user goal:** See the raw evidence text, its chunks, and problems extracted from it.

**Layout:**
```
┌──────────────────────────────────────────────────────────────────┐
│  ← Back to Evidence                                              │
│  PageHeader: "{Evidence Title}"                                  │
│  Meta: Interview · PM · Enterprise · Jan 15, 2026 · 12 chunks   │
├──────────────────────────────────────────────────────────────────┤
│  [Tab: Raw Text] [Tab: Extracted Problems] [Tab: Processing]     │
│                                                                  │
│  ── Tab 1: Raw Text ──                                           │
│  Full text content with chunk boundaries highlighted as          │
│  alternating subtle background bands (surface-1 / surface-2).   │
│  Each chunk shows its index as a small left-margin badge.        │
│                                                                  │
│  ── Tab 2: Extracted Problems ──                                 │
│  Table of problems extracted from THIS evidence only.            │
│  Columns: Problem Statement, Severity, Quote Preview, Tags      │
│  Click row → navigate to /pm/problems/[id]                       │
│                                                                  │
│  ── Tab 3: Processing History ──                                 │
│  List of jobs run against this evidence:                         │
│  Job type, status, started_at, duration, token cost              │
│  [Re-extract Problems] button                                    │
└──────────────────────────────────────────────────────────────────┘
```

**Data requirements:**
```
GET /api/v1/evidence/{id}                      → full detail with chunks
GET /api/v1/problems?evidence_id={id}          → problems from this evidence
```

---

### 4.5 Problems Table (`/pm/problems`)

**Primary user goal:** Find, filter, and explore all extracted problem mentions across all evidence.

**Layout:**
```
┌──────────────────────────────────────────────────────────────────────┐
│  PageHeader: "Problems" + "All extracted problem mentions"           │
│  47 total  |  🔴 5 Critical  🟠 18 High  🟡 16 Medium  🟢 8 Low    │
├──────────────────────────────────────────────────────────────────────┤
│  [Persona ▼] [Severity ▼] [Tags ▼] [Source ▼] [Search___________]  │
├──────────────────────────────────────────────────────────────────────┤
│  ┌───┬──────────────────────────────┬─────────┬────────┬──────────┐ │
│  │   │ Problem                      │Severity │Persona │ Tags     │ │
│  ├───┼──────────────────────────────┼─────────┼────────┼──────────┤ │
│  │ ▸ │ Permissions config too complex│ 🔴 CRIT │ PM     │ perms    │ │
│  │ ▾ │ Onboarding takes >2 hours    │ 🔴 HIGH │ Admin  │ onboard  │ │
│  │   ├──────────────────────────────────────────────────────────────│ │
│  │   │ Expanded: Full quote + source evidence link + [View Similar]│ │
│  │   ├──────────────────────────────────────────────────────────────│ │
│  │ ▸ │ Reports load too slowly      │ 🟡 MED  │ Analyst│ perf     │ │
│  └───┴──────────────────────────────┴─────────┴────────┴──────────┘ │
│  Page 1 of 3  [← Prev] [Next →]                                     │
└──────────────────────────────────────────────────────────────────────┘
```

**Key interactions:**
- **Filter bar:** Four dropdown filters + free text search. These update URL query params and refetch server-side data. Filter state persisted in `useFilterStore` (Zustand) and synced with URL.
- **Expandable rows:** Click row chevron to expand. Shows full `quote_text` in a `QuoteBlock` component, plus source evidence title (linked to `/pm/evidence/[id]`), plus "View Similar" button.
- **"View Similar" button:** Opens a slide-over panel from the right (Sheet component). Panel fetches `/problems/similar?text={problem_statement}` and shows ranked list with similarity scores (0-1, displayed as percentage). Each result is clickable.
- **Click problem statement link:** Navigates to `/pm/problems/[id]`.
- **Severity stats bar:** Shows total count + distribution badges at the top. Data from `/problems/stats`. Badges are clickable (act as quick severity filter).
- **Pagination:** Server-side via `page` and `per_page` params.

**Component:** `ProblemsDataTable` using TanStack Table with:
- Column definitions: `problem_statement`, `severity`, `persona`, `tags`, `created_at`
- Expandable row rendering
- Sort state (client-side within page, server-side across pages)
- `SeverityBadge` sub-component with color mapping

**Data requirements:**
```
GET /api/v1/problems?page=1&per_page=20&severity={}&persona={}&search={}
GET /api/v1/problems/stats
GET /api/v1/problems/similar?text={}&limit=10  (on "View Similar" click)
```

**States:**
| State | Behavior |
|---|---|
| **Empty** | "No problems extracted yet. Upload evidence to get started." + [Upload Evidence →] |
| **Loading** | Table skeleton: 8 shimmer rows with correct column widths |
| **Filtered empty** | "No problems match your filters." + [Clear Filters] |
| **Error** | Inline error banner with retry |

---

### 4.6 Problem Detail (`/pm/problems/[id]`)

**Primary user goal:** Deep dive into a single problem — full quote, source, and similar problems.

**Layout:**
```
┌──────────────────────────────────────────────────────────────────┐
│  ← Back to Problems                                              │
│  PageHeader: Problem statement text                              │
│  [Severity Badge]  ·  Persona  ·  Tags                          │
├──────────────────────────────────────────────────────────────────┤
│  ┌─ QuoteBlock ──────────────────────────────────────────────┐  │
│  │  "Full verbatim quote from customer..."                    │  │
│  │  — Source: Customer Interview - Acme Corp  ·  Jan 2026     │  │
│  │  [View Source Evidence →]                                  │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  Similar Problems (8 found)                                      │
│  ┌──────────────────────────────────────────────────┬──────┐    │
│  │ Problem Statement                                │ Score│    │
│  ├──────────────────────────────────────────────────┼──────┤    │
│  │ Onboarding wizard doesn't save progress          │ 94%  │    │
│  │ New users confused by permissions setup           │ 87%  │    │
│  │ First project creation requires IT help           │ 81%  │    │
│  └──────────────────────────────────────────────────┴──────┘    │
└──────────────────────────────────────────────────────────────────┘
```

---

### 4.7 Clusters Grid (`/pm/clusters`)

**Primary user goal:** See grouped pain themes, sorted by magnitude, and trigger proposal generation.

**Layout:**
```
┌──────────────────────────────────────────────────────────────────┐
│  PageHeader: "Clusters" + "Grouped pain themes from evidence"    │
│                          [Run Clustering] (if unclustered exist)  │
├──────────────────────────────────────────────────────────────────┤
│  ┌─ ClusterCard ────────────┐  ┌─ ClusterCard ────────────┐     │
│  │ Onboarding confusion     │  │ Report performance       │     │
│  │ 23 mentions              │  │ 15 mentions              │     │
│  │ ████████░░ Sev: 3.2 avg  │  │ █████░░░░░ Sev: 2.4 avg │     │
│  │                          │  │                          │     │
│  │ "our team gave up on     │  │ "reports take 45 seconds │     │
│  │  onboarding after day 2" │  │  to load every time"     │     │
│  │  — Support #4521         │  │  — Analyst Interview     │     │
│  │                          │  │                          │     │
│  │ [View Details]           │  │ [View Details]           │     │
│  └──────────────────────────┘  └──────────────────────────┘     │
│                                                                  │
│  ┌─ ClusterCard ────────────┐  ┌─ ClusterCard ────────────┐     │
│  │ ...                      │  │ ...                      │     │
│  └──────────────────────────┘  └──────────────────────────┘     │
└──────────────────────────────────────────────────────────────────┘
```

**Cluster card anatomy:**
```tsx
<ClusterCard>
  <CardTitle>{cluster.label}</CardTitle>
  <MentionCount>{cluster.mention_count} mentions</MentionCount>
  <SeverityBar distribution={cluster.severity_distribution} />
  <TopQuote text={...} source={...} />
  <CardActions>
    <Link href={`/pm/clusters/${cluster.id}`}>View Details</Link>
  </CardActions>
</ClusterCard>
```

**Key interactions:**
- Cards sorted by `mention_count` descending (biggest problem = first card)
- "Run Clustering" button appears when there are problems without clusters. Triggers `POST /clusters/run`, shows job progress via toast + polling.
- Grid: 3 columns on desktop XL, 2 on desktop, 1 on tablet/mobile
- Hovering a card → subtle `shadow-sm` lift

**Data requirements:**
```
GET /api/v1/clusters?page=1&per_page=30
POST /api/v1/clusters/run?threshold=0.75    (on "Run Clustering" click)
```

**States:**
| State | Behavior |
|---|---|
| **Empty (no problems)** | "No clusters yet. Extract problems first, then cluster." + [View Problems →] |
| **Unclustered** | Banner: "23 unclustered problems." + [Run Clustering] button |
| **Clustering** | Job progress inline: "Clustering in progress…" + progress bar |
| **Populated** | Card grid sorted by mention count |

---

### 4.8 Cluster Detail (`/pm/clusters/[id]`)

**Primary user goal:** Understand a pain cluster deeply — see all quotes, severity distribution, and generate a feature proposal.

**Layout:**
```
┌──────────────────────────────────────────────────────────────────┐
│  ← Back to Clusters                                              │
│  PageHeader: "{Cluster Label}"                                   │
│  23 mentions · Avg severity: 3.2                                 │
├──────────────────────────────────────────────────────────────────┤
│  ┌─ Summary Card ────────────────────────────────────────────┐  │
│  │  Multiple users across enterprise and mid-market segments  │  │
│  │  report difficulty completing initial setup. Key friction   │  │
│  │  points: permissions config, project creation workflows.   │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌─ Severity Distribution ──────┐  ┌─ By Persona ───────────┐  │
│  │  Critical ████████     4     │  │  PM: 12                 │  │
│  │  High     ██████████████ 11  │  │  Admin: 7               │  │
│  │  Medium   ████████████  7    │  │  Developer: 3           │  │
│  │  Low      ██           1     │  │  User: 1                │  │
│  └──────────────────────────────┘  └──────────────────────────┘  │
│                                                                  │
│  Top Quotes                                                      │
│  ┌─ QuoteBlock ──────────────────────────────────────────────┐  │
│  │  "our team gave up on onboarding after day two"            │  │
│  │  — Support Ticket #4521  ·  🔴 Critical                   │  │
│  ├────────────────────────────────────────────────────────────┤  │
│  │  "I spent 3 hours trying to set up my first project"      │  │
│  │  — Acme Corp PM Interview  ·  🟠 High                     │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  All Members (23)                                                │
│  [Full problems table with severity, persona, quote preview]     │
│                                                                  │
│  ┌─ Proposal Section ───────────────────────────────────────┐   │
│  │  Status: Not generated                                    │   │
│  │  [Generate Feature Proposal →]                            │   │
│  └───────────────────────────────────────────────────────────┘   │
│                                                                  │
│  OR (if proposal exists):                                        │
│  ┌─ Linked Proposal ────────────────────────────────────────┐   │
│  │  "Guided Onboarding Wizard"  ·  Status: Draft            │   │
│  │  [View Proposal →]                                        │   │
│  └───────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

**Key interactions:**
- Severity distribution rendered as horizontal bar chart (`SeverityChart` component using Recharts `BarChart`, horizontal layout, severity colors)
- Persona breakdown as simple list with counts
- Top quotes: 3–5 highest severity quotes rendered as `QuoteBlock` components with `onClick` → navigate to problem detail
- "Generate Feature Proposal" triggers `POST /api/v1/jobs/generate_proposal` → polls job → on completion, invalidates cluster detail query and shows linked proposal
- Members table: same `ProblemsDataTable` component, filtered to this cluster's members

**Data requirements:**
```
GET /api/v1/clusters/{id}    → detail with members, proposals, severity stats
POST /api/v1/jobs/generate_proposal  { cluster_id }   (on button click)
```

---

### 4.9 Proposals List (`/pm/proposals`)

**Primary user goal:** Browse all generated feature proposals, filter by status, and take action.

**Layout:**
```
┌──────────────────────────────────────────────────────────────────┐
│  PageHeader: "Proposals" + "AI-generated feature specifications"  │
│  [Filter: All ▼ | Draft | Approved | Rejected]                   │
├──────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────…──────────┬───────┬──────────┐ │
│  │ Proposal                                │ Scope │ Status   │ │
│  ├──────────────────────────────…──────────┼───────┼──────────┤ │
│  │ Guided Onboarding Wizard               │ M     │ 🔵 Draft │ │
│  │ "Step-by-step setup flow..."           │       │          │ │
│  │ Cluster: Onboarding confusion · 23 men.│       │          │ │
│  ├──────────────────────────────…──────────┼───────┼──────────┤ │
│  │ Real-time Report Engine                │ L     │ ✅ Appr. │ │
│  │ "Sub-second report generation..."      │       │          │ │
│  │ Cluster: Report performance · 15 men.  │       │          │ │
│  └──────────────────────────────…──────────┴───────┴──────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

**Key interactions:**
- Click row → navigate to `/pm/proposals/[id]`
- Status filter tabs: All, Draft, Approved, Rejected
- Each row shows: proposal title, one-liner, source cluster label + mention count, scope badge, status badge
- Status badges: Draft (blue), Approved (green), Rejected (gray)
- Scope badges: S/M/L/XL with effort color coding

**Data requirements:**
```
GET /api/v1/roadmap                           → proposals with scored ranking
GET /api/v1/feature_proposals?status={filter}  → filtered proposal list (future)
```

---

### 4.10 Proposal Detail (`/pm/proposals/[id]`)

**Primary user goal:** Review a full feature spec, verify citations, and approve/reject.

**Layout:**
```
┌──────────────────────────────────────────────────────────────────────┐
│  ← Back to Proposals                                                 │
│  PageHeader: "{Feature Name}"                  Status: [DRAFT ▼]     │
│  "{One-liner description}"                                           │
│  Scope: M (1-3 weeks)  |  Cluster: Onboarding confusion             │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─ User Story ─────────────────────────────────────────────────┐   │
│  │  As a new admin, I want a guided setup wizard so that I can   │   │
│  │  configure permissions and create my first project in under   │   │
│  │  30 minutes.                                                  │   │
│  │                                                    [✏️ Edit] │   │
│  └───────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌─ Rationale (with citations) ─────────────────────────────────┐   │
│  │  23 customers report onboarding friction as #1 pain point.    │   │
│  │                                                               │   │
│  │  Users abandon setup entirely: [1] "our team gave up on       │   │
│  │  onboarding after day two" (Support Ticket #4521).            │   │
│  │                                                               │   │
│  │  [1] = clickable → opens source problem in side panel         │   │
│  └───────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌─ Success Metrics ────────────────────────────────────────────┐   │
│  │  Metric                    │ Target  │ Reasoning              │   │
│  │  Onboarding completion     │ >80%    │ Currently ~40%         │   │
│  │  Time to first project     │ <30min  │ Currently 2-3 hours    │   │
│  └───────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌─ Risks ──────────────────────────────────────────────────────┐   │
│  │  Risk                      │ Severity │ Mitigation            │   │
│  │  Power users feel limited  │ Medium   │ "Skip wizard" option  │   │
│  └───────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌─ Actions ────────────────────────────────────────────────────┐   │
│  │  [✅ Approve]  [❌ Reject]  [🔄 Regenerate]  [🔨 Gen Tasks]│   │
│  └───────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
```

**Key interactions:**
- **Inline editing:** Click [✏️ Edit] on any section → fields become editable textareas. Save/Cancel buttons appear. Calls `PATCH /feature_proposals/{id}`.
- **Citation links:** Superscript `[1]` rendered as clickable links. On click, opens a `Sheet` (right panel) showing the source problem's full quote, evidence title, and severity. Click "View full" → navigates to problem detail.
- **Approve/Reject:** Confirmation dialog. Updates proposal `status`. Approved proposals appear in roadmap. Rejected proposals are grayed out.
- **Regenerate:** Confirmation dialog ("This will replace the current spec"). Triggers LLM job. Shows progress. On completion, refreshes proposal detail.
- **Generate Tasks:** Triggers `POST /jobs/generate_tasks`. On completion, navigates to `/pm/proposals/[id]/tasks`.

**Data requirements:**
```
GET /api/v1/feature_proposals/{id}              → full proposal with citations
PATCH /api/v1/feature_proposals/{id}            → update fields
POST /api/v1/feature_proposals/{id}/approve     → status change
POST /api/v1/feature_proposals/{id}/reject      → status change
POST /api/v1/feature_proposals/{id}/regenerate  → triggers job
POST /api/v1/jobs/generate_tasks { proposal_id }→ triggers job
```

**States:**
| State | Behavior |
|---|---|
| **Draft** | All sections editable. All action buttons enabled. |
| **Approved** | Sections read-only (no edit button). Only "Reject" and "Generate Tasks" available. Green status badge. |
| **Rejected** | Sections read-only. Only "Approve" available. Gray status badge. |
| **Generating** | Skeleton placeholder for content. Progress bar. |

---

### 4.11 Task Tree (`/pm/proposals/[id]/tasks`)

**Primary user goal:** See the implementation breakdown for a proposal, ready to export to engineering tools.

**Layout:**
```
┌──────────────────────────────────────────────────────────────────┐
│  ← Back to Proposal: "Guided Onboarding Wizard"                 │
│  PageHeader: "Implementation Plan"                               │
│  18 tasks  ·  Generated from: Guided Onboarding Wizard          │
├──────────────────────────────────────────────────────────────────┤
│  [Backend (6)] [Frontend (5)] [Data (3)] [QA (4)]  ← Tabs       │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ▾ Create onboarding state machine API              [M]          │
│    Depends on: Create onboarding_progress table                  │
│    Description: Build REST endpoints for managing onboarding...  │
│                                                                  │
│    Acceptance Criteria:                                           │
│    ☐ POST /onboarding/start creates progress record             │
│    ☐ POST /onboarding/next advances to next step               │
│    ☐ GET /onboarding/status returns completion state            │
│                                                                  │
│  ▸ Create permissions template endpoint             [S]          │
│  ▸ Add onboarding progress tracking                 [M]          │
│  ▸ Create project scaffolding endpoint              [M]          │
│                                                                  │
├──────────────────────────────────────────────────────────────────┤
│  [📋 Copy as Markdown]  [📥 Download JSON]  [🔄 Regenerate]     │
└──────────────────────────────────────────────────────────────────┘
```

**Key interactions:**
- **Category tabs:** Backend, Frontend, Data, QA. Each tab shows its task list. Tab label includes count.
- **Collapsible nodes:** Click the ▸/▾ chevron to expand/collapse. Expanded shows: description, acceptance criteria, dependencies.
- **Effort badges:** XS (slate), S (blue), M (amber), L (orange), XL (red). Small pill on right side of each task row.
- **Dependencies:** Shown as a muted text line linking to the dependency task name. Click → scrolls to and highlights the dependency task.
- **Export:**
  - "Copy as Markdown" → generates structured markdown with headers per category, nested bullets per task, acceptance criteria as checkbox list. Copies to clipboard. Toast: "Copied to clipboard."
  - "Download JSON" → downloads structured JSON file with all task data. Browser download.
- **Regenerate:** Confirmation → triggers new LLM job → replaces tree.

**Data requirements:**
```
GET /api/v1/feature_proposals/{id}/tasks    → full task tree
```

**Component:** `TaskTree` with `TaskNode` recursive renderer, `TaskCategoryTabs` (Radix Tabs), `AcceptanceCriteria` checklist, `TaskExport` button group.

---

### 4.12 Roadmap (`/pm/roadmap`)

**Primary user goal:** See all proposals ranked by priority score, with transparent scoring and adjustable weights.

**Layout:**
```
┌──────────────────────────────────────────────────────────────────────┐
│  PageHeader: "Roadmap" + "Prioritized feature ranking"               │
│  12 proposals  ·  Last clustered: Feb 12, 2026                      │
│  [Persona ▼] [Segment ▼] [Status ▼]                                │
├──────────────────────────────────────────────────────────────────────┤
│  ┌──┬────────────────────────────┬──────┬───────┬───────┬────────┐  │
│  │# │ Feature                    │Scope │Status │ Score │Breakdown│  │
│  ├──┼────────────────────────────┼──────┼───────┼───────┼────────┤  │
│  │1 │ Guided Onboarding Wizard   │ M    │✅ Appr│ 42.5  │ ▸      │  │
│  │  │ Cluster: Onboarding · 23m │      │       │       │        │  │
│  ├──┼────────────────────────────┼──────┼───────┼───────┼────────┤  │
│  │2 │ Real-time Report Engine    │ L    │✅ Appr│ 38.1  │ ▾      │  │
│  │  │ Cluster: Performance · 15m│      │       │       │        │  │
│  │  │                                                             │  │
│  │  │  Score Breakdown:                                           │  │
│  │  │  Formula: (frequency × severity × weight) / effort          │  │
│  │  │  Frequency:  28.0  ████████░░░░                             │  │
│  │  │  Severity:   2.8   ███████░░░░░                             │  │
│  │  │  Weight:     1.0   [━━━━━━━━●━━] ← adjustable slider       │  │
│  │  │  Effort:     8     (L scope)                                │  │
│  │  │  ─────────────────────────────                              │  │
│  │  │  Final:      38.1  = (28 × 2.8 × 1.0) / 8                  │  │
│  │  │                                                             │  │
│  ├──┼────────────────────────────┼──────┼───────┼───────┼────────┤  │
│  │3 │ Granular Permissions v2    │ L    │🔵 Drf │ 31.7  │ ▸      │  │
│  └──┴────────────────────────────┴──────┴───────┴───────┴────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

**Key interactions:**
- **Expandable score breakdown:** Click ▸ on any row → inline expansion shows full formula, each variable with value + visual bar + explanation.
- **Strategic weight slider:** Radix Slider component (0.1 to 3.0, step 0.1). On change, recalculates score client-side for instant feedback. On release, persists via `PATCH /roadmap/{proposalId}/weight`. Re-sorts the list automatically.
- **Filters:** Persona, Segment, Status dropdowns. Filter the ranked list.
- **Click proposal name:** Navigates to `/pm/proposals/[id]`.
- **Click cluster name:** Navigates to `/pm/clusters/[id]`.

**Data requirements:**
```
GET /api/v1/roadmap                                   → ranked proposals
PATCH /api/v1/roadmap/{proposalId}/weight { weight }  → update strategic weight
```

---

### 4.13 Settings (`/pm/settings`)

**Primary user goal:** Configure API keys and prompt settings.

**Layout:**
```
┌──────────────────────────────────────────────────────────────────┐
│  PageHeader: "Settings" + "API configuration and prompt tuning"  │
├──────────────────────────────────────────────────────────────────┤
│  ┌─ API Configuration ──────────────────────────────────────┐   │
│  │  OpenAI API Key:    [●●●●●●●●●●●●sk-proj-xxxx]  [Edit] │   │
│  │  Model:             [gpt-4o-mini ▼]                      │   │
│  │  Backend URL:       http://localhost:8000   [Test ✅]    │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─ Extraction Settings ────────────────────────────────────┐   │
│  │  Chunk size:        [1000] tokens                         │   │
│  │  Chunk overlap:     [200] tokens                          │   │
│  │  Extraction model:  [gpt-4o-mini ▼]                       │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─ Clustering Settings ────────────────────────────────────┐   │
│  │  Similarity threshold: [0.75] [━━━━━━━━●━━]              │   │
│  │  Min cluster size:     [3]                                │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│                                              [Save Settings]     │
└──────────────────────────────────────────────────────────────────┘
```

---

### 4.14 Usage / Cost Tracking (`/pm/usage`)

**Primary user goal:** Understand LLM costs and token consumption.

**Layout:**
```
┌──────────────────────────────────────────────────────────────────┐
│  PageHeader: "Usage" + "LLM cost tracking and job history"       │
├────────────┬────────────┬────────────┬──────────────────────────┤
│ Total Calls│ Input Tok  │ Output Tok │ Total Cost               │
│ 156        │ 245,800    │ 48,200     │ $3.42                    │
├────────────┴────────────┴────────────┴──────────────────────────┤
│                                                                  │
│  ┌─ Cost by Model ──────────────────────────────────────────┐   │
│  │  Model          │ Calls │ In Tokens │ Out Tokens │ Cost   │   │
│  │  gpt-4o-mini    │  142  │ 231,000   │  44,500    │ $2.18  │   │
│  │  gpt-4o         │   14  │  14,800   │   3,700    │ $1.24  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─ Job History ────────────────────────────────────────────┐   │
│  │  Job Type         │ Status │ Started     │ Duration │ Cost │   │
│  │  extract_problems  │ ✅     │ 2 min ago   │ 12s     │$0.04│   │
│  │  embed_problems    │ ✅     │ 5 min ago   │ 8s      │$0.02│   │
│  │  cluster           │ ⏳     │ just now    │ —       │ —   │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

**Data requirements:**
```
GET /api/v1/llm/costs     → aggregated cost stats
GET /api/v1/llm/calls     → individual job records
```

---

## 5. Component Architecture

### Directory Structure

```
components/
├── pm/
│   ├── layout/
│   │   ├── PMSidebar.tsx              # Sidebar navigation
│   │   └── PMLayout.tsx               # Layout wrapper (pipeline indicator + content area)
│   │
│   ├── pipeline/
│   │   ├── PipelineIndicator.tsx      # Horizontal pipeline status bar
│   │   └── PipelineStep.tsx           # Single step: dot + label + count
│   │
│   ├── evidence/
│   │   ├── EvidenceTable.tsx          # TanStack Table for evidence list
│   │   ├── EvidenceUploadForm.tsx     # Upload form with dropzone + metadata
│   │   ├── EvidenceDetailTabs.tsx     # Tabs: Raw Text / Problems / Processing
│   │   └── ChunkViewer.tsx            # Text with chunk boundary highlighting
│   │
│   ├── problems/
│   │   ├── ProblemsDataTable.tsx      # TanStack Table with expandable rows
│   │   ├── ProblemFilters.tsx         # Filter bar with dropdowns + search
│   │   ├── SimilarProblemsPanel.tsx   # Sheet (slide-over) for similar problems
│   │   └── ProblemExpandedRow.tsx     # Expanded row content: quote + actions
│   │
│   ├── clusters/
│   │   ├── ClusterGrid.tsx            # Responsive card grid
│   │   ├── ClusterCard.tsx            # Summary card with severity bar + quote
│   │   ├── ClusterDetailView.tsx      # Full cluster detail layout
│   │   └── SeverityChart.tsx          # Horizontal bar chart (Recharts)
│   │
│   ├── proposals/
│   │   ├── ProposalList.tsx           # List with status badges
│   │   ├── ProposalDetailView.tsx     # Full proposal spec with citations
│   │   ├── ProposalEditor.tsx         # Inline editing for proposal fields
│   │   ├── CitationLink.tsx           # Clickable [1] citation → side panel
│   │   ├── MetricsTable.tsx           # Success metrics table
│   │   ├── RisksTable.tsx             # Risks + mitigations table
│   │   └── ProposalActions.tsx        # Approve/Reject/Regenerate/GenTasks
│   │
│   ├── tasks/
│   │   ├── TaskTree.tsx               # Full tree with category tabs
│   │   ├── TaskNode.tsx               # Collapsible single task node
│   │   ├── TaskCategoryTabs.tsx       # Backend/Frontend/Data/QA tabs
│   │   ├── AcceptanceCriteria.tsx     # Checkbox-style criteria list
│   │   ├── EffortBadge.tsx            # XS/S/M/L/XL colored pill
│   │   └── TaskExport.tsx             # Copy markdown + download JSON
│   │
│   ├── roadmap/
│   │   ├── RoadmapTable.tsx           # Ranked table with expandable scores
│   │   ├── ScoreBreakdown.tsx         # Formula visualization
│   │   ├── WeightSlider.tsx           # Strategic weight adjuster (Radix Slider)
│   │   └── RoadmapFilters.tsx         # Persona/Segment/Status filters
│   │
│   └── shared/
│       ├── PageHeader.tsx             # [EXISTS] Page title + description + actions
│       ├── QuoteBlock.tsx             # [EXISTS] Styled quote with severity color
│       ├── SeverityBadge.tsx          # Colored pill: CRITICAL / HIGH / MED / LOW
│       ├── StatusBadge.tsx            # Draft (blue) / Approved (green) / Rejected (gray)
│       ├── ScopeBadge.tsx             # S / M / L / XL effort pill
│       ├── EmptyState.tsx             # Icon + message + CTA button
│       ├── JobProgress.tsx            # Inline progress bar with polling
│       ├── DataTable.tsx              # TanStack Table wrapper with pagination
│       ├── FilterDropdown.tsx         # Reusable filter dropdown
│       ├── ConfirmDialog.tsx          # Confirmation modal for destructive actions
│       └── SkeletonTable.tsx          # Shimmer skeleton for table loading
│
└── ui/                                # shadcn/ui primitives (unchanged)
    ├── button.tsx
    ├── card.tsx
    ├── dialog.tsx
    ├── dropdown-menu.tsx
    ├── input.tsx
    ├── progress.tsx
    ├── scroll-area.tsx
    ├── separator.tsx
    ├── tabs.tsx
    ├── textarea.tsx
    ├── tooltip.tsx
    ├── badge.tsx
    ├── avatar.tsx
    ├── sheet.tsx                       # NEW — side panels
    ├── slider.tsx                     # NEW — Radix Slider for weight
    ├── select.tsx                     # NEW — Radix Select for filters
    ├── skeleton.tsx                   # NEW — skeleton loading
    └── toast.tsx / toaster.tsx        # NEW — sonner integration
```

### Key Component Specifications

#### `SeverityBadge`
```tsx
// components/pm/shared/SeverityBadge.tsx
//
// Props: { severity: "critical" | "high" | "medium" | "low" }
//
// Renders a small pill with text and background color:
//   critical → bg-red-100 text-red-700 border-red-200      "CRITICAL"
//   high     → bg-orange-100 text-orange-700 border-orange-200  "HIGH"
//   medium   → bg-amber-100 text-amber-700 border-amber-200    "MEDIUM"
//   low      → bg-green-100 text-green-700 border-green-200    "LOW"
//
// Size: text-[11px] font-semibold px-2 py-0.5 rounded-full
// Always uppercase.
```

#### `StatusBadge`
```tsx
// components/pm/shared/StatusBadge.tsx
//
// Props: { status: "draft" | "approved" | "rejected" | "running" | "failed" }
//
// Renders:
//   draft    → bg-blue-100 text-blue-700       "Draft"
//   approved → bg-green-100 text-green-700     "Approved"
//   rejected → bg-gray-100 text-gray-500       "Rejected"
//   running  → bg-amber-100 text-amber-700     "Running"   + pulse animation
//   failed   → bg-red-100 text-red-700         "Failed"
//
// Size: same as SeverityBadge
```

#### `EmptyState`
```tsx
// components/pm/shared/EmptyState.tsx
//
// Props: {
//   icon: LucideIcon;
//   title: string;              // "No problems extracted yet"
//   description: string;        // "Upload evidence to get started."
//   actionLabel?: string;       // "Upload Evidence"
//   actionHref?: string;        // "/pm/evidence/upload"
// }
//
// Layout: centered icon (24px, muted), title (16px), description (14px, muted),
//         optional primary button below. Vertical stack with 12px gaps.
```

#### `JobProgress`
```tsx
// components/pm/shared/JobProgress.tsx
//
// Props: {
//   jobId: string;
//   label: string;              // "Extracting problems..."
//   onComplete?: () => void;    // Callback when job finishes
//   onError?: (error: string) => void;
// }
//
// Behavior:
// 1. Polls GET /api/v1/jobs/{jobId}/status every 2 seconds
// 2. Shows: label + animated progress bar (indeterminate or percentage-based)
// 3. On status === "completed": calls onComplete, shows success toast
// 4. On status === "failed": calls onError, shows error toast with retry option
// 5. Uses useJobsStore to track globally (PipelineIndicator can reflect it)
```

#### `DataTable` (TanStack Table Wrapper)
```tsx
// components/pm/shared/DataTable.tsx
//
// Props: {
//   columns: ColumnDef[];
//   data: T[];
//   pagination?: { page: number; totalPages: number; onPageChange: (p: number) => void };
//   sorting?: boolean;          // Enable sortable headers
//   expandable?: boolean;       // Enable row expansion
//   renderExpanded?: (row: T) => ReactNode;
//   emptyState?: ReactNode;     // Custom empty state
//   loading?: boolean;          // Show skeleton
//   skeletonRows?: number;      // Default: 8
// }
//
// Features:
// - TanStack Table core with flexRender
// - Sortable column headers (arrow indicators)
// - Expandable rows (chevron on leftmost column)
// - Pagination controls (Previous/Next + page numbers)
// - Skeleton loading state (shimmer rows matching column count)
// - Responsive: full table on desktop, stacked cards on mobile
```

---

## 6. Interaction & Animation System

### Animation Principles

1. **Purpose over polish** — Every animation communicates state change (appears, disappears, transitions, loads). No animation for decoration.
2. **Duration: 150–300ms** — Fast enough to feel instant, slow enough to be perceived. Never exceed 500ms for any UI transition.
3. **Easing: ease-out** — Elements settle naturally. Use `cubic-bezier(0.16, 1, 0.3, 1)` for entrances (Vercel-style spring). Use `ease-in` only for exits.
4. **Reduce motion** — Respect `prefers-reduced-motion`. All animations wrapped in a `motion.div` that checks this preference. Reduced motion = instant state changes, no transitions.

### Specific Animations

| Element | Animation | Duration | Implementation |
|---|---|---|---|
| **Page transitions** | Fade + slide up 8px on enter | 200ms | Framer Motion `initial={{ opacity: 0, y: 8 }}` on page container |
| **Card hover** | Subtle shadow lift | 200ms | CSS `transition-shadow duration-200 hover:shadow-md` |
| **Button press** | Scale down 98% | 100ms | CSS `active:scale-[0.98]` |
| **Button hover** | Background color shift | 150ms | CSS `transition-colors duration-150` |
| **Toast notification** | Slide in from bottom-right | 300ms | Sonner default animation |
| **Sheet (side panel)** | Slide in from right | 250ms | Radix Sheet default |
| **Dialog** | Fade + scale from 95% | 200ms | Radix Dialog default |
| **Row expansion** | Height animate + fade | 200ms | `accordion-down` keyframe (already defined) |
| **Pipeline indicator** | Staggered fade-in on mount | 50ms stagger | Framer Motion `delay: index * 0.05` (already implemented) |
| **Skeleton shimmer** | Horizontal gradient sweep | 2s loop | CSS `shimmer` keyframe (already defined) |
| **Job progress bar** | Width transition | 300ms | CSS `transition-all duration-300` on width |
| **Status dot (running)** | Pulse opacity | 2s loop | CSS `animate-pulse` on running status |
| **Filter dropdown** | Fade + scale from top | 150ms | Radix Popover default |
| **Score breakdown expand** | Height + opacity | 200ms | Framer Motion `AnimatePresence` |
| **Weight slider** | Thumb drag → score recalculate | Instant + 200ms | Value changes instantly, score number fades to new value |

### Micro-Interactions

| Trigger | Feedback |
|---|---|
| Upload success | Toast with green check. Evidence list auto-refreshes. |
| Extraction complete | Toast: "47 problems extracted from '{title}'". Problems page shows new data. |
| Clustering complete | Toast: "8 clusters created." Clusters page auto-refreshes. |
| Proposal approved | Button flashes green briefly. Status badge transitions. Row re-sorts in roadmap. |
| Copy to clipboard | Toast: "Copied to clipboard." Button text briefly changes to "Copied ✓" |
| Drag file over dropzone | Border transitions to blue dashed. Background lightens. "Drop to upload" text appears. |
| Error | Toast with red color. Retry button inline. |
| Empty filter result | Table content fades to empty state with "No matches" + clear filter button |

### Loading Skeleton Specifications

Every data-dependent view has a corresponding skeleton:

| Page | Skeleton Description |
|---|---|
| Evidence List | 8 table rows, each with 5 cells. Cells are rounded rectangles (60-80% width) shimmer. |
| Problems Table | Same as evidence but with 6 columns. Severity column skeleton is a small circle. |
| Cluster Grid | 6 cards (3×2 grid). Each card: rectangle for title, thin bar for severity, 2-line rectangle for quote. |
| Proposal Detail | Full-width rectangle for title. 3 section blocks each 80px tall. Action buttons row at bottom. |
| Roadmap Table | 5 table rows with rank numbers, title rectangles, score circles. |
| Dashboard | 5 stat cards as rectangles. Next actions panel as 3 lines. Jobs list as 5 rows. |

---

## 7. User Experience & Navigation Strategy

### Information Architecture

```
Primary Navigation (Sidebar — always visible on desktop)
├── Dashboard       → Pipeline overview, next actions
├── Evidence        → Upload, browse, manage source material
├── Problems        → All extracted problem mentions (table)
├── Clusters        → Grouped pain themes (card grid)
├── Proposals       → Feature specifications (list)
├── Tasks           → Implementation plans (tree view)
├── Roadmap         → Prioritized ranking (scored table)
├── ───────────     → Divider
├── Settings        → API keys, model config, thresholds
└── Usage           → Cost tracking, job history
```

### Sidebar Behavior

| Viewport | Behavior |
|---|---|
| Desktop (≥1024px) | Fixed 256px sidebar. Always visible. Active item highlighted with primary bg color. |
| Tablet (768–1023px) | Collapsed to 56px icon-only sidebar. Hover/click to expand temporarily. |
| Mobile (<768px) | Hidden. Hamburger icon in top bar. Slides in as overlay on click. |

**Sidebar design rules:**
- Logo/brand area at top: "PM" icon badge + "Nexus Pipeline" text
- Nav items: icon (16px) + label. 8px gap. Rounded-xl shape for active state.
- Active state: `bg-primary text-white shadow-sm`
- Hover state: `bg-muted`
- Divider between main nav (Dashboard–Roadmap) and settings nav (Settings, Usage)
- Footer: tagline "Evidence to roadmap, end to end." in muted caption text

### Pipeline Indicator (Horizontal Breadcrumb)

Always visible at the top of the content area. Shows 6 pipeline steps with status dots and counts:

```
● Evidence (12) ── ● Problems (47) ── ● Clusters (8) ── ○ Proposals ── ○ Tasks ── ○ Roadmap
  complete           complete            complete           pending        pending      pending
```

**Behavior:**
- Dots: green (complete), amber pulse (running), gray (pending)
- Counts: shown in muted pill badge next to label
- Click step label → navigates to that page
- On pages within a step (e.g., `/pm/evidence/[id]`), the Evidence step is highlighted/underlined

### Breadcrumb System

Not a traditional breadcrumb bar. Instead, each detail page has a **"← Back to {Parent}"** link at the top left of PageHeader. This is simpler and cleaner than full breadcrumbs for a linear pipeline.

Examples:
- Evidence detail: `← Back to Evidence`
- Problem detail: `← Back to Problems`
- Proposal detail: `← Back to Proposals`
- Task tree: `← Back to Proposal: "Guided Onboarding Wizard"`
- Cluster detail: `← Back to Clusters`

### Search Behavior

- Global search: NOT implemented in Phase 1–3. The pipeline is navigated via sidebar + pipeline indicator.
- Per-page search: Free text search field in filter bars on Problems and Evidence pages. Searches `problem_statement` and `evidence.title` respectively. Debounced 300ms. Server-side via query parameter.

### Keyboard Shortcuts

| Key | Action | Scope |
|---|---|---|
| `1`–`7` | Navigate to Dashboard/Evidence/.../Roadmap | Global (when no input focused) |
| `/` | Focus search field (if present on current page) | Problems, Evidence pages |
| `Esc` | Close open Sheet, Dialog, or expanded row | Global |
| `Enter` on table row | Navigate to detail page | Problems, Evidence, Proposals tables |
| `j` / `k` | Move selection down/up in table | Problems, Evidence tables |

Implementation: `useEffect` with `keydown` listener in `PMLayout`. Check `document.activeElement` to avoid conflicts with text inputs.

### Accessibility Standards

| Standard | Requirement |
|---|---|
| **WCAG 2.1 AA** | Full compliance target |
| **Focus management** | All interactive elements focusable. Focus ring visible (2px blue outline). Tab order follows visual layout. |
| **Color contrast** | All text meets 4.5:1 ratio against background. Tested with our warm neutrals: `ink-primary` (#1A2332) on `surface-0` (#FAFAF6) = 14.8:1. `ink-muted` (#8A9AB5) on `surface-0` = 3.8:1 (use only for non-essential labels; pair with icons). |
| **Screen reader** | All icons have `aria-label`. Tables use `thead`/`tbody`. Status badges have `role="status"`. Pipeline indicator has `aria-live="polite"` for dynamic updates. |
| **Motion** | All animations respect `prefers-reduced-motion: reduce`. |
| **Labels** | All form inputs have associated `<label>` elements. Error messages linked via `aria-describedby`. |

---

## 8. State Management & Data Layer

### Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Data Flow                                  │
│                                                                  │
│  Server Components (RSC)          Client Components              │
│  ┌────────────────────┐          ┌────────────────────────┐     │
│  │ Direct fetch in     │          │ TanStack Query for:    │     │
│  │ page.tsx for initial │          │ - Mutations (POST/PATCH)│    │
│  │ page data (SSR)     │          │ - Polling (job status)  │    │
│  │                     │          │ - Re-fetches after      │    │
│  │ pmFetch() /         │          │   mutation              │    │
│  │ pmFetchSafe()       │          │ - Client-side filters   │    │
│  └────────────────────┘          │   that trigger refetch   │    │
│                                  └────────────────────────────┘  │
│                                                                  │
│  Zustand Stores (Client-only)                                    │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ useJobsStore    → Active job tracking (global)              │  │
│  │ useFilterStore  → Problem/Evidence filters (persist in URL) │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

### Data Fetching Strategy

| Scenario | Strategy | Why |
|---|---|---|
| Initial page load (list pages) | Server Component `fetch()` via `pmFetchSafe()` | SEO not relevant, but avoids client-side loading spinner on first paint. Fast. |
| Detail pages | Server Component `fetch()` | Single resource, no interactivity needed for initial render. |
| Mutations (create, update, delete) | TanStack Query `useMutation()` in client component | Optimistic updates, error rollback, automatic cache invalidation. |
| Filter changes | URL search params + server refetch | Filters bookmarkable/shareable. `useRouter.push()` updates URL, `page.tsx` re-renders with new params. |
| Job polling | TanStack Query `useQuery()` with `refetchInterval` | Automatic 2s polling when job is `pending` or `running`. Stops on `completed`/`failed`. |
| Job tracking (global) | Zustand `useJobsStore` | Jobs started on one page should be visible on all pages (PipelineIndicator, Dashboard). Store syncs across components. |

### Zustand Store Definitions

#### `useJobsStore` (unchanged from current implementation)

```typescript
interface JobsState {
  activeJobs: Map<string, JobStatusResponse>;
  setJob: (id: string, job: JobStatusResponse) => void;
  removeJob: (id: string) => void;
}
```

#### `useFilterStore` (with URL sync)

```typescript
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
}

// Usage: Sync with URL search params on change.
// On page load, initialize from URL params.
// On filter change, update both store and URL.
```

### TanStack Query Configuration

```typescript
// lib/pm/queryClient.ts

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30_000,           // 30 seconds before refetch
      gcTime: 5 * 60_000,         // 5 minutes garbage collection
      retry: 1,                    // One retry on failure
      refetchOnWindowFocus: false, // Don't refetch on tab focus (PM tool, not real-time)
    },
    mutations: {
      retry: 0,                    // No retry on mutations
    },
  },
});
```

### Query Key Structure

```typescript
export const pmKeys = {
  evidence: {
    all: ['pm', 'evidence'] as const,
    list: (filters?: Record<string, string>) => ['pm', 'evidence', 'list', filters] as const,
    detail: (id: string) => ['pm', 'evidence', id] as const,
  },
  problems: {
    all: ['pm', 'problems'] as const,
    list: (filters?: Record<string, string>) => ['pm', 'problems', 'list', filters] as const,
    detail: (id: string) => ['pm', 'problems', id] as const,
    similar: (text: string) => ['pm', 'problems', 'similar', text] as const,
    stats: ['pm', 'problems', 'stats'] as const,
  },
  clusters: {
    all: ['pm', 'clusters'] as const,
    list: () => ['pm', 'clusters', 'list'] as const,
    detail: (id: string) => ['pm', 'clusters', id] as const,
  },
  proposals: {
    all: ['pm', 'proposals'] as const,
    list: (filters?: Record<string, string>) => ['pm', 'proposals', 'list', filters] as const,
    detail: (id: string) => ['pm', 'proposals', id] as const,
  },
  tasks: {
    byProposal: (proposalId: string) => ['pm', 'tasks', proposalId] as const,
  },
  roadmap: {
    ranked: (filters?: Record<string, string>) => ['pm', 'roadmap', filters] as const,
  },
  jobs: {
    detail: (id: string) => ['pm', 'jobs', id] as const,
  },
  costs: {
    summary: ['pm', 'costs', 'summary'] as const,
    calls: ['pm', 'costs', 'calls'] as const,
  },
};
```

### API Client Extensions

Current `lib/pm/api.ts` needs these additions beyond what exists:

```typescript
// ── Proposals (CRUD + actions) ──

export const getProposals = (filters?: Record<string, string>) =>
  pmFetch<PaginatedResponse<Proposal>>(`/feature_proposals?${toQS(filters)}`);

export const getProposalDetail = (id: string) =>
  pmFetch<ProposalDetail>(`/feature_proposals/${id}`);

export const updateProposal = (id: string, data: Partial<Proposal>) =>
  pmFetch<Proposal>(`/feature_proposals/${id}`, {
    method: "PATCH",
    body: JSON.stringify(data),
  });

export const approveProposal = (id: string) =>
  pmFetch<Proposal>(`/feature_proposals/${id}/approve`, { method: "POST" });

export const rejectProposal = (id: string) =>
  pmFetch<Proposal>(`/feature_proposals/${id}/reject`, { method: "POST" });

export const regenerateProposal = (id: string) =>
  pmFetch<JobResponse>(`/feature_proposals/${id}/regenerate`, { method: "POST" });

// ── Tasks ──

export const getTasks = (proposalId: string) =>
  pmFetch<TaskTree>(`/feature_proposals/${proposalId}/tasks`);

export const generateTasks = (proposalId: string) =>
  pmFetch<JobResponse>(`/jobs/generate_tasks`, {
    method: "POST",
    body: JSON.stringify({ proposal_id: proposalId }),
  });

export const generateProposal = (clusterId: string) =>
  pmFetch<JobResponse>(`/jobs/generate_proposal`, {
    method: "POST",
    body: JSON.stringify({ cluster_id: clusterId }),
  });

// ── Roadmap ──

export const updateWeight = (proposalId: string, weight: number) =>
  pmFetch<void>(`/roadmap/${proposalId}/weight`, {
    method: "PATCH",
    body: JSON.stringify({ strategic_weight: weight }),
  });
```

### Types Extensions

Current `lib/pm/types.ts` needs these additions:

```typescript
// ── Proposal Detail ──

export type ProposalStatus = "draft" | "approved" | "rejected" | "archived";
export type ScopeEstimate = "S" | "M" | "L" | "XL";

export interface ProposalDetail extends Proposal {
  user_story: string | null;
  jtbd_framing: string | null;
  rationale: string;
  success_metrics: SuccessMetric[];
  risks: Risk[];
  edge_cases: string[];
  scope_estimate: ScopeEstimate;
  status: ProposalStatus;
  citations: Citation[];
  cluster: Cluster;
  tasks_generated: boolean;
}

export interface SuccessMetric {
  metric: string;
  target: string;
  reasoning: string;
}

export interface Risk {
  risk: string;
  mitigation: string;
  severity: "high" | "medium" | "low";
}

export interface Citation {
  id: string;
  problem_id: string;
  citation_context: string;
  quote_text: string;
  evidence_title: string;
}

// ── Tasks ──

export type TaskCategory = "backend" | "frontend" | "data" | "qa";
export type TaskEffort = "XS" | "S" | "M" | "L" | "XL";

export interface Task {
  id: string;
  proposal_id: string;
  parent_task_id: string | null;
  title: string;
  description: string | null;
  category: TaskCategory;
  acceptance_criteria: string[];
  estimated_effort: TaskEffort | null;
  dependencies: string[];
  sort_order: number;
  subtasks: Task[];
}

export interface TaskTree {
  proposal_id: string;
  feature_name: string;
  backend: Task[];
  frontend: Task[];
  data: Task[];
  qa: Task[];
  total_tasks: number;
}

// ── Roadmap (Extended) ──

export interface PriorityScore {
  frequency_score: number;
  severity_score: number;
  strategic_weight: number;
  effort_estimate: number;
  final_score: number;
}

export interface ScoreBreakdown {
  formula: string;
  frequency: { value: number; explanation: string };
  severity: { value: number; distribution: Record<Severity, number> };
  weight: { value: number; reason: string };
  effort: { value: number; scope: ScopeEstimate };
  final: number;
}
```

---

## 9. Technical Implementation Strategy

### Framework & Stack

| Layer | Technology | Version | Status |
|---|---|---|---|
| **Framework** | Next.js (App Router) | 16.1.6 | Installed |
| **Language** | TypeScript (strict) | 5.x | Configured |
| **Components** | shadcn/ui (Radix primitives) | Latest | Installed |
| **Styling** | Tailwind CSS | 3.x | Configured |
| **State (local)** | Zustand | 5.x | Installed |
| **Data fetching** | TanStack React Query | 5.x | Installed (not yet used) |
| **Tables** | TanStack React Table | 8.x | Installed (not yet used) |
| **Charts** | Recharts | 3.x | Installed (not yet used) |
| **Forms** | React Hook Form + Zod | 7.x + 4.x | Installed (not yet used) |
| **Drag & Drop** | react-dropzone | 14.x | Installed (not yet used) |
| **Toasts** | Sonner | 2.x | Installed (not yet used) |
| **Animation** | Framer Motion | 11.x | Installed, used in PipelineIndicator |
| **Icons** | Lucide React | Latest | Installed, used throughout |

### Component System

- **Base layer:** shadcn/ui provides accessible, unstyled primitives (Button, Dialog, Tabs, etc.). These are in `components/ui/`.
- **PM layer:** Domain-specific components compose shadcn primitives with PM business logic. These are in `components/pm/`.
- **No component duplication:** If a shadcn component does the job (Button, Badge, Card), use it directly with Tailwind class overrides. Only create a PM-specific wrapper when the component needs PM domain logic (e.g., `SeverityBadge` wraps `Badge` with severity → color mapping).

### Theming Strategy

- **Light mode only** for the PM pipeline. The `.pm-root` class overrides CSS custom properties from the dark-mode root. This is intentional: PM pipeline is a daytime productivity tool with warm, paper-like tones.
- The existing dark-mode root layout (`<html className="dark">`) is preserved for the non-PM workspace (canvas, chat). The PM layout wraps children in `.pm-root` which flips all HSL variables to light values.
- **No dark mode toggle** for PM pipeline. Complexity not justified for V1.

### Performance Optimization

| Area | Strategy |
|---|---|
| **Bundle size** | Tree-shake Lucide icons (named imports only). Tree-shake Recharts (import only `BarChart`, not all). Avoid importing full `lodash` — use native methods. |
| **Images** | No images in PM pipeline. Data-driven UI only. |
| **Server Components** | All list and detail pages are Server Components (`async function Page()`). This means initial HTML includes data — no client-side loading spinner on first paint. |
| **Client Components** | Only interactive elements (`"use client"`): filter bar, pipeline indicator, job progress, forms, expandable rows, sliders. |
| **Code splitting** | Next.js App Router handles this automatically. Each page is lazy-loaded. |
| **Font loading** | IBM Plex Sans and Fraunces loaded via `next/font/google` with `display: swap`. Already implemented. |
| **API proxy** | Next.js rewrites `/api/v1/*` to backend. Avoids CORS and reduces client-side complexity. |
| **Caching** | TanStack Query `staleTime: 30s`. Server Component fetches use `cache: "no-store"` (pipeline data changes frequently). |
| **Prefetching** | Next.js `<Link>` automatically prefetches linked pages on hover. |

### Backend Integration

```
Frontend (Next.js)                    Backend (FastAPI)
─────────────────                    ──────────────────
Browser → /api/v1/*  ──rewrite──→    localhost:8000/api/v1/*
SSR     → http://localhost:8000/api/v1/*  (direct, skips rewrite)

next.config.js:
{
  async rewrites() {
    return [
      { source: '/api/:path*', destination: 'http://localhost:8000/api/:path*' }
    ];
  }
}
```

All API calls go through `pmFetch()` / `pmFetchSafe()` which handle:
- Auto-detecting server vs. client context for URL resolution
- JSON Content-Type headers
- Error extraction from response body
- Type-safe response parsing

---

## 10. Engagement Strategy

### Onboarding Flow (First-Use Experience)

When a user first arrives at `/pm` with no evidence:

1. **Dashboard** shows a single, prominent empty state:
   ```
   ┌─────────────────────────────────────────────────────┐
   │                                                     │
   │  📄  Start your product discovery pipeline          │
   │                                                     │
   │  Upload a customer interview, support ticket,       │
   │  or sales note. Nexus will extract problems,        │
   │  find patterns, and generate feature proposals.     │
   │                                                     │
   │  [Upload Your First Evidence →]                     │
   │                                                     │
   └─────────────────────────────────────────────────────┘
   ```

2. After upload and extraction, the **Problems page** shows a banner:
   ```
   ✅ 12 problems extracted from "Customer Interview - Acme Corp"
   Next step: Upload more evidence, or cluster your problems → [Run Clustering]
   ```

3. After clustering, the **Clusters page** shows:
   ```
   ✅ 4 clusters created from 47 problems.
   Next step: Click a cluster to explore, then generate a feature proposal.
   ```

4. This pattern continues through the pipeline. Each page suggests the next action until the user reaches the roadmap.

**Implementation:** `NextActionsPanel` component on the dashboard computes the suggested action based on pipeline counts. Individual pages show contextual banners via `EmptyState` or inline alert when the pipeline has a clear next step.

### Progressive Disclosure

- List pages show summary data (title, severity badge, count). Click to see detail.
- Detail pages show structured sections. Each section is collapsed by default except the first. Click to expand.
- Roadmap scores show a single number. Click to expand the full formula breakdown.
- Task tree shows titles with effort badges. Click to expand description and acceptance criteria.

### Tooltips

- **Severity badges:** Tooltip on hover explains severity level: "Critical: Product is unusable for this use case."
- **Pipeline status dots:** Tooltip shows count and "Click to view."
- **Score formula:** Tooltip on the "Score" column header explains: "Priority score = (frequency × severity × weight) / effort."
- **Citation links:** Tooltip on hover shows the quoted text preview (first 100 chars). Click to see full context.

**Implementation:** Radix `Tooltip` (already configured with `TooltipProvider` in root layout, `delayDuration={0}`).

### Subtle Delight Moments

| Moment | Implementation |
|---|---|
| Pipeline step completes | Status dot transitions from gray → green with a brief scale-up pulse (1.2x → 1x, 300ms) |
| Score recalculation | Final score number does a slight counter-roll animation (old number fades up, new number fades in from below) |
| Job completes | Sonner toast slides in with a green check icon. Auto-dismisses after 5 seconds. |
| Upload form drop | Dropzone border pulses blue once on successful file accept |
| All pipeline steps complete | Dashboard shows a subtle confetti-like particle effect (10 small dots, once, 1 second). Intentionally understated. Optional — can be disabled in settings. |

---

## 11. Quality Bar

### Measurable Standards

| Metric | Target | Measurement |
|---|---|---|
| **Lighthouse Performance** | ≥ 90 | Chrome DevTools audit on all page routes |
| **Lighthouse Accessibility** | ≥ 95 | Chrome DevTools audit |
| **Lighthouse Best Practices** | ≥ 95 | Chrome DevTools audit |
| **First Contentful Paint** | < 1.0s | On local dev; < 2.0s on deployed |
| **Largest Contentful Paint** | < 1.5s | Server components render data without client waterfall |
| **Cumulative Layout Shift** | < 0.05 | Skeletons match final layout dimensions |
| **Animation Performance** | 60fps | All transitions use GPU-composited properties (transform, opacity). No layout thrashing during animation. |
| **Mobile Responsiveness** | All pages usable at 375px width | Tables switch to stacked card layout. Sidebar hidden. |
| **TypeScript Strict** | Zero `any` types in PM code | `strict: true` in tsconfig. No `@ts-ignore` in PM files. |
| **Bundle Size (JS)** | < 200KB first-load JS per page | Next.js bundle analyzer |
| **Build** | Zero warnings | `next build` completes cleanly |

### Design Consistency Rules

1. All border-radius values are multiples of 4px. Cards: 16px. Buttons: 12px. Badges: 9999px (pill). Inputs: 12px.
2. All shadows use the same set: `shadow-none`, `shadow-sm` (hover), `shadow-md` (elevated cards). No custom shadows.
3. All spacing uses the Tailwind scale. No `px-[17px]` arbitrary values except for the 28px grid background.
4. All severity colors are used consistently across every page. Red = critical. Orange = high. Amber = medium. Green = low. Never mixed.
5. All page headers use the same `PageHeader` component. No custom hero sections.
6. All empty states use the same `EmptyState` component. Same icon size, same text hierarchy, same CTA button style.
7. All tables use the same `DataTable` wrapper. Same row height, same header style, same pagination controls.

---

## 12. Development Phases

### Phase 1: Foundation + Evidence + Problems (Weeks 1–3)

**Goal:** Core infrastructure + upload + browse + extract pipeline.

| Week | Deliverables |
|---|---|
| **1** | Install `sonner` toaster in PM layout. Create `EmptyState`, `SeverityBadge`, `StatusBadge`, `ScopeBadge`, `SkeletonTable` shared components. Update CSS variables to match new color spec. Add file dropzone to evidence upload (react-dropzone). |
| **2** | Build `DataTable` wrapper (TanStack Table). Refactor Evidence list to use `DataTable` with sorting + pagination controls. Add evidence source type & persona filter dropdowns. Build `JobProgress` component with polling. |
| **3** | Build `ProblemFilters` filter bar (persona, severity, tags, search). Refactor Problems table to use `DataTable` with expandable rows. Build `ProblemExpandedRow` with full quote + source link. Build `SimilarProblemsPanel` (Sheet). Wire `useFilterStore` to URL params. |

**Exit Criteria:**
- Upload a transcript (paste or drag-drop) → see it in evidence list → extraction job runs with visible progress → problems appear in filterable table
- Expand problem row → see full quote with source attribution → click "View Similar" → side panel shows ranked similar problems
- All loading states show skeletons. All empty states show CTA. All errors show toast + retry.

---

### Phase 2: Clusters + Proposals (Weeks 4–6)

**Goal:** Clustering visualization + proposal generation + review workflow.

| Week | Deliverables |
|---|---|
| **4** | Build `ClusterCard` with severity mini-bar and top quote. Build `ClusterGrid` (responsive). Add "Run Clustering" button with `JobProgress`. Build `SeverityChart` (Recharts horizontal bar chart). |
| **5** | Build `ClusterDetailView` with severity chart, persona breakdown, full quote list, members table. Add "Generate Proposal" button with `JobProgress`. Build `ProposalList` with status badges and filters. |
| **6** | Build `ProposalDetailView` — full spec rendering with user story, rationale, metrics table, risks table. Build `CitationLink` (clickable superscript → Sheet showing source). Build `ProposalActions` (approve/reject/regenerate) with confirmation dialogs. Build `ProposalEditor` for inline field editing. |

**Exit Criteria:**
- Click "Run Clustering" → job runs → cluster cards appear sorted by mention count with severity bars
- Click cluster → see full detail with quotes → click "Generate Proposal" → proposal appears
- Open proposal → see full spec with clickable citations → approve/reject proposal → status updates in list + roadmap
- Edit proposal fields inline → save → changes persist

---

### Phase 3: Tasks + Roadmap (Weeks 7–9)

**Goal:** Complete the pipeline — task generation and priority ranking.

| Week | Deliverables |
|---|---|
| **7** | Build `TaskTree`, `TaskNode`, `TaskCategoryTabs` (Radix Tabs). Render collapsible tree with effort badges and acceptance criteria. Add "Generate Tasks" button on proposal detail. |
| **8** | Build `TaskExport` — copy as structured markdown (clipboard API) + download as JSON (Blob). Build `RoadmapTable` — ranked list using `DataTable` with additional columns (rank, scope, status, score). |
| **9** | Build `ScoreBreakdown` — expandable inline row with formula visualization + value bars. Build `WeightSlider` (Radix Slider) — adjusts strategic weight, recalculates score client-side, persists on release. Build `RoadmapFilters` — persona/segment/status dropdowns. |

**Exit Criteria:**
- From approved proposal, click "Generate Tasks" → task tree appears with Backend/Frontend/Data/QA tabs → each task expandable with acceptance criteria
- Export task tree as markdown → paste into Linear/GitHub issue and formatting is correct
- Roadmap shows all proposals ranked by score → expand score breakdown → adjust weight → ranking re-sorts
- Full end-to-end pipeline: Upload → Extract → Cluster → Generate Proposal → Review → Approve → Generate Tasks → View Roadmap

---

### Phase 4: Polish + Settings + Dashboard (Weeks 10–12)

| Week | Deliverables |
|---|---|
| **10** | Enhance Dashboard: add `NextActionsPanel` (computed suggestions), `RecentJobsList`, `ActiveJobsPanel` with live polling. First-use empty state with onboarding CTA. |
| **11** | Build Settings page: API key config (masked input), model selector, chunk size/overlap, clustering threshold slider. Build Usage page: job history table with per-job cost breakdown. |
| **12** | Keyboard shortcuts (`1`–`7` nav, `/` search, `j`/`k` row nav). Evidence detail tabs (raw text with chunk highlighting, extracted problems, processing history). Responsive breakpoints for tablet (collapsible sidebar). Accessibility audit against WCAG 2.1 AA. Loading skeleton refinement. Error boundary wrappers on all pages. |

**Exit Criteria:**
- Dashboard gives pipeline status at a glance with computed next-action suggestions
- Settings page persists API config and extraction parameters
- Usage page shows per-job and per-model cost breakdowns
- All keyboard shortcuts functional. Tab navigation works for all interactive elements. Screen reader announcements for status changes.
- Lighthouse scores meet targets: Performance ≥ 90, Accessibility ≥ 95

---

## Summary

Nexus PM is a **six-stage pipeline tool** — not a dashboard, not an editor, not a project manager. Every design decision serves the pipeline:

```
Upload Evidence → Extract Problems → Cluster Pains → Generate Proposals → Plan Tasks → Rank Roadmap
```

**Every screen traces back to a quote.** Every quote traces back to a document. Every proposal traces back to a cluster of quotes. Every roadmap rank traces back to a formula the user can inspect and adjust.

The visual identity is warm, confident, and data-dense: **Fraunces serif headings** for editorial authority, **IBM Plex Sans** for clinical precision, **teal blue** for trust, **amber yellow** for insight, and **warm ivory surfaces** that make the tool feel premium without being precious.

The technical stack is proven: **Next.js App Router** with Server Components for fast initial render, **TanStack Query** for mutation management and job polling, **TanStack Table** for data-dense tables, **Recharts** for severity charts, and **shadcn/ui** for accessible primitives.

Build Phase 1 in parallel with backend. Ship evidence upload + problem extraction first. Validate the UX. Then proceed.
