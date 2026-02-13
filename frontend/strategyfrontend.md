# Nexus Frontend Strategy

> **"Cursor for Product Managers"** — A clean, evidence-driven interface that takes PMs from raw customer signal to prioritized roadmap + dev-ready task trees.

---

## Table of Contents

1. [Design Philosophy](#design-philosophy)
2. [Tech Stack](#tech-stack)
3. [Information Architecture](#information-architecture)
4. [Core UI Flows](#core-ui-flows)
5. [Component Architecture](#component-architecture)
6. [State Management](#state-management)
7. [API Integration Layer](#api-integration-layer)
8. [Page-by-Page Specification](#page-by-page-specification)
9. [Design System & Patterns](#design-system--patterns)
10. [Development Phases](#development-phases)

---

## Design Philosophy

### Core Principles

1. **Evidence is king** — Every screen traces back to quotes. No insight without a source.
2. **Progressive disclosure** — Show summaries first, drill into detail on demand.
3. **Pipeline visibility** — Users always know where they are in Evidence → Problems → Clusters → Proposals → Tasks → Roadmap.
4. **Async-aware UX** — LLM jobs take time. Show progress, not spinners. Let users continue working.
5. **Light editing, not authoring** — AI generates, humans approve/refine. Don't build a document editor.
6. **Density over decoration** — PMs scan lots of data. Information density matters more than whitespace.

### What This Is NOT

- Not a document editor (no rich-text authoring)
- Not a project management tool (no sprints, no kanban, no assignees)
- Not a dashboard-heavy analytics product (minimal charts, max utility)
- Not the RAPTOR/canvas/agent interface from the README (that's a separate UI surface)

The PM pipeline UI is a **linear workflow tool** with six distinct views, each corresponding to one pipeline stage.

---

## Tech Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| **Framework** | Next.js 14 (App Router) | Already in workspace; Server Components for data-heavy pages |
| **Language** | TypeScript (strict) | Type safety for complex data models |
| **Components** | shadcn/ui | Already installed; clean, accessible, customizable |
| **Styling** | Tailwind CSS | Already configured; utility-first for fast iteration |
| **State** | Zustand | Already in use (`store.ts`); lightweight, simple |
| **Data Fetching** | TanStack Query (React Query) | Cache management, background refetching, optimistic updates |
| **Tables** | TanStack Table | Sortable, filterable, paginated data tables for problems/clusters |
| **Charts** | Recharts (minimal) | Severity distribution bars, score breakdowns only |
| **Forms** | React Hook Form + Zod | Validation for evidence upload, proposal editing |
| **Drag & Drop** | Native HTML5 / react-dropzone | File upload only (no complex DnD needed) |
| **Toast/Notifications** | sonner | Job completion notifications, error alerts |
| **Icons** | Lucide React | Already used with shadcn/ui |

### What We're NOT Adding

- No React Flow (that's for the canvas/RAPTOR workspace, separate from PM pipeline)
- No Three.js / XR (that's AR mode, future)
- No complex animation libraries
- No markdown editors (light text editing only)

---

## Information Architecture

### Navigation Structure

```
┌─────────────────────────────────────────────────────────────┐
│  Sidebar (persistent)                                        │
│  ┌─────────────────┐                                        │
│  │ 🏠 Dashboard     │  ← Pipeline overview + recent activity│
│  │ 📄 Evidence      │  ← Upload & browse source material    │
│  │ ⚠️ Problems      │  ← Extracted problem mentions         │
│  │ 📊 Clusters      │  ← Grouped pain themes                │
│  │ 💡 Proposals     │  ← Feature proposals                  │
│  │ 🔨 Tasks         │  ← Implementation task trees          │
│  │ 🗺️ Roadmap       │  ← Prioritized ranking                │
│  │ ─────────────── │                                        │
│  │ ⚙️ Settings      │  ← API keys, prompt versions          │
│  │ 📈 Usage         │  ← Cost tracking, job history         │
│  └─────────────────┘                                        │
└─────────────────────────────────────────────────────────────┘
```

### Pipeline Flow (Visual Breadcrumb)

Every page shows a horizontal pipeline indicator at the top:

```
Evidence ──→ Problems ──→ Clusters ──→ Proposals ──→ Tasks ──→ Roadmap
   ✅           ✅          ⏳            ○            ○         ○
 12 docs      47 items    running...    pending      pending   pending
```

This gives users constant awareness of pipeline state and progress.

### URL Structure

```
/pm                              → Dashboard (pipeline overview)
/pm/evidence                     → Evidence list
/pm/evidence/upload              → Upload new evidence
/pm/evidence/[id]                → Evidence detail (text + chunks + extracted problems)
/pm/problems                     → All problem mentions (filterable table)
/pm/problems/[id]                → Problem detail (quote, source, similar problems)
/pm/clusters                     → Cluster grid/list
/pm/clusters/[id]                → Cluster detail (members, quotes, severity breakdown)
/pm/proposals                    → Proposal list (with status filters)
/pm/proposals/[id]               → Proposal detail (full spec, citations, edit mode)
/pm/proposals/[id]/tasks         → Task tree for proposal
/pm/roadmap                      → Ranked roadmap view
/pm/settings                     → Configuration
/pm/usage                        → Cost & job tracking
```

---

## Core UI Flows

### Flow 1: Upload Evidence

```
User Journey:
1. Click "Upload Evidence" or drag file onto Evidence page
2. Modal: paste text OR drop file (txt, pdf, csv)
3. Fill metadata: title, source type, persona, segment, date
4. Submit → toast: "Processing started" → redirect to evidence list
5. Evidence row shows status: uploading → chunking → extracting → done
6. Click evidence → see extracted problems inline

States:
- Empty state: "No evidence yet. Upload your first transcript."
- Loading: skeleton rows
- Processing: progress indicator per evidence item
- Error: retry button + error message
```

**Upload Modal Wireframe:**
```
┌──────────────────────────────────────────────────────┐
│  Upload Evidence                              [✕]    │
│                                                      │
│  ┌──────────────────────────────────────────────┐   │
│  │                                              │   │
│  │     Drop transcript file here                │   │
│  │     or paste text below                      │   │
│  │     (.txt, .pdf, .csv)                       │   │
│  │                                              │   │
│  └──────────────────────────────────────────────┘   │
│                                                      │
│  ┌──────────────────────────────────────────────┐   │
│  │  [Paste transcript text here...]             │   │
│  │                                              │   │
│  │                                              │   │
│  └──────────────────────────────────────────────┘   │
│                                                      │
│  Title:     [Customer Interview - Acme Corp     ]   │
│  Type:      [Interview          ▼]                  │
│  Persona:   [Product Manager    ]                   │
│  Segment:   [Enterprise         ]                   │
│  Date:      [2026-01-15         ]                   │
│                                                      │
│                          [Cancel]  [Upload & Process]│
└──────────────────────────────────────────────────────┘
```

---

### Flow 2: Review Problems

```
User Journey:
1. Navigate to Problems page
2. See table of all extracted problem mentions
3. Filter by: persona, severity, tags, source type, date range
4. Each row shows: problem statement, severity badge, persona, tags, quote preview
5. Click row → expand to show full quote + source reference
6. Click "View Similar" → see embedding-based similar problems
7. Bulk actions: re-extract, delete, tag

States:
- Empty: "No problems extracted yet. Upload evidence to get started."
- Filtering: instant client-side filter + server pagination
- Similarity panel: side drawer with similar problems ranked by distance
```

**Problems Table Wireframe:**
```
┌──────────────────────────────────────────────────────────────────────┐
│  Problems (47)                                                       │
│                                                                      │
│  [Persona ▼] [Severity ▼] [Tags ▼] [Source Type ▼] [Search...    ] │
│                                                                      │
│  ┌───┬──────────────────────────────┬──────────┬─────────┬────────┐ │
│  │   │ Problem                      │ Severity │ Persona │ Tags   │ │
│  ├───┼──────────────────────────────┼──────────┼─────────┼────────┤ │
│  │ ▸ │ Permissions config too complex│ 🔴 HIGH  │ PM      │ perms  │ │
│  │ ▸ │ Onboarding takes >2 hours    │ 🔴 HIGH  │ Admin   │ onbd   │ │
│  │ ▸ │ Reports load too slowly      │ 🟡 MED   │ Analyst │ perf   │ │
│  │ ▸ │ Can't export to CSV          │ 🟡 MED   │ PM      │ export │ │
│  │ ▸ │ Mobile app crashes on login  │ 🔴 CRIT  │ User    │ mobile │ │
│  └───┴──────────────────────────────┴──────────┴─────────┴────────┘ │
│                                                                      │
│  Expanded row:                                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ "I spent 3 hours trying to set up my first project and      │   │
│  │  still couldn't figure out permissions"                     │   │
│  │                                                              │   │
│  │  Source: Customer Interview - Acme Corp PM  |  Jan 15, 2026 │   │
│  │  [View Similar] [View Source] [Edit Tags]                   │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  Page 1 of 3  [← Prev] [Next →]                                    │
└──────────────────────────────────────────────────────────────────────┘
```

---

### Flow 3: Explore Clusters

```
User Journey:
1. Navigate to Clusters page
2. See card grid of pain clusters
3. Each card: label, member count, severity distribution mini-bar, top quote
4. Click card → detail page with all members, full quote list, severity chart
5. From detail: "Generate Feature Proposal" button → triggers LLM job
6. See job progress → proposal appears when done

States:
- Empty: "No clusters yet. Extract problems first, then cluster."
- Unclustered: "23 unclustered problems. [Run Clustering]"
- Processing: job progress bar during clustering
- Results: card grid sorted by member count (biggest pains first)
```

**Cluster Card Wireframe:**
```
┌─────────────────────────────────────────┐
│  Onboarding flow is confusing           │
│                                         │
│  23 mentions                            │
│  ████████░░░░  Severity: 3.2 avg        │
│  CRIT:4  HIGH:11  MED:7  LOW:1         │
│                                         │
│  "I spent 3 hours trying to set up      │
│   my first project..."                  │
│   — Acme Corp PM                        │
│                                         │
│  [View Details]  [Generate Proposal →]  │
└─────────────────────────────────────────┘
```

**Cluster Detail Page Wireframe:**
```
┌──────────────────────────────────────────────────────────────────┐
│  ← Back to Clusters                                              │
│                                                                  │
│  Onboarding flow is confusing                                    │
│  ──────────────────────────────────────────                      │
│                                                                  │
│  Summary: Multiple users across enterprise and mid-market        │
│  segments report difficulty completing initial setup. Key         │
│  friction points include permissions configuration and           │
│  project creation workflows.                                     │
│                                                                  │
│  ┌──────────────────────────────┐  ┌─────────────────────────┐  │
│  │  Severity Distribution       │  │  By Persona             │  │
│  │  ████████████ Critical: 4    │  │  PM: 12                 │  │
│  │  ██████████████████ High: 11 │  │  Admin: 7               │  │
│  │  ████████████ Medium: 7      │  │  Developer: 3           │  │
│  │  ██ Low: 1                   │  │  User: 1                │  │
│  └──────────────────────────────┘  └─────────────────────────┘  │
│                                                                  │
│  Top Quotes                                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ "our team gave up on onboarding after day two"          │   │
│  │  — Support Ticket #4521  |  🔴 CRITICAL                 │   │
│  ├──────────────────────────────────────────────────────────┤   │
│  │ "I spent 3 hours trying to set up my first project"     │   │
│  │  — Acme Corp PM Interview  |  🔴 HIGH                   │   │
│  ├──────────────────────────────────────────────────────────┤   │
│  │ "the permissions model is incomprehensible"             │   │
│  │  — Sales Call Notes - BigCo  |  🔴 HIGH                 │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  All Members (23)                                                │
│  [Sortable table of problem mentions...]                        │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Proposal Status: ○ Not generated                        │  │
│  │                                                           │  │
│  │  [Generate Feature Proposal →]                           │  │
│  └───────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

---

### Flow 4: Review & Edit Proposals

```
User Journey:
1. Navigate to Proposals page (or arrive from cluster detail)
2. See list of proposals with status badges (draft, approved, rejected)
3. Click proposal → full spec view
4. Review: feature name, one-liner, user story, rationale WITH citations
5. Citations are clickable → jump to source problem/quote
6. Light edit: modify any text field inline
7. Actions: Approve / Reject / Regenerate / Generate Tasks
8. Status changes reflected in list + roadmap

States:
- Draft: editable, can regenerate
- Approved: locked for editing, counts toward roadmap
- Rejected: grayed out, excluded from roadmap
- Generating: LLM job in progress
```

**Proposal Detail Wireframe:**
```
┌──────────────────────────────────────────────────────────────────────┐
│  ← Back to Proposals                                                 │
│                                                                      │
│  Guided Onboarding Wizard                          Status: DRAFT     │
│  "Step-by-step setup flow replacing the current blank-slate          │
│   experience"                                                        │
│  ──────────────────────────────────────────────────────              │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  User Story                                                  │    │
│  │  As a new admin, I want a guided setup wizard so that I can  │    │
│  │  configure permissions and create my first project in under  │    │
│  │  30 minutes.                                                 │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Why This Matters                                            │    │
│  │                                                               │    │
│  │  23 customers across enterprise and mid-market segments      │    │
│  │  report onboarding friction as their #1 pain point.          │    │
│  │                                                               │    │
│  │  Users are abandoning setup entirely: [1] "our team gave up  │    │
│  │  on onboarding after day two" (Support Ticket #4521).        │    │
│  │                                                               │    │
│  │  The time investment is prohibitive: [2] "I spent 3 hours    │    │
│  │  trying to set up my first project" (Acme Corp PM).          │    │
│  │                                                               │    │
│  │  [1][2] = clickable citation links → source evidence         │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  Success Metrics                                                     │
│  ┌────────────────────────────┬──────────┬───────────────────────┐  │
│  │ Metric                     │ Target   │ Reasoning              │  │
│  ├────────────────────────────┼──────────┼───────────────────────┤  │
│  │ Onboarding completion rate │ >80%     │ Currently ~40% est.   │  │
│  │ Time to first project      │ <30 min  │ Currently 2-3 hours   │  │
│  │ Support tickets (onboard)  │ -50%     │ 34% of tickets today  │  │
│  └────────────────────────────┴──────────┴───────────────────────┘  │
│                                                                      │
│  Risks                                                               │
│  ┌────────────────────────────┬──────────┬──────────────────────┐   │
│  │ Risk                       │ Severity │ Mitigation            │   │
│  ├────────────────────────────┼──────────┼──────────────────────┤   │
│  │ Power users feel restricted│ Medium   │ "Skip wizard" option │   │
│  │ Edge cases in permissions  │ High     │ Fallback to manual   │   │
│  └────────────────────────────┴──────────┴──────────────────────┘   │
│                                                                      │
│  Scope: M (1-3 weeks)  |  Cluster: Onboarding flow is confusing    │
│                                                                      │
│  [✏️ Edit] [🔄 Regenerate] [✅ Approve] [❌ Reject] [🔨 Generate Tasks] │
└──────────────────────────────────────────────────────────────────────┘
```

---

### Flow 5: View Task Tree

```
User Journey:
1. From proposal detail, click "Generate Tasks"
2. Job runs → task tree appears
3. Nested tree view grouped by category: Backend / Frontend / Data / QA
4. Each task shows: title, effort badge, acceptance criteria (expandable)
5. Dependencies shown as subtle connector lines or badges
6. Export button: copy as markdown, future: push to Linear/GitHub

States:
- Not generated: "Generate Tasks" button
- Generating: progress indicator
- Generated: collapsible tree view
- Exportable: copy/download as markdown or JSON
```

**Task Tree Wireframe:**
```
┌──────────────────────────────────────────────────────────────────┐
│  Implementation Plan: Guided Onboarding Wizard                   │
│  Proposal: Guided Onboarding Wizard  |  Total: 18 tasks         │
│  ──────────────────────────────────────────                      │
│                                                                  │
│  [Backend] [Frontend] [Data] [QA]  ← category tabs              │
│                                                                  │
│  Backend (6 tasks)                                               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ ▾ Create onboarding state machine API                    │   │
│  │   Effort: M (1-3 days)                                   │   │
│  │   Depends on: Create onboarding_progress table           │   │
│  │                                                           │   │
│  │   Acceptance Criteria:                                    │   │
│  │   ☐ Given a new user, when POST /onboarding/start,       │   │
│  │     then create progress record with step=1              │   │
│  │   ☐ Given step completion, when POST /onboarding/next,   │   │
│  │     then advance to next step and persist state          │   │
│  │   ☐ Given all steps complete, when GET /onboarding/status│   │
│  │     then return {completed: true}                        │   │
│  ├──────────────────────────────────────────────────────────┤   │
│  │ ▸ Create permissions template endpoint          S        │   │
│  ├──────────────────────────────────────────────────────────┤   │
│  │ ▸ Add onboarding progress tracking             M        │   │
│  ├──────────────────────────────────────────────────────────┤   │
│  │ ▸ Create project scaffolding endpoint           M        │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  Data (2 tasks)                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ ▸ Create onboarding_progress table              S        │   │
│  │ ▸ Add default permission templates seed data    S        │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  [📋 Copy as Markdown]  [📥 Download JSON]  [🔄 Regenerate]    │
└──────────────────────────────────────────────────────────────────┘
```

---

### Flow 6: Roadmap View

```
User Journey:
1. Navigate to Roadmap page
2. See ranked list of approved proposals sorted by priority score
3. Each row: rank, feature name, score, score breakdown (expandable)
4. Click score breakdown → see frequency × severity × weight / effort
5. Adjust strategic weight via inline slider → score recalculates
6. Filter by persona, segment, tag
7. Visual: simple ranked list (not a timeline/gantt chart)

States:
- Empty: "No approved proposals. Generate and approve proposals first."
- Populated: ranked list with expandable score details
- Adjustable: strategic weight slider triggers recalculation
```

**Roadmap Wireframe:**
```
┌──────────────────────────────────────────────────────────────────────┐
│  Roadmap                                                             │
│  12 proposals  |  Last clustered: Feb 12, 2026                      │
│                                                                      │
│  [Persona ▼] [Segment ▼] [Tag ▼]                                   │
│                                                                      │
│  ┌────┬───────────────────────────┬───────┬───────────────────────┐  │
│  │ #  │ Feature                    │ Score │ Breakdown             │  │
│  ├────┼───────────────────────────┼───────┼───────────────────────┤  │
│  │ 1  │ Guided Onboarding Wizard  │ 42.5  │ ▸ freq:34 sev:3.2    │  │
│  │    │ M scope  |  ✅ approved    │       │   wt:1.2 eff:3       │  │
│  ├────┼───────────────────────────┼───────┼───────────────────────┤  │
│  │ 2  │ Real-time Report Engine   │ 38.1  │ ▸ freq:28 sev:2.8    │  │
│  │    │ L scope  |  ✅ approved    │       │   wt:1.0 eff:8       │  │
│  ├────┼───────────────────────────┼───────┼───────────────────────┤  │
│  │ 3  │ Granular Permissions v2   │ 31.7  │ ▸ freq:19 sev:3.5    │  │
│  │    │ L scope  |  ✅ approved    │       │   wt:1.0 eff:8       │  │
│  ├────┼───────────────────────────┼───────┼───────────────────────┤  │
│  │ 4  │ CSV/Excel Export          │ 28.3  │ ▸ freq:15 sev:2.1    │  │
│  │    │ S scope  |  📝 draft      │       │   wt:1.0 eff:1       │  │
│  └────┴───────────────────────────┴───────┴───────────────────────┘  │
│                                                                      │
│  Expanded Score Breakdown (#1):                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Formula: (frequency × severity × weight) / effort           │   │
│  │                                                               │   │
│  │  Frequency:  34.0  (23 of 68 total mentions)    ████████░░  │   │
│  │  Severity:   3.2   (avg across cluster)         ████████░░  │   │
│  │  Weight:     1.2   [━━━━━━━●━━━] adjustable     ████████░░  │   │
│  │  Effort:     3     (M scope = 3 units)          ██████░░░░  │   │
│  │  ─────────────────────────────────────────                   │   │
│  │  Score:      42.5  = (34 × 3.2 × 1.2) / 3                  │   │
│  └──────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Component Architecture

### Directory Structure (PM Pipeline)

```
frontend/
├── app/
│   ├── pm/                              # PM pipeline route group
│   │   ├── layout.tsx                   # PM layout with sidebar + pipeline indicator
│   │   ├── page.tsx                     # Dashboard / pipeline overview
│   │   ├── evidence/
│   │   │   ├── page.tsx                 # Evidence list
│   │   │   ├── upload/
│   │   │   │   └── page.tsx             # Upload page (or modal)
│   │   │   └── [id]/
│   │   │       └── page.tsx             # Evidence detail
│   │   ├── problems/
│   │   │   ├── page.tsx                 # Problem mentions table
│   │   │   └── [id]/
│   │   │       └── page.tsx             # Problem detail
│   │   ├── clusters/
│   │   │   ├── page.tsx                 # Cluster grid
│   │   │   └── [id]/
│   │   │       └── page.tsx             # Cluster detail
│   │   ├── proposals/
│   │   │   ├── page.tsx                 # Proposal list
│   │   │   └── [id]/
│   │   │       ├── page.tsx             # Proposal detail
│   │   │       └── tasks/
│   │   │           └── page.tsx         # Task tree for proposal
│   │   ├── roadmap/
│   │   │   └── page.tsx                 # Prioritized roadmap
│   │   ├── settings/
│   │   │   └── page.tsx                 # API keys, prompt versions
│   │   └── usage/
│   │       └── page.tsx                 # Cost tracking
│   ├── globals.css
│   ├── layout.tsx                       # Root layout
│   └── page.tsx                         # Landing / workspace selector
├── components/
│   ├── pm/                              # PM pipeline components
│   │   ├── pipeline/
│   │   │   ├── PipelineIndicator.tsx    # Horizontal pipeline status bar
│   │   │   └── PipelineStep.tsx         # Individual step with status
│   │   ├── evidence/
│   │   │   ├── EvidenceList.tsx         # Evidence table/list
│   │   │   ├── EvidenceCard.tsx         # Evidence summary card
│   │   │   ├── EvidenceDetail.tsx       # Full evidence view with chunks
│   │   │   ├── UploadModal.tsx          # Upload form with drag-drop
│   │   │   └── EvidenceFilters.tsx      # Filter controls
│   │   ├── problems/
│   │   │   ├── ProblemTable.tsx         # Filterable problem table
│   │   │   ├── ProblemRow.tsx           # Expandable table row
│   │   │   ├── ProblemDetail.tsx        # Full problem view
│   │   │   ├── SimilarProblems.tsx      # Side panel with similar items
│   │   │   ├── SeverityBadge.tsx        # Color-coded severity indicator
│   │   │   └── ProblemFilters.tsx       # Filter controls
│   │   ├── clusters/
│   │   │   ├── ClusterGrid.tsx          # Card grid layout
│   │   │   ├── ClusterCard.tsx          # Summary card with mini chart
│   │   │   ├── ClusterDetail.tsx        # Detail page content
│   │   │   ├── SeverityChart.tsx        # Horizontal bar chart
│   │   │   └── QuoteList.tsx            # Formatted quote list with sources
│   │   ├── proposals/
│   │   │   ├── ProposalList.tsx         # List with status badges
│   │   │   ├── ProposalDetail.tsx       # Full proposal spec
│   │   │   ├── ProposalEditor.tsx       # Inline editable fields
│   │   │   ├── CitationLink.tsx         # Clickable citation → source
│   │   │   ├── MetricsTable.tsx         # Success metrics display
│   │   │   ├── RisksTable.tsx           # Risks + mitigations
│   │   │   └── ProposalActions.tsx      # Approve/Reject/Regenerate buttons
│   │   ├── tasks/
│   │   │   ├── TaskTree.tsx             # Collapsible hierarchical tree
│   │   │   ├── TaskNode.tsx             # Individual task with details
│   │   │   ├── TaskCategoryTabs.tsx     # Backend/Frontend/Data/QA tabs
│   │   │   ├── AcceptanceCriteria.tsx   # Checklist-style criteria
│   │   │   └── TaskExport.tsx           # Export buttons (markdown, JSON)
│   │   ├── roadmap/
│   │   │   ├── RoadmapTable.tsx         # Ranked list with scores
│   │   │   ├── ScoreBreakdown.tsx       # Expandable score details
│   │   │   ├── WeightSlider.tsx         # Strategic weight adjuster
│   │   │   └── RoadmapFilters.tsx       # Filter controls
│   │   ├── shared/
│   │   │   ├── JobProgress.tsx          # Async job progress indicator
│   │   │   ├── EmptyState.tsx           # Consistent empty states
│   │   │   ├── QuoteBlock.tsx           # Styled quote with source attribution
│   │   │   ├── StatusBadge.tsx          # Generic status badge
│   │   │   ├── DataTable.tsx            # Reusable TanStack Table wrapper
│   │   │   └── PageHeader.tsx           # Consistent page header
│   │   └── layout/
│   │       ├── PMSidebar.tsx            # PM pipeline sidebar navigation
│   │       └── PMLayout.tsx             # PM-specific layout wrapper
│   ├── canvas/                          # Existing canvas components (RAPTOR workspace)
│   ├── chat/                            # Existing chat components
│   ├── documents/                       # Existing document components
│   ├── layout/                          # Existing layout components
│   └── ui/                              # shadcn/ui primitives (shared)
├── lib/
│   ├── pm/                              # PM-specific lib code
│   │   ├── api.ts                       # PM API client functions
│   │   ├── store.ts                     # PM Zustand store
│   │   ├── types.ts                     # PM TypeScript types
│   │   ├── hooks.ts                     # PM-specific React hooks
│   │   └── constants.ts                 # PM constants (severity, tags, etc.)
│   ├── api.ts                           # Existing shared API client
│   ├── store.ts                         # Existing shared store
│   ├── types.ts                         # Existing shared types
│   └── utils.ts                         # Existing shared utilities
```

---

## State Management

### Zustand Store (PM Pipeline)

```typescript
// lib/pm/store.ts

interface PMStore {
  // Pipeline status
  pipeline: {
    evidenceCount: number;
    problemCount: number;
    clusterCount: number;
    proposalCount: number;
    taskCount: number;
    lastClusteredAt: string | null;
  };

  // Active jobs
  activeJobs: Job[];
  addJob: (job: Job) => void;
  updateJob: (id: string, update: Partial<Job>) => void;
  removeJob: (id: string) => void;

  // Filters (client-side, persisted)
  problemFilters: {
    persona: string | null;
    severity: Severity | null;
    tags: string[];
    sourceType: string | null;
    search: string;
  };
  setFilter: (key: string, value: any) => void;
  clearFilters: () => void;

  // UI state
  selectedProposalId: string | null;
  expandedTaskIds: string[];
  toggleTaskExpanded: (id: string) => void;
}
```

### TanStack Query Keys

```typescript
// lib/pm/hooks.ts

export const pmKeys = {
  evidence: {
    all: ['pm', 'evidence'] as const,
    list: (filters: EvidenceFilters) => ['pm', 'evidence', 'list', filters] as const,
    detail: (id: string) => ['pm', 'evidence', id] as const,
  },
  problems: {
    all: ['pm', 'problems'] as const,
    list: (filters: ProblemFilters) => ['pm', 'problems', 'list', filters] as const,
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
    list: (filters: ProposalFilters) => ['pm', 'proposals', 'list', filters] as const,
    detail: (id: string) => ['pm', 'proposals', id] as const,
  },
  tasks: {
    byProposal: (proposalId: string) => ['pm', 'tasks', proposalId] as const,
  },
  roadmap: {
    ranked: (filters: RoadmapFilters) => ['pm', 'roadmap', filters] as const,
  },
  jobs: {
    detail: (id: string) => ['pm', 'jobs', id] as const,
  },
};
```

### Data Fetching Patterns

```typescript
// Example: useProblems hook
export function useProblems(filters: ProblemFilters) {
  return useQuery({
    queryKey: pmKeys.problems.list(filters),
    queryFn: () => pmApi.getProblems(filters),
    staleTime: 30_000,  // 30 seconds
  });
}

// Example: mutation with optimistic update
export function useApproveProposal() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (id: string) => pmApi.approveProposal(id),
    onSuccess: (_, id) => {
      queryClient.invalidateQueries({ queryKey: pmKeys.proposals.detail(id) });
      queryClient.invalidateQueries({ queryKey: pmKeys.roadmap.ranked({}) });
    },
  });
}

// Example: polling for async job
export function useJobStatus(jobId: string | null) {
  return useQuery({
    queryKey: pmKeys.jobs.detail(jobId!),
    queryFn: () => pmApi.getJobStatus(jobId!),
    enabled: !!jobId,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      return status === 'running' || status === 'pending' ? 2000 : false;
    },
  });
}
```

---

## API Integration Layer

### PM API Client

```typescript
// lib/pm/api.ts

const PM_BASE = '/api/v1';

export const pmApi = {
  // Evidence
  uploadEvidence: (data: EvidenceUpload) =>
    post<Evidence>(`${PM_BASE}/evidence`, data),
  
  getEvidence: (filters?: EvidenceFilters) =>
    get<PaginatedResponse<Evidence>>(`${PM_BASE}/evidence`, filters),
  
  getEvidenceDetail: (id: string) =>
    get<EvidenceDetail>(`${PM_BASE}/evidence/${id}`),
  
  deleteEvidence: (id: string) =>
    del(`${PM_BASE}/evidence/${id}`),

  // Problems
  getProblems: (filters?: ProblemFilters) =>
    get<PaginatedResponse<ProblemMention>>(`${PM_BASE}/problems`, filters),
  
  getProblemDetail: (id: string) =>
    get<ProblemDetail>(`${PM_BASE}/problems/${id}`),
  
  getSimilarProblems: (text: string, limit?: number) =>
    get<SimilarProblem[]>(`${PM_BASE}/problems/similar`, { text, limit }),
  
  getProblemStats: () =>
    get<ProblemStats>(`${PM_BASE}/problems/stats`),

  // Jobs
  triggerExtraction: (evidenceId: string) =>
    post<Job>(`${PM_BASE}/jobs/extract_problems`, { evidence_id: evidenceId }),
  
  triggerClustering: () =>
    post<Job>(`${PM_BASE}/jobs/cluster`),
  
  triggerProposalGeneration: (clusterId: string) =>
    post<Job>(`${PM_BASE}/jobs/generate_proposal`, { cluster_id: clusterId }),
  
  triggerTaskGeneration: (proposalId: string) =>
    post<Job>(`${PM_BASE}/jobs/generate_tasks`, { proposal_id: proposalId }),
  
  getJobStatus: (jobId: string) =>
    get<Job>(`${PM_BASE}/jobs/${jobId}`),

  // Clusters
  getClusters: () =>
    get<Cluster[]>(`${PM_BASE}/clusters`),
  
  getClusterDetail: (id: string) =>
    get<ClusterDetail>(`${PM_BASE}/clusters/${id}`),

  // Proposals
  getProposals: (filters?: ProposalFilters) =>
    get<FeatureProposal[]>(`${PM_BASE}/feature_proposals`, filters),
  
  getProposalDetail: (id: string) =>
    get<FeatureProposalDetail>(`${PM_BASE}/feature_proposals/${id}`),
  
  updateProposal: (id: string, data: Partial<FeatureProposal>) =>
    patch<FeatureProposal>(`${PM_BASE}/feature_proposals/${id}`, data),
  
  approveProposal: (id: string) =>
    post(`${PM_BASE}/feature_proposals/${id}/approve`),
  
  rejectProposal: (id: string) =>
    post(`${PM_BASE}/feature_proposals/${id}/reject`),
  
  regenerateProposal: (id: string) =>
    post<Job>(`${PM_BASE}/feature_proposals/${id}/regenerate`),

  // Tasks
  getTasks: (proposalId: string) =>
    get<TaskTree>(`${PM_BASE}/feature_proposals/${proposalId}/tasks`),
  
  updateTask: (taskId: string, data: Partial<Task>) =>
    patch<Task>(`${PM_BASE}/tasks/${taskId}`, data),

  // Roadmap
  getRoadmap: (filters?: RoadmapFilters) =>
    get<RoadmapResponse>(`${PM_BASE}/roadmap`, filters),
  
  updateWeight: (proposalId: string, weight: number) =>
    patch(`${PM_BASE}/roadmap/${proposalId}/weight`, { strategic_weight: weight }),
};
```

---

## Page-by-Page Specification

### Dashboard (`/pm`)

**Purpose:** Pipeline overview — at-a-glance status of every stage.

**Content:**
- Pipeline indicator (large, prominent)
- Stats cards: evidence count, problem count, cluster count, proposal count
- Recent activity feed (last 10 jobs with status)
- Quick actions: "Upload Evidence", "Run Clustering", "View Roadmap"
- Active jobs list with progress

**Data Requirements:**
- `GET /api/v1/problems/stats`
- `GET /api/v1/clusters` (count)
- `GET /api/v1/feature_proposals` (count by status)
- Recent jobs (last 10)

---

### Evidence List (`/pm/evidence`)

**Purpose:** Browse and manage uploaded source material.

**Content:**
- Table: title, source type, persona, segment, date, chunk count, extraction status
- Sortable by any column
- "Upload Evidence" button (top right)
- Row actions: view detail, re-extract, delete

**Data Requirements:**
- `GET /api/v1/evidence` (paginated)

---

### Evidence Detail (`/pm/evidence/[id]`)

**Purpose:** View source text, chunks, and extracted problems for a single evidence item.

**Content:**
- Header: title, metadata (type, persona, segment, date)
- Tab 1: Raw text (with chunk boundaries highlighted)
- Tab 2: Extracted problems (inline table)
- Tab 3: Processing history (jobs, status, costs)
- Action: "Re-extract Problems"

**Data Requirements:**
- `GET /api/v1/evidence/{id}`

---

### Problems Table (`/pm/problems`)

**Purpose:** Searchable, filterable table of all extracted problem mentions.

**Content:**
- TanStack Table with columns: problem statement, severity, persona, tags, source, date
- Multi-filter: persona dropdown, severity dropdown, tag multi-select, free text search
- Expandable rows: show full quote + source reference
- Row action: "View Similar" opens side panel
- Aggregate stats bar at top: total count, severity distribution

**Data Requirements:**
- `GET /api/v1/problems` (paginated, filtered)
- `GET /api/v1/problems/stats`

---

### Cluster Grid (`/pm/clusters`)

**Purpose:** Visual overview of grouped pain themes.

**Content:**
- Card grid (responsive: 1-3 columns)
- Each card: label, count, severity mini-bar, top quote
- Sorted by member count (biggest pains first)
- "Run Clustering" button (if unclustered problems exist)
- Filter: minimum member count

**Data Requirements:**
- `GET /api/v1/clusters`

---

### Cluster Detail (`/pm/clusters/[id]`)

**Purpose:** Deep dive into a single pain cluster.

**Content:**
- Header: label, summary, member count
- Severity distribution chart (horizontal bars)
- Persona breakdown
- Top quotes (3-5, styled as blockquotes with attribution)
- Full member table (all problem mentions in cluster)
- Proposal status: generated / not generated
- "Generate Feature Proposal" button

**Data Requirements:**
- `GET /api/v1/clusters/{id}`

---

### Proposal List (`/pm/proposals`)

**Purpose:** Browse all feature proposals with status management.

**Content:**
- List with status badges: draft (blue), approved (green), rejected (gray)
- Each item: feature name, one-liner, scope, status, cluster link
- Filter by status
- Sort by: created date, scope, cluster size

**Data Requirements:**
- `GET /api/v1/feature_proposals`

---

### Proposal Detail (`/pm/proposals/[id]`)

**Purpose:** Full feature specification with citation verification.

**Content:**
- Feature name + one-liner (editable)
- User story (editable)
- JTBD framing (editable)
- Rationale with clickable citations
- Success metrics table
- Risks table
- Edge cases list
- Scope estimate
- Source cluster link
- Version history (collapsible)
- Actions: Edit, Regenerate, Approve, Reject, Generate Tasks

**Data Requirements:**
- `GET /api/v1/feature_proposals/{id}`

**Citation Behavior:**
- Citations rendered as superscript links: `[1]`
- Hover: tooltip with quote preview
- Click: navigate to source problem detail or open side panel

---

### Task Tree (`/pm/proposals/[id]/tasks`)

**Purpose:** Dev-ready implementation breakdown.

**Content:**
- Category tabs: Backend, Frontend, Data, QA
- Collapsible tree within each category
- Each task node: title, effort badge, description (expand), acceptance criteria (expand)
- Dependency indicators
- Export: copy as markdown, download as JSON

**Data Requirements:**
- `GET /api/v1/feature_proposals/{id}/tasks`

---

### Roadmap (`/pm/roadmap`)

**Purpose:** Prioritized ranking of all proposals.

**Content:**
- Ranked table: rank, feature name, scope, status, score
- Expandable score breakdown per row
- Strategic weight slider (inline adjustment)
- Filters: persona, segment, tag, status
- Total proposal count, last clustered timestamp

**Data Requirements:**
- `GET /api/v1/roadmap`
- `PATCH /api/v1/roadmap/{proposalId}/weight` (on slider change)

---

## Design System & Patterns

### Color Palette (Severity)

```
Critical:  bg-red-100    text-red-700    border-red-300
High:      bg-orange-100 text-orange-700 border-orange-300
Medium:    bg-yellow-100 text-yellow-700 border-yellow-300
Low:       bg-green-100  text-green-700  border-green-300
```

### Status Badges

```
Draft:      bg-blue-100   text-blue-700
Approved:   bg-green-100  text-green-700
Rejected:   bg-gray-100   text-gray-500
Generating: bg-purple-100 text-purple-700 (pulse animation)
```

### Effort Badges

```
XS:  bg-slate-100  text-slate-600  "XS"
S:   bg-blue-100   text-blue-600   "S"
M:   bg-yellow-100 text-yellow-600 "M"
L:   bg-orange-100 text-orange-600 "L"
XL:  bg-red-100    text-red-600    "XL"
```

### Quote Block Component

```tsx
<QuoteBlock
  text="I spent 3 hours trying to set up my first project"
  source="Customer Interview - Acme Corp PM"
  date="Jan 15, 2026"
  severity="high"
  onClick={() => navigateToSource(evidenceId)}
/>
```

Rendered as:
```
┌─────────────────────────────────────────────────────┐
│  ❝ I spent 3 hours trying to set up my first       │
│    project and still couldn't figure out             │
│    permissions ❞                                     │
│                                                     │
│  — Customer Interview - Acme Corp PM  ·  Jan 2026  │
│                                                 🔴  │
└─────────────────────────────────────────────────────┘
```

### Pipeline Indicator Component

```tsx
<PipelineIndicator
  steps={[
    { label: "Evidence", count: 12, status: "complete" },
    { label: "Problems", count: 47, status: "complete" },
    { label: "Clusters", count: null, status: "running" },
    { label: "Proposals", count: null, status: "pending" },
    { label: "Tasks", count: null, status: "pending" },
    { label: "Roadmap", count: null, status: "pending" },
  ]}
/>
```

### Async Job Pattern

All LLM operations are async. The frontend handles this consistently:

1. **Trigger action** → API returns `job_id`
2. **Show inline progress** → Poll `GET /jobs/{id}` every 2s
3. **On completion** → Invalidate relevant queries, show toast
4. **On failure** → Show error with retry button

```tsx
function useAsyncJob(triggerFn: () => Promise<Job>) {
  const [jobId, setJobId] = useState<string | null>(null);
  const jobStatus = useJobStatus(jobId);
  
  const trigger = async () => {
    const job = await triggerFn();
    setJobId(job.id);
  };

  useEffect(() => {
    if (jobStatus.data?.status === 'completed') {
      toast.success('Processing complete');
      // invalidate relevant queries
    }
    if (jobStatus.data?.status === 'failed') {
      toast.error(`Failed: ${jobStatus.data.error_message}`);
    }
  }, [jobStatus.data?.status]);

  return { trigger, isRunning: jobStatus.data?.status === 'running', progress: jobStatus.data };
}
```

### Responsive Breakpoints

```
Mobile  (< 768px):   Single column, stacked cards, simplified tables
Tablet  (768-1024px): Two column grid, sidebar collapses
Desktop (> 1024px):   Full layout, sidebar + content + optional side panel
```

Priority: **Desktop first** — PMs primarily work on desktop. Mobile is nice-to-have, not critical.

---

## TypeScript Types

### Core Types

```typescript
// lib/pm/types.ts

// ── Evidence ──

type SourceType = 'interview' | 'support_ticket' | 'sales_note' | 'survey' | 'other';

interface Evidence {
  id: string;
  title: string;
  source_type: SourceType;
  persona: string | null;
  segment: string | null;
  source_date: string | null;
  chunk_count: number;
  extraction_status: JobStatus;
  created_at: string;
}

interface EvidenceDetail extends Evidence {
  raw_text: string;
  chunks: EvidenceChunk[];
  problems: ProblemMention[];
  jobs: Job[];
}

interface EvidenceChunk {
  id: string;
  chunk_index: number;
  chunk_text: string;
  start_offset: number;
  end_offset: number;
  token_count: number;
}

interface EvidenceUpload {
  title: string;
  source_type: SourceType;
  persona?: string;
  segment?: string;
  source_date?: string;
  raw_text: string;
}

// ── Problems ──

type Severity = 'critical' | 'high' | 'medium' | 'low';

interface ProblemMention {
  id: string;
  evidence_id: string;
  problem_statement: string;
  persona: string | null;
  segment: string | null;
  severity: Severity;
  quote_text: string;
  tags: string[];
  created_at: string;
}

interface ProblemDetail extends ProblemMention {
  evidence_title: string;
  source_type: SourceType;
  source_date: string | null;
  chunk_text: string;
  similar_problems: SimilarProblem[];
}

interface SimilarProblem {
  problem: ProblemMention;
  similarity_score: number;
}

interface ProblemStats {
  total: number;
  by_severity: Record<Severity, number>;
  by_persona: Record<string, number>;
  by_tag: Record<string, number>;
  by_source_type: Record<SourceType, number>;
}

// ── Clusters ──

interface Cluster {
  id: string;
  label: string;
  summary: string | null;
  member_count: number;
  avg_severity: number;
  severity_distribution: Record<Severity, number>;
  top_quotes: ClusterQuote[];
  has_proposal: boolean;
  created_at: string;
}

interface ClusterDetail extends Cluster {
  members: ProblemMention[];
  persona_distribution: Record<string, number>;
  proposal: FeatureProposal | null;
}

interface ClusterQuote {
  text: string;
  source: string;
  severity: Severity;
  problem_id: string;
}

// ── Proposals ──

type ProposalStatus = 'draft' | 'approved' | 'rejected' | 'archived';
type ScopeEstimate = 'S' | 'M' | 'L' | 'XL';

interface FeatureProposal {
  id: string;
  cluster_id: string;
  feature_name: string;
  one_liner: string;
  user_story: string | null;
  jtbd_framing: string | null;
  rationale: string;
  success_metrics: SuccessMetric[];
  risks: Risk[];
  edge_cases: string[];
  scope_estimate: ScopeEstimate;
  status: ProposalStatus;
  created_at: string;
  updated_at: string;
}

interface FeatureProposalDetail extends FeatureProposal {
  citations: Citation[];
  versions: ProposalVersion[];
  cluster: Cluster;
  tasks_generated: boolean;
  priority_score: PriorityScore | null;
}

interface SuccessMetric {
  metric: string;
  target: string;
  reasoning: string;
}

interface Risk {
  risk: string;
  mitigation: string;
  severity: 'high' | 'medium' | 'low';
}

interface Citation {
  id: string;
  problem_id: string;
  citation_context: string;
  quote_text: string;
  evidence_title: string;
}

interface ProposalVersion {
  id: string;
  version_number: number;
  change_reason: string | null;
  created_at: string;
}

// ── Tasks ──

type TaskCategory = 'backend' | 'frontend' | 'data' | 'qa';
type TaskEffort = 'XS' | 'S' | 'M' | 'L' | 'XL';

interface Task {
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
  subtasks: Task[];  // nested children
}

interface TaskTree {
  proposal_id: string;
  feature_name: string;
  backend: Task[];
  frontend: Task[];
  data: Task[];
  qa: Task[];
  total_tasks: number;
}

// ── Roadmap ──

interface PriorityScore {
  frequency_score: number;
  severity_score: number;
  strategic_weight: number;
  effort_estimate: number;
  final_score: number;
  score_breakdown: ScoreBreakdown;
}

interface ScoreBreakdown {
  formula: string;
  frequency: { value: number; explanation: string };
  severity: { value: number; distribution: Record<Severity, number> };
  weight: { value: number; reason: string };
  effort: { value: number; scope: ScopeEstimate };
  final: number;
}

interface RoadmapEntry {
  rank: number;
  proposal: FeatureProposal;
  score: PriorityScore;
  cluster_label: string;
}

interface RoadmapResponse {
  proposals: RoadmapEntry[];
  total_proposals: number;
  last_clustered_at: string | null;
}

// ── Jobs ──

type JobType = 'extract_problems' | 'cluster' | 'generate_proposal' | 'generate_tasks';
type JobStatus = 'pending' | 'running' | 'completed' | 'failed';

interface Job {
  id: string;
  job_type: JobType;
  status: JobStatus;
  error_message: string | null;
  token_usage: TokenUsage | null;
  started_at: string | null;
  completed_at: string | null;
  created_at: string;
}

interface TokenUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_cost: number;
}

// ── Filters ──

interface EvidenceFilters {
  source_type?: SourceType;
  persona?: string;
  segment?: string;
  page?: number;
  per_page?: number;
}

interface ProblemFilters {
  persona?: string;
  severity?: Severity;
  tags?: string[];
  source_type?: SourceType;
  search?: string;
  page?: number;
  per_page?: number;
}

interface ProposalFilters {
  status?: ProposalStatus;
}

interface RoadmapFilters {
  persona?: string;
  segment?: string;
  tag?: string;
  status?: ProposalStatus;
}

interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  per_page: number;
  total_pages: number;
}
```

---

## Development Phases

### Phase 1: Evidence + Problems UI (Weeks 1-3)

**Goal:** Upload transcripts and view extracted problems.

| Week | Deliverable |
|------|------------|
| 1 | PM layout shell: sidebar, pipeline indicator, routing structure |
| 1 | Evidence upload modal: drag-drop, paste text, metadata form |
| 2 | Evidence list page: table with sorting, status indicators |
| 2 | Evidence detail page: raw text view, chunk boundaries |
| 3 | Problems table: filterable, sortable, expandable rows |
| 3 | Job progress component: polling, toasts, status badges |

**Exit Criteria:**
- Upload a transcript → see it in list → extraction job runs → problems appear in table
- Filter problems by severity, persona, tags
- Expand problem rows to see full quotes with source attribution

**Dependencies:** Backend Phase 1 (Evidence + Extraction services)

### Phase 2: Clusters + Proposals (Weeks 4-6)

**Goal:** See grouped pain themes and generate feature proposals.

| Week | Deliverable |
|------|------------|
| 4 | Cluster grid page: cards with mini severity chart, sorted by count |
| 4 | Cluster detail page: summary, quotes list, severity/persona charts |
| 5 | Proposal detail page: full spec with citation links |
| 5 | Citation interaction: hover preview, click to navigate to source |
| 6 | Proposal editing: inline editable fields, approve/reject actions |
| 6 | Proposal list page: status badges, filters |

**Exit Criteria:**
- Click "Run Clustering" → clusters appear with labeled cards
- Click cluster → see detail with quotes → "Generate Proposal" → proposal appears
- Citation links in proposals navigate to source problems
- Approve/reject proposals updates their status

**Dependencies:** Backend Phase 2 (Clustering + Proposal services)

### Phase 3: Tasks + Roadmap (Weeks 7-9)

**Goal:** Complete pipeline from proposals to prioritized roadmap.

| Week | Deliverable |
|------|------------|
| 7 | Task tree page: category tabs, collapsible hierarchy |
| 7 | Acceptance criteria display, effort badges, dependency indicators |
| 8 | Task export: copy as markdown, download as JSON |
| 8 | Roadmap page: ranked table with score breakdown |
| 9 | Score breakdown component: expandable, formula display |
| 9 | Strategic weight slider: inline adjustment, recalculation |

**Exit Criteria:**
- Generate tasks from proposal → see structured tree with acceptance criteria
- Export task tree as markdown (ready for Linear/GitHub import)
- Roadmap shows ranked proposals with explainable scores
- Adjusting strategic weight recalculates ranking

**Dependencies:** Backend Phase 3 (Task Tree + Prioritization services)

### Phase 4: Polish + Dashboard (Weeks 10-12)

| Week | Deliverable |
|------|------------|
| 10 | Dashboard page: pipeline overview, stats cards, recent activity |
| 10 | Settings page: API key config, prompt version selection |
| 11 | Usage page: cost tracking, job history, token usage charts |
| 11 | Similarity panel: side drawer on problems page |
| 12 | Responsive refinements, loading skeletons, error boundaries |
| 12 | End-to-end UX polish, keyboard navigation, accessibility audit |

**Exit Criteria:**
- Dashboard gives at-a-glance pipeline status
- Cost tracking shows LLM usage per job and cumulative totals
- All pages handle loading, empty, and error states gracefully
- Full pipeline walkthrough works smoothly end-to-end

---

## Integration with Existing Nexus Frontend

The PM pipeline lives under `/pm/*` routes and uses a dedicated layout. It coexists with the existing Nexus workspace (canvas, chat, documents) without conflicts:

```
/                    → Landing page (existing)
/workspace           → RAPTOR canvas workspace (existing)
/documents           → Document management (existing)
/pm                  → PM pipeline dashboard (NEW)
/pm/evidence         → Evidence management (NEW)
/pm/problems         → Problem analysis (NEW)
/pm/clusters         → Pain clustering (NEW)
/pm/proposals        → Feature proposals (NEW)
/pm/roadmap          → Prioritized roadmap (NEW)
```

**Shared infrastructure:**
- shadcn/ui components (already installed)
- Tailwind config (already configured)
- Root layout + global styles
- Utility functions (`lib/utils.ts`)

**Separate infrastructure:**
- PM-specific API client (`lib/pm/api.ts`)
- PM-specific Zustand store (`lib/pm/store.ts`)
- PM-specific types (`lib/pm/types.ts`)
- PM-specific layout + sidebar (`components/pm/layout/`)

This separation keeps the PM pipeline self-contained while leveraging the existing UI foundation.

---

## Summary

The frontend is a **six-stage pipeline visualizer + editor**:

```
Upload → Browse Problems → Explore Clusters → Review Proposals → View Tasks → Rank Roadmap
```

Every stage:
- **Shows provenance** (trace any claim back to a quote)
- **Handles async** (LLM jobs with polling + progress)
- **Supports filtering** (persona, severity, tags, source type)
- **Favors density** (tables and cards over dashboards)
- **Enables light editing** (modify AI output, don't author from scratch)

Build Phase 1 in parallel with backend Phase 1. Ship evidence upload + problem extraction first. Validate the UX. Then proceed.
