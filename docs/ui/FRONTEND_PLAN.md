# Nexus Frontend Plan

## Primary User Personas & Jobs-to-be-Done

| Persona | Description | Primary Jobs |
|---------|-------------|-------------|
| **Researcher** | Academic or professional who manages large document collections | Upload & organize documents, query across collections, get cited answers |
| **Engineer** | Technical user building hardware/software projects | Ask specialist AI agents questions, delegate physical tasks, iterate on designs |
| **Knowledge Worker** | Anyone who needs to synthesize information from many sources | Build a visual knowledge map, get summaries, learn through dialogue |

## Information Architecture

```
/                   → Landing page (value prop, feature highlights, CTA)
/workspace          → Infinite canvas (drag-drop document groups, mind-map view)
/documents          → Document management (upload, status, chunking state)
/retrieval          → Query UI (search, filters, results with citations)
/agents             → Agent library (view personas, create/edit, run history)
/settings           → Preferences (API keys, model config, theme)
```

### Navigation
- **Sidebar** (collapsible): Canvas, Documents, Retrieval, Agents + AI Chat toggle
- **Global search** (⌘K): Search documents, groups, or ask questions
- **Settings**: Accessible from sidebar footer

## Core Flows

1. **Documents → Retrieval → Agent Run → Results**
   - Upload documents → auto-parse, chunk, embed
   - Organize on canvas into groups with connections
   - Query via Retrieval page or chat panel
   - RAPTOR tree selects appropriate detail layer
   - AI persona responds with citations + source links
   - Human-in-the-loop: agent may delegate physical tasks back to user

2. **Agent Orchestration**
   - Select persona from agent library
   - Persona uses tools (query_group, search_all, request_human_task)
   - Results stream back with sources and actions
   - Run history tracks previous interactions

## Visual Direction

- **Tone**: Premium, professional, dark-mode-first — like Linear meets Notion
- **Typography**: Inter font family, strong hierarchy (display → heading → body → caption)
- **Color**: Deep navy/slate background, blue-purple gradient accents, persona-specific colors
- **Motion**: Framer Motion for page transitions, spring-based micro-interactions, skeleton loaders
- **Components**: Consistent spacing scale, rounded corners (0.75rem default), glass-morphism for overlays
- **States**: Every component has loading/empty/error/success states with helpful copy

## Component Principles

1. **Composition over configuration** — primitives compose into complex patterns
2. **Design tokens as single source of truth** — `lib/tokens.ts` drives all styling decisions
3. **Accessible by default** — semantic HTML, ARIA labels, keyboard navigation
4. **Responsive** — mobile-first layout, sidebar collapses, grid adapts
5. **Merge-friendly** — UI is isolated from backend logic via typed API contracts
