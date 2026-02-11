# Nexus AR Vision — Glasses-First UX Concept

> This document describes the AR-first interaction model for Nexus — how the AI research workspace would feel on lightweight AR glasses (e.g., Meta Orion, future Meta glasses). It covers what we can test now, what requires future hardware, and the next milestones.

---

## 1. The AR-First Concept

### "Your Knowledge, In Your World"

Imagine wearing lightweight AR glasses at your desk. Instead of staring at a monitor, your research workspace *is* the room:

- **Document groups** float as translucent card clusters around your desk
- A **query bar** hovers at the edge of your vision, always accessible
- **Retrieval results** fan out in front of you like a hand of cards
- **Agent progress** streams along a timeline ribbon at the side of your gaze
- **Citations** glow softly on the source documents when referenced

The key insight: AR doesn't replace your desk — it **augments** it. Your physical papers, monitors, and whiteboards coexist with Nexus overlays.

---

## 2. Interaction Model

### Primary Input: Voice + Gaze + Hand Gestures

| Input | Role | Example |
|-------|------|---------|
| **Voice** | Primary query input | "Hey Nexus, what did the RAPTOR paper say about clustering?" |
| **Gaze** | Selection and focus | Look at a document card to highlight it |
| **Hand pinch** | Confirm / select | Pinch thumb+index to select a highlighted card |
| **Hand swipe** | Navigate / dismiss | Swipe left to dismiss a result, swipe right to bookmark |
| **Hand grab** | Spatial arrange | Grab and move document groups to reorganize |

### Secondary Input: Controller (Fallback)

When available (Quest 2/3 controllers), standard pointer ray + trigger remains the primary input. The voice + gaze model is the AR-native target.

### Interaction Priority

```
AR Glasses:    Voice → Gaze + Pinch → Swipe/Grab
Quest 3 + PT:  Voice → Hand Tracking → Controller fallback
Quest 2 (VR):  Controller (pointer ray + trigger) → Gaze fallback
Rift DK:       Controller (pointer ray + trigger)
```

---

## 3. Spatial UI Design — "Cards & Ribbons"

### Core UI Elements

#### 3.1 Command Bar (Always Visible)
- A subtle horizontal bar at the lower edge of your field of view
- Shows: microphone status, current query, quick actions
- Taps into **peripheral awareness** — visible but not distracting
- In AR: semi-transparent, ~40% opacity
- In VR: slightly more opaque, ~70%

#### 3.2 Result Cards (On Demand)
- Each retrieval result is a floating card (~A5 size in AR, larger in VR)
- Cards fan out in a gentle arc in front of the user
- Each card shows: source document name, excerpt, relevance score, layer badge
- Gaze at a card for 0.5s to expand it; pinch to select
- Cards have a subtle depth effect (parallax) to feel "physical"

#### 3.3 Agent Timeline Ribbon
- A vertical ribbon along the right side of your gaze
- Events stack chronologically: thinking → tool call → result → output
- Each event is a small card with an icon and brief text
- Active events pulse subtly
- In AR: the ribbon is world-locked to a spatial anchor (e.g., your desk edge)

#### 3.4 Document Clusters
- Groups of documents rendered as stacked card piles
- Positioned around your workspace using spatial anchors
- Each cluster shows: group name, document count, index status
- Grab a cluster to reposition it (spatial memory)
- Tap a cluster to expand and see individual documents

#### 3.5 Citation Connections
- When viewing a result, thin glowing lines connect to the source document clusters
- Shows "where this fact came from" spatially
- Inspired by mind-map connection lines in the 2D frontend

---

## 4. Spatial Anchors & World Locking

### AR Glasses (Future)
- UI elements anchored to real-world surfaces (desk, wall, whiteboard)
- Document clusters stay where you put them across sessions
- Uses ARCore/ARKit or Meta Spatial Anchors API
- "Your research room" has persistent spatial layout

### VR (Now)
- UI elements anchored to virtual room positions
- Panels at comfortable reading distance (1.5m)
- Document clusters arranged in a virtual semicircle
- Layout resets between sessions (persistence is a future milestone)

---

## 5. What We Can Test Now

### ✅ Testable on Quest 2 / Rift DK (VR)

| Feature | Status | Notes |
|---------|--------|-------|
| Spatial result cards in arc layout | ✅ Implemented | `RetrievalResultsController` |
| Agent streaming timeline | ✅ Implemented | `AgentWorkspaceController` |
| Document grid browse | ✅ Implemented | `DocsLibraryController` |
| Text input via system keyboard | ✅ Implemented | `TextInputPanel` + Quest keyboard |
| Teleport locomotion | ✅ Implemented | `TeleportLocomotion` |
| Continuous locomotion + snap turn | ✅ Implemented | `ContinuousLocomotion` |
| Pointer ray selection | ✅ Via XR Interaction Toolkit | Standard XR interactors |
| Gaze pointer (fallback) | ✅ Implemented | `GazePointer` with dwell select |
| AR simulation mode | ✅ Implemented | `ARSimulatedOverlay` (dark BG, transparent panels) |
| Floating HUD | ✅ Implemented | `FloatingHUD` with AR/VR opacity modes |
| Demo walkthrough script | ✅ Implemented | `DemoWalkthrough` — press D key |
| Mock API mode | ✅ Implemented | `NexusApiClient` with `MockDataProvider` |

### 🔮 Requires Future Hardware / SDK

| Feature | Dependency | Timeline |
|---------|-----------|----------|
| Voice input ("Hey Nexus...") | Meta Voice SDK or Whisper integration | Next milestone |
| Hand tracking pinch-to-select | Quest hand tracking (partial now) | Near-term |
| Real passthrough AR | Quest 3 passthrough API | Near-term |
| World-locked spatial anchors | Meta Spatial Anchors SDK | Medium-term |
| Glasses-native rendering | Meta Orion / AR glasses SDK | Long-term |
| Persistent spatial layouts | Spatial anchor persistence API | Medium-term |
| Citation glow connections | Custom shader + spatial graph | Next milestone |

---

## 6. AR Comfort Guidelines

### Panel Placement
- **Distance**: 0.6–1.2m for primary content (AR), 1.2–2.0m for VR
- **Angle**: Slightly below eye level (5–15° downward) to match natural reading posture
- **Width**: Max 40° of horizontal FOV for primary content to avoid neck strain
- **Depth layers**: Max 3 depth planes to avoid vergence-accommodation conflict

### Text Legibility
- **Minimum font size**: 1.5mm at the target distance (approximately 24pt at 1m)
- **Contrast**: Light text on dark semi-transparent backgrounds
- **Line length**: Max 60 characters per line at reading distance

### Motion
- **No elements attached rigidly to the head** — the HUD uses lazy follow with smoothing
- **No moving backgrounds** — environment is static in VR mode
- **Fade transitions** between scenes (0.3s default)
- **Teleport** as default locomotion to eliminate vection sickness

---

## 7. Next Milestones

### Milestone 1: Voice Integration (Near-Term)
- Integrate Meta Voice SDK or local Whisper model for voice queries
- Add voice-activated command bar: "Hey Nexus, search for..."
- Evaluate latency and accuracy in VR headset environment
- **Estimated effort**: 2–3 weeks

### Milestone 2: Hand Tracking Polish (Near-Term)
- Implement pinch-to-select using Unity XR Hands
- Add grab gesture for repositioning panels
- Add swipe gesture for navigation
- Test on Quest 2 (supported) and future Quest 3
- **Estimated effort**: 2 weeks

### Milestone 3: Quest 3 Passthrough AR (Near-Term)
- Enable Meta Quest 3 color passthrough mode
- Adjust UI opacity and colors for real-world blending
- Test panel placement relative to real desk surfaces
- Basic spatial anchor placement (no persistence yet)
- **Estimated effort**: 1–2 weeks (once Quest 3 hardware available)

### Milestone 4: Spatial Anchors & Persistence (Medium-Term)
- Implement Meta Spatial Anchors for world-locking document clusters
- Save/restore spatial layout between sessions
- Allow users to "pin" panels to walls, desks, whiteboards
- **Estimated effort**: 3–4 weeks

### Milestone 5: Citation Graph Visualization (Medium-Term)
- Render glowing connection lines between results and source documents
- Animate connections when retrieval results arrive
- Spatial mind-map view of the entire knowledge base
- Custom shader for glow effect
- **Estimated effort**: 2–3 weeks

### Milestone 6: AR Glasses Prototype (Long-Term)
- Port to Meta Orion SDK (when available)
- Optimize rendering for glasses-class hardware (lower resolution, power budget)
- Full voice + gaze + pinch interaction model
- Persistent spatial workspace across sessions
- **Estimated effort**: Depends on SDK availability

---

## 8. Design Principles

1. **Ambient, not intrusive** — AR UI should feel like a helpful layer, not a wall of screens
2. **Peripheral first** — Most information lives at the edges; the center of vision stays clear
3. **Spatial memory** — Users remember *where* they put things; persistent layout matters
4. **Progressive disclosure** — Start with a simple command bar, expand on demand
5. **Graceful degradation** — The experience works with just controllers (VR) and gets better with hands + voice (AR)
6. **Comfort always wins** — If a feature causes discomfort, it gets cut. No exceptions.

---

## 9. Comparison: VR vs AR Experience

| Aspect | VR (Quest 2 / Rift DK) | AR (Glasses, Future) |
|--------|------------------------|---------------------|
| Environment | Virtual dark room | Your actual room |
| Panel opacity | 85% (opaque) | 40–60% (see-through) |
| Panel distance | 1.5m | 0.8m |
| Primary input | Controller pointer ray | Voice + gaze + pinch |
| Locomotion | Teleport / smooth | Walk physically |
| Document clusters | Virtual semicircle | Anchored to desk/walls |
| Text input | System keyboard overlay | Voice dictation |
| Session length | 30–60 min (comfort limit) | Hours (lightweight glasses) |

---

*This document is a living spec. Updated as we learn from testing and as new hardware becomes available.*
