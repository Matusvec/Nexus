# Nexus XR — Scene Setup Guide

This guide explains how to create the four Unity scenes from the provided C# scripts. Each scene follows the same pattern: an XR Origin, a World Space Canvas, and the corresponding scene controller.

## Common Setup (All Scenes)

Every scene needs:

1. **XR Origin (XR Rig)**
   - GameObject → XR → XR Origin (Action-based)
   - This creates the camera, controller objects, and interaction manager
   - Add `XRInputManager` component to the XR Origin

2. **Event System**
   - GameObject → UI → Event System
   - Ensure `XR UI Input Module` is added (replaces Standalone Input Module)

3. **Environment** (optional)
   - A simple ground plane at Y=0
   - Ambient lighting: slight blue tint for VR, near-black for AR mode
   - Skybox: dark gradient or solid dark color

---

## Scene 1: HomeCommand

**The main entry point of the app.**

### Objects to Create

```
HomeCommand (scene root)
├── XR Origin (Action-based)
│   ├── Camera Offset
│   │   ├── Main Camera
│   │   ├── LeftHand Controller
│   │   └── RightHand Controller
│   └── XRInputManager (script)
├── [NexusXRManager] (empty GO, DontDestroyOnLoad)
│   ├── NexusXRManager.cs
│   └── SceneNavigator.cs
├── [DemoWalkthrough] (empty GO)
│   └── DemoWalkthrough.cs
├── Canvas_Home (World Space Canvas)
│   ├── Title_Text (TextMeshPro - "NEXUS")
│   ├── InputField (TMP Input Field)
│   ├── Buttons_Container
│   │   ├── Btn_Search (Button + TextMeshPro "Search")
│   │   ├── Btn_Agent (Button + TextMeshPro "Run Agent")
│   │   └── Btn_Docs (Button + TextMeshPro "Documents")
│   ├── Status_Text (TextMeshPro)
│   └── HomeCommandController.cs ← wire up references
├── FloatingHUD (World Space Canvas)
│   └── FloatingHUD.cs
├── ARSimulatedOverlay (empty GO)
│   └── ARSimulatedOverlay.cs
├── Teleport_Locomotion (empty GO)
│   └── TeleportLocomotion.cs
└── Ground_Plane (3D Plane)
```

### Canvas Settings
- Render Mode: **World Space**
- Position: (0, 1.2, 1.5)
- Scale: (0.001, 0.001, 0.001)
- Width: 800, Height: 600
- Add `SpatialPanel` component for comfort placement

### Wiring
- Drag `InputField` → HomeCommandController.inputField
- Drag `Btn_Search` → HomeCommandController.searchButton
- Drag `Btn_Agent` → HomeCommandController.agentButton
- Drag `Btn_Docs` → HomeCommandController.docsButton
- Drag `Status_Text` → HomeCommandController.statusText
- Drag `Title_Text` → HomeCommandController.titleText

---

## Scene 2: RetrievalResults

**Displays query results as spatial cards with citations.**

### Objects to Create

```
RetrievalResults (scene root)
├── XR Origin (Action-based)
├── Canvas_Results (World Space Canvas)
│   ├── Query_Text (TextMeshPro)
│   ├── Answer_Panel
│   │   └── Answer_Text (TextMeshPro)
│   ├── Results_Container (empty GO)
│   ├── Why_Panel
│   │   └── Why_Text (TextMeshPro)
│   ├── Loading_Indicator (spinner or text)
│   ├── Btn_Back (Button)
│   └── RetrievalResultsController.cs
├── FloatingHUD
└── Ground_Plane
```

### Result Card Prefab

Create a prefab `ResultCard.prefab`:
```
ResultCard
├── Background (Quad with semi-transparent material)
├── Title_Text (TextMeshPro - document name)
├── Body_Text (TextMeshPro - chunk content)
├── Footer_Text (TextMeshPro - layer, score, chunk ID)
└── SpatialCard.cs component
```

Save as `Assets/Prefabs/UI/ResultCard.prefab`

---

## Scene 3: AgentWorkspace

**Shows streaming agent progress with a timeline.**

### Objects to Create

```
AgentWorkspace (scene root)
├── XR Origin (Action-based)
├── Canvas_Agent (World Space Canvas)
│   ├── Task_Text (TextMeshPro)
│   ├── Timeline_Container (empty GO, left side)
│   ├── Output_Panel (right side)
│   │   └── Output_Text (TextMeshPro)
│   ├── Status_Text (TextMeshPro)
│   ├── Btn_Run (Button + Run_Text)
│   ├── Btn_Back (Button)
│   └── AgentWorkspaceController.cs
├── FloatingHUD
└── Ground_Plane
```

### Event Card Prefab

Create `EventCard.prefab`:
```
EventCard
├── Background (Quad)
├── Content_Text (TextMeshPro)
└── SpatialCard.cs (optional)
```

Save as `Assets/Prefabs/UI/EventCard.prefab`

---

## Scene 4: DocsLibrary

**Browse and manage documents.**

### Objects to Create

```
DocsLibrary (scene root)
├── XR Origin (Action-based)
├── Canvas_Docs (World Space Canvas)
│   ├── Stats_Text (TextMeshPro)
│   ├── Docs_Container (empty GO)
│   ├── Btn_Add (Button)
│   ├── Btn_Refresh (Button)
│   ├── Btn_Back (Button)
│   ├── Loading_Indicator
│   └── DocsLibraryController.cs
├── FloatingHUD
└── Ground_Plane
```

### Document Card Prefab

Create `DocCard.prefab`:
```
DocCard
├── Background (Quad)
├── Content_Text (TextMeshPro)
├── Btn_Delete (Button)
└── SpatialCard.cs
```

Save as `Assets/Prefabs/UI/DocCard.prefab`

---

## Build Settings

Add all four scenes to **File → Build Settings** in this order:

| Index | Scene |
|-------|-------|
| 0 | HomeCommand |
| 1 | RetrievalResults |
| 2 | AgentWorkspace |
| 3 | DocsLibrary |

---

## Testing Checklist

After setting up all scenes, verify:

- [ ] HomeCommand loads and shows title + input + buttons
- [ ] Clicking "Search" navigates to RetrievalResults
- [ ] RetrievalResults displays mock query answer + source cards
- [ ] Clicking "Back" returns to HomeCommand
- [ ] Clicking "Run Agent" navigates to AgentWorkspace
- [ ] AgentWorkspace shows streaming timeline events
- [ ] Clicking "Documents" navigates to DocsLibrary
- [ ] DocsLibrary shows list of mock documents with status
- [ ] Demo walkthrough (press D) cycles through all scenes
- [ ] Teleport locomotion works with left stick
- [ ] Pointer ray selects UI elements
- [ ] FloatingHUD displays in all scenes
