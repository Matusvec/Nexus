# Nexus XR — AR/VR Frontend Prototype

An alternative AR/VR frontend for the Nexus AI Research Workspace, built with Unity + OpenXR. This prototype demonstrates core product flows in immersive VR, with an AR-first design philosophy for future glasses-style devices.

## Why Unity + OpenXR?

1. **Cross-device support** — OpenXR provides a single API targeting Quest 2, Rift DK, and future AR glasses with minimal code changes.
2. **Fastest path to prototype** — Unity's XR Interaction Toolkit provides battle-tested pointer rays, teleportation, and UI interaction out of the box.
3. **Ecosystem maturity** — Rich asset ecosystem, world-space UI (TextMeshPro + Canvas), and proven VR performance on mobile chipsets (Quest 2 XR2).
4. **Hand tracking ready** — Unity XR Hands package supports Quest 2 hand tracking, which is the foundation for the AR glasses interaction model.
5. **Team accessibility** — C# + Unity has the widest VR developer community, making future contributions easier.

## Target Devices

| Device | Type | Status | Notes |
|--------|------|--------|-------|
| **Meta Quest 2** | Standalone VR | ✅ Primary target | Build Android APK, deploy via ADB or Quest Link |
| **Oculus Rift DK** | PC VR | ✅ Supported | Run as Windows Standalone build via Oculus runtime |
| **Meta Quest 3** | Standalone VR + Passthrough | 🔮 Future | Full color passthrough enables real AR simulation |
| **AR Glasses (Meta Orion, etc.)** | AR | 🔮 Future | See [AR_VISION.md](AR_VISION.md) for the UX spec |

## Project Structure

```
xr-app/
├── Assets/
│   ├── Scripts/
│   │   ├── Core/                  # App lifecycle, scene navigation
│   │   │   ├── NexusXRManager.cs  # Root manager, config, singleton
│   │   │   └── SceneNavigator.cs  # Scene transitions with fade
│   │   ├── API/                   # Backend communication
│   │   │   ├── NexusApiClient.cs  # HTTP client with mock toggle
│   │   │   ├── MockDataProvider.cs # Realistic mock data
│   │   │   └── Models/            # Data classes (Query, Document, Agent)
│   │   ├── Scenes/                # Scene-specific controllers
│   │   │   ├── HomeCommandController.cs
│   │   │   ├── RetrievalResultsController.cs
│   │   │   ├── AgentWorkspaceController.cs
│   │   │   └── DocsLibraryController.cs
│   │   ├── UI/                    # Spatial UI components
│   │   │   ├── SpatialPanel.cs    # World-space comfort panel
│   │   │   ├── SpatialCard.cs     # Interactive result card
│   │   │   ├── TextInputPanel.cs  # VR text input
│   │   │   ├── FloatingHUD.cs     # Persistent heads-up display
│   │   │   └── ARSimulatedOverlay.cs  # AR mode simulation
│   │   ├── Input/                 # XR input abstraction
│   │   │   ├── XRInputManager.cs  # Controller input state
│   │   │   └── GazePointer.cs     # Gaze-based pointing (AR mode)
│   │   ├── Locomotion/            # VR movement
│   │   │   ├── TeleportLocomotion.cs    # Teleport (comfort default)
│   │   │   └── ContinuousLocomotion.cs  # Smooth move + snap turn
│   │   └── Demo/
│   │       └── DemoWalkthrough.cs # Deterministic demo sequence
│   ├── Scenes/                    # Unity scene files (created in editor)
│   ├── Prefabs/                   # Reusable prefabs
│   ├── Materials/                 # Shaders and materials
│   └── Resources/MockData/        # Mock JSON data files
├── Packages/
│   └── manifest.json              # Unity package dependencies
└── ProjectSettings/
    └── ProjectSettings.asset      # Build and XR settings
```

## Setup Instructions

### Prerequisites

- **Unity 2022.3 LTS** or newer (2023.2+ recommended for latest XR features)
- **Unity Hub** installed
- **Android Build Support** module (for Quest 2 builds)
- **Windows Build Support** module (for Rift DK builds)

### 1. Open the Project

```bash
# Clone the repo (if not already)
git clone https://github.com/Matusvec/Nexus.git
cd Nexus

# Open xr-app in Unity Hub
# Unity Hub → Open → select the xr-app/ folder
```

Unity will import packages automatically on first open:
- XR Interaction Toolkit 2.5.4
- OpenXR Plugin 1.9.1
- XR Hands 1.4.1
- Input System 1.7.0
- TextMeshPro 3.0.6
- Universal Render Pipeline 14.0.11

### 2. Configure XR

After Unity finishes importing:

1. **Edit → Project Settings → XR Plug-in Management**
   - **PC tab**: Enable "OpenXR", add "Oculus Touch Controller Profile"
   - **Android tab**: Enable "OpenXR", add "Oculus Touch Controller Profile" + "Meta Quest Feature"

2. **Edit → Project Settings → XR Plug-in Management → OpenXR**
   - Set Render Mode: Multi-pass
   - Enable interaction profiles for your target device

3. **Edit → Project Settings → Player**
   - Android: Set Minimum API Level to 29, Target API Level to 32
   - Set Company Name and Product Name as desired

### 3. Create Scenes (First Time)

The C# scripts are provided; Unity scene files must be created in the editor:

1. **Create scene files** in `Assets/Scenes/`:
   - `HomeCommand.unity`
   - `RetrievalResults.unity`
   - `AgentWorkspace.unity`
   - `DocsLibrary.unity`

2. **For each scene**, set up:
   - Add an XR Origin (from XR Interaction Toolkit samples)
   - Add a World Space Canvas with the appropriate controller script
   - Add the `NexusXRManager` prefab (only in HomeCommand, mark DontDestroyOnLoad)
   - Add the `SceneNavigator` prefab (only in HomeCommand)

3. **HomeCommand scene setup**:
   - Create a World Space Canvas at position (0, 1.2, 1.5)
   - Add `TMP_InputField` for query input
   - Add 3 buttons: Search, Run Agent, Docs
   - Add `TextMeshProUGUI` for title ("NEXUS") and status
   - Attach `HomeCommandController.cs` and wire up references
   - Add `DemoWalkthrough.cs` to a GameObject

4. **Add all scenes to Build Settings** (File → Build Settings → Add Open Scenes)

### 4. Mock Mode (Default)

The app runs in **mock mode** by default — no backend needed:

- `NexusXRManager.useMockMode = true` (Inspector toggle)
- All API calls return realistic demo data
- Simulated network latency (configurable, default 0.5s)

To connect to a live backend:
- Set `useMockMode = false` on the NexusXRManager
- Set `backendUrl` to your Nexus FastAPI server (default: `http://localhost:8000`)
- Ensure the backend implements endpoints from `frontend/API_SPECIFICATION.md`

## Build & Run

### Quest 2 (Android)

```bash
# 1. Switch platform: File → Build Settings → Android → Switch Platform
# 2. Connect Quest 2 via USB, enable Developer Mode
# 3. Build and Run (Ctrl+B or File → Build and Run)
```

**Or via ADB:**
```bash
# After building the APK:
adb install -r NexusXR.apk
# Launch from Quest 2's Unknown Sources menu
```

**Via Quest Link (wireless/tethered):**
- Enable Quest Link on the headset
- Press Play in Unity Editor — the app runs on the headset

### Rift DK (Windows Standalone)

```bash
# 1. Switch platform: File → Build Settings → PC, Mac & Linux Standalone
# 2. Ensure Oculus Desktop app is running
# 3. Press Play in Editor (or Build and Run)
```

The app will launch in VR through the Oculus runtime.

### Editor Testing (No Headset)

Press Play in Unity Editor. Use:
- **WASD** keys for movement
- **Mouse** for look/point
- **Left click** for select
- **D key** to start the demo walkthrough
- XR Device Simulator (from XR Interaction Toolkit samples) for controller simulation

## Demo Walkthrough

A deterministic demo sequence is included. To run it:

1. Open the `HomeCommand` scene
2. Press Play
3. Press **D** on keyboard (or **B button** on Quest controller)
4. The demo automatically walks through:
   - Home → Search query → Retrieval results with citations
   - Home → Agent task → Streaming progress with tool calls
   - Home → Document library → Browse documents

Check the Console window for step-by-step logs prefixed with `[Demo]`.

## API Contracts

The XR app consumes the same API as the web frontend. Key endpoints:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/query` | POST | Retrieval query with sources |
| `/documents` | GET | List all documents |
| `/documents/{id}` | DELETE | Remove a document |
| `/agent/run` | POST | Start agent task (SSE stream) |
| `/stats` | GET | Database statistics |

See `frontend/API_SPECIFICATION.md` for full contract details. All endpoints are mocked in `MockDataProvider.cs`.

## Architecture Decisions

### Comfort-First VR Design
- Panels placed at 1.5m distance, slightly below eye level (-0.1m)
- Teleport locomotion as default (continuous locomotion optional)
- Smooth scene transitions with fade-to-black
- 72 FPS target (Quest 2 native)
- No moving UI elements attached to head (HUD follows with lag)

### AR Simulation Mode
The `ARSimulatedOverlay` component can be toggled to preview how content would appear on AR glasses:
- Near-black background (simulating passthrough)
- Semi-transparent panels (50% opacity)
- Closer panel distance (0.8m vs 1.5m)
- Reduced HUD opacity

This is a design tool, not real AR — see [AR_VISION.md](AR_VISION.md) for the full AR vision.

### Clean Integration
- **No backend modifications** — consumes the existing Nexus API
- **Isolated directory** — all XR code lives in `xr-app/`, no entanglement with `frontend/` or `backend/`
- **Mock mode toggle** — works standalone or connected to backend
