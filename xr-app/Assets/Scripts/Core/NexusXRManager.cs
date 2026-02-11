using UnityEngine;
using NexusXR.API;

namespace NexusXR.Core
{
    /// <summary>
    /// Root manager for the Nexus XR application.
    /// Handles global initialization, configuration, and scene orchestration.
    /// Attach to a persistent GameObject in the boot scene.
    /// </summary>
    public class NexusXRManager : MonoBehaviour
    {
        // ── Singleton ──────────────────────────────────────────────
        public static NexusXRManager Instance { get; private set; }

        // ── Configuration ──────────────────────────────────────────
        [Header("App Configuration")]
        [Tooltip("Application version string")]
        public string appVersion = "0.1.0-prototype";

        [Tooltip("Enable AR-simulated overlay mode (passthrough-style HUD)")]
        public bool arSimulationMode = false;

        [Header("Backend")]
        [Tooltip("Use mock data instead of live backend")]
        public bool useMockMode = true;

        [Tooltip("Backend API base URL")]
        public string backendUrl = "http://localhost:8000";

        [Header("UI Preferences")]
        [Tooltip("Distance of floating panels from the user (meters)")]
        [Range(0.8f, 3f)]
        public float panelDistance = 1.5f;

        [Tooltip("Height offset for panels relative to eye level (meters)")]
        [Range(-0.5f, 0.5f)]
        public float panelHeightOffset = -0.1f;

        [Tooltip("Opacity of spatial panels (0 = transparent, 1 = opaque)")]
        [Range(0.3f, 1f)]
        public float panelOpacity = 0.85f;

        // ── State ──────────────────────────────────────────────────
        public bool IsInitialized { get; private set; }

        private void Awake()
        {
            if (Instance != null && Instance != this)
            {
                Destroy(gameObject);
                return;
            }
            Instance = this;
            DontDestroyOnLoad(gameObject);

            Initialize();
        }

        private void Initialize()
        {
            Debug.Log($"[NexusXR] Initializing v{appVersion}");

            // Configure API client
            var apiClient = NexusApiClient.Instance;
            apiClient.useMockMode = useMockMode;
            apiClient.baseUrl = backendUrl;

            Debug.Log($"[NexusXR] Mock mode: {useMockMode}");
            Debug.Log($"[NexusXR] Backend URL: {backendUrl}");
            Debug.Log($"[NexusXR] AR simulation: {arSimulationMode}");

            // Configure XR comfort settings
            Application.targetFrameRate = 72; // Quest 2 default
            QualitySettings.vSyncCount = 0;

            IsInitialized = true;
            Debug.Log("[NexusXR] Initialization complete");
        }

        /// <summary>
        /// Toggle between mock and live API mode at runtime.
        /// </summary>
        public void SetMockMode(bool enabled)
        {
            useMockMode = enabled;
            NexusApiClient.Instance.useMockMode = enabled;
            Debug.Log($"[NexusXR] Mock mode changed to: {enabled}");
        }

        /// <summary>
        /// Toggle AR simulation overlay mode at runtime.
        /// </summary>
        public void SetARSimulationMode(bool enabled)
        {
            arSimulationMode = enabled;
            Debug.Log($"[NexusXR] AR simulation mode: {enabled}");
        }
    }
}
