using UnityEngine;

namespace NexusXR.UI
{
    /// <summary>
    /// Renders an AR-simulated overlay that mimics a passthrough
    /// glasses-style experience when running in VR headsets.
    ///
    /// When enabled, this adjusts the environment to simulate how
    /// content would appear on AR glasses:
    /// - Reduces background to dark/transparent
    /// - Makes UI panels semi-transparent with glow edges
    /// - Adjusts panel distances to closer (AR comfort range ~0.5-1m)
    /// - Adds subtle world-locked anchor indicators
    ///
    /// This is a design exploration tool - real AR passthrough requires
    /// hardware support (Quest 3+ passthrough API or dedicated AR glasses).
    /// </summary>
    public class ARSimulatedOverlay : MonoBehaviour
    {
        [Header("AR Simulation Settings")]
        [Tooltip("Enable AR simulation mode")]
        public bool arModeEnabled;

        [Tooltip("Background color when AR mode is on (simulates passthrough)")]
        public Color arBackgroundColor = new Color(0.02f, 0.02f, 0.04f, 1f);

        [Tooltip("Normal VR background color")]
        public Color vrBackgroundColor = new Color(0.05f, 0.05f, 0.1f, 1f);

        [Tooltip("Panel opacity in AR mode")]
        [Range(0.2f, 0.8f)]
        public float arPanelOpacity = 0.5f;

        [Tooltip("Panel distance in AR mode (closer than VR)")]
        [Range(0.4f, 1.5f)]
        public float arPanelDistance = 0.8f;

        [Header("Glow Effect")]
        [Tooltip("Edge glow color for panels in AR mode")]
        public Color glowColor = new Color(0.2f, 0.6f, 1f, 0.3f);

        [Tooltip("Glow intensity")]
        [Range(0f, 2f)]
        public float glowIntensity = 0.8f;

        private Camera _mainCamera;
        private Color _originalBackgroundColor;

        private void Start()
        {
            _mainCamera = Camera.main;
            if (_mainCamera != null)
            {
                _originalBackgroundColor = _mainCamera.backgroundColor;
            }

            if (arModeEnabled)
            {
                EnableARMode();
            }
        }

        /// <summary>Toggle AR simulation mode.</summary>
        public void SetARMode(bool enabled)
        {
            arModeEnabled = enabled;
            if (enabled)
                EnableARMode();
            else
                DisableARMode();
        }

        private void EnableARMode()
        {
            Debug.Log("[AROverlay] Enabling AR simulation mode");

            // Set camera background to near-black (simulating passthrough)
            if (_mainCamera != null)
            {
                _mainCamera.backgroundColor = arBackgroundColor;
                _mainCamera.clearFlags = CameraClearFlags.SolidColor;
            }

            // Adjust all spatial panels in the scene
            var panels = FindObjectsOfType<SpatialPanel>();
            foreach (var panel in panels)
            {
                panel.distance = arPanelDistance;
                panel.SetOpacity(arPanelOpacity);
                panel.PositionPanel();
            }
        }

        private void DisableARMode()
        {
            Debug.Log("[AROverlay] Disabling AR simulation mode");

            if (_mainCamera != null)
            {
                _mainCamera.backgroundColor = _originalBackgroundColor;
            }

            var panels = FindObjectsOfType<SpatialPanel>();
            foreach (var panel in panels)
            {
                panel.distance = 1.5f;
                panel.SetOpacity(0.85f);
                panel.PositionPanel();
            }
        }
    }
}
