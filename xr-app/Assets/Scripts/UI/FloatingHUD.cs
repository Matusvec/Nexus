using UnityEngine;
using UnityEngine.UI;
using TMPro;
using NexusXR.Core;
using NexusXR.API;

namespace NexusXR.UI
{
    /// <summary>
    /// A floating heads-up display that persists across scenes.
    /// Shows essential information: current scene, mock/live mode status,
    /// time, and quick-navigation breadcrumbs.
    ///
    /// In AR simulation mode, this renders as a translucent overlay
    /// to mimic what a glasses-style AR HUD would look like.
    /// </summary>
    public class FloatingHUD : MonoBehaviour
    {
        [Header("UI References")]
        public TextMeshProUGUI sceneLabel;
        public TextMeshProUGUI modeLabel;
        public TextMeshProUGUI breadcrumbText;
        public Button homeButton;

        [Header("Placement")]
        [Tooltip("Offset from the camera's forward direction")]
        public Vector3 positionOffset = new Vector3(0f, 0.35f, 1.5f);

        [Tooltip("Scale of the HUD")]
        public float hudScale = 0.001f;

        [Header("AR Simulation")]
        [Tooltip("Canvas group for controlling HUD transparency")]
        public CanvasGroup canvasGroup;

        [Tooltip("Opacity in normal VR mode")]
        [Range(0.3f, 1f)]
        public float vrOpacity = 0.9f;

        [Tooltip("Opacity in AR simulation mode (more transparent)")]
        [Range(0.1f, 0.7f)]
        public float arOpacity = 0.4f;

        private Transform _cameraTransform;

        private void Start()
        {
            _cameraTransform = Camera.main?.transform;

            if (homeButton != null)
                homeButton.onClick.AddListener(() =>
                    SceneNavigator.Instance?.GoHome());

            UpdateDisplay();
        }

        private void LateUpdate()
        {
            if (_cameraTransform == null) return;

            // Keep HUD anchored relative to the camera
            transform.position = _cameraTransform.position
                + _cameraTransform.forward * positionOffset.z
                + _cameraTransform.up * positionOffset.y
                + _cameraTransform.right * positionOffset.x;

            transform.rotation = Quaternion.LookRotation(
                transform.position - _cameraTransform.position,
                Vector3.up
            );
        }

        /// <summary>Refresh all HUD labels.</summary>
        public void UpdateDisplay()
        {
            var manager = NexusXRManager.Instance;
            bool isAR = manager != null && manager.arSimulationMode;

            // Scene label
            if (sceneLabel != null)
            {
                string scene = SceneNavigator.Instance?.CurrentScene ?? "Home";
                sceneLabel.text = scene;
            }

            // Mode label
            if (modeLabel != null)
            {
                bool mock = manager?.useMockMode ?? true;
                string modeText = mock ? "MOCK" : "LIVE";
                string arText = isAR ? " | AR" : "";
                modeLabel.text = $"{modeText}{arText}";
            }

            // Opacity
            if (canvasGroup != null)
            {
                canvasGroup.alpha = isAR ? arOpacity : vrOpacity;
            }
        }
    }
}
