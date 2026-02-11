using UnityEngine;
using TMPro;

namespace NexusXR.UI
{
    /// <summary>
    /// A world-space UI panel that follows XR comfort guidelines.
    /// Panels are positioned at a comfortable reading distance, slightly
    /// below eye level, and can optionally follow the user's gaze.
    ///
    /// Use for primary content areas (answer panels, document lists, etc).
    /// </summary>
    public class SpatialPanel : MonoBehaviour
    {
        [Header("Placement")]
        [Tooltip("Distance from the user's head (meters)")]
        [Range(0.5f, 3f)]
        public float distance = 1.5f;

        [Tooltip("Vertical offset from eye level (meters). Negative = below eye level.")]
        [Range(-0.8f, 0.8f)]
        public float heightOffset = -0.1f;

        [Tooltip("Horizontal offset (meters). 0 = centered.")]
        public float horizontalOffset = 0f;

        [Header("Behavior")]
        [Tooltip("If true, panel follows the user's head rotation (lazy follow)")]
        public bool followGaze;

        [Tooltip("How quickly the panel follows gaze (0 = instant, higher = slower)")]
        [Range(0f, 10f)]
        public float followSmoothing = 3f;

        [Tooltip("If true, panel always faces the user")]
        public bool billboard = true;

        [Header("Appearance")]
        [Tooltip("Background panel opacity")]
        [Range(0f, 1f)]
        public float opacity = 0.85f;

        [Tooltip("Panel corner radius (visual only, set in material)")]
        public float cornerRadius = 0.02f;

        private Transform _cameraTransform;
        private Vector3 _targetPosition;
        private bool _initialized;

        private void Start()
        {
            _cameraTransform = Camera.main?.transform;
            if (_cameraTransform == null)
            {
                Debug.LogWarning("[SpatialPanel] No main camera found");
                return;
            }

            PositionPanel();
            _initialized = true;
        }

        private void LateUpdate()
        {
            if (!_initialized || _cameraTransform == null) return;

            if (followGaze)
            {
                UpdateTargetPosition();
                transform.position = Vector3.Lerp(
                    transform.position,
                    _targetPosition,
                    Time.deltaTime * (10f / Mathf.Max(1f, followSmoothing))
                );
            }

            if (billboard)
            {
                // Face the camera while keeping upright
                Vector3 lookDir = _cameraTransform.position - transform.position;
                lookDir.y = 0;
                if (lookDir.sqrMagnitude > 0.001f)
                {
                    transform.rotation = Quaternion.LookRotation(-lookDir, Vector3.up);
                }
            }
        }

        /// <summary>Position the panel relative to the camera.</summary>
        public void PositionPanel()
        {
            if (_cameraTransform == null) return;

            UpdateTargetPosition();
            transform.position = _targetPosition;

            if (billboard)
            {
                Vector3 lookDir = _cameraTransform.position - transform.position;
                lookDir.y = 0;
                if (lookDir.sqrMagnitude > 0.001f)
                {
                    transform.rotation = Quaternion.LookRotation(-lookDir, Vector3.up);
                }
            }
        }

        private void UpdateTargetPosition()
        {
            Vector3 forward = _cameraTransform.forward;
            forward.y = 0;
            forward.Normalize();

            Vector3 right = _cameraTransform.right;
            right.y = 0;
            right.Normalize();

            _targetPosition = _cameraTransform.position
                + forward * distance
                + Vector3.up * heightOffset
                + right * horizontalOffset;
        }

        /// <summary>Update panel opacity (applies to CanvasGroup if present).</summary>
        public void SetOpacity(float newOpacity)
        {
            opacity = Mathf.Clamp01(newOpacity);
            var canvasGroup = GetComponent<CanvasGroup>();
            if (canvasGroup != null)
            {
                canvasGroup.alpha = opacity;
            }
        }
    }
}
