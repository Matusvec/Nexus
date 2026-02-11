using UnityEngine;
using UnityEngine.XR;
using System.Collections.Generic;

namespace NexusXR.Locomotion
{
    /// <summary>
    /// Teleport-based locomotion for comfortable VR movement.
    /// Uses the left thumbstick to aim a teleport arc, then releases
    /// to teleport to the target position. This is the default
    /// locomotion mode as it minimizes motion sickness.
    ///
    /// Compatible with both Quest 2 and Rift DK controllers.
    /// </summary>
    public class TeleportLocomotion : MonoBehaviour
    {
        [Header("Configuration")]
        [Tooltip("XR node used for teleport aiming (usually left hand)")]
        public XRNode aimHand = XRNode.LeftHand;

        [Tooltip("Maximum teleport distance")]
        [Range(1f, 20f)]
        public float maxDistance = 10f;

        [Tooltip("Teleport arc height")]
        [Range(0.5f, 5f)]
        public float arcHeight = 2f;

        [Tooltip("Layer mask for valid teleport surfaces")]
        public LayerMask teleportMask = -1;

        [Tooltip("Thumbstick deadzone")]
        [Range(0.05f, 0.5f)]
        public float deadzone = 0.3f;

        [Header("Visual Feedback")]
        [Tooltip("Line renderer for the teleport arc")]
        public LineRenderer arcLine;

        [Tooltip("Teleport target indicator")]
        public GameObject targetIndicator;

        [Tooltip("Number of arc segments")]
        public int arcSegments = 30;

        [Tooltip("Valid target color")]
        public Color validColor = new Color(0.2f, 0.8f, 1f, 0.8f);

        [Tooltip("Invalid target color")]
        public Color invalidColor = new Color(1f, 0.3f, 0.2f, 0.5f);

        [Header("References")]
        [Tooltip("The XR camera rig/origin to move")]
        public Transform xrRig;

        private bool _isAiming;
        private Vector3 _targetPosition;
        private bool _validTarget;
        private InputDevice _aimDevice;

        private void Start()
        {
            if (arcLine != null)
            {
                arcLine.positionCount = arcSegments;
                arcLine.enabled = false;
            }
            if (targetIndicator != null)
                targetIndicator.SetActive(false);
        }

        private void Update()
        {
            UpdateDevice();
            if (!_aimDevice.isValid) return;

            _aimDevice.TryGetFeatureValue(CommonUsages.primary2DAxis, out Vector2 stick);

            if (stick.magnitude > deadzone)
            {
                _isAiming = true;
                UpdateArc();
            }
            else if (_isAiming)
            {
                // Release - perform teleport if valid
                if (_validTarget)
                {
                    Teleport(_targetPosition);
                }
                _isAiming = false;
                HideVisuals();
            }
        }

        private void UpdateDevice()
        {
            if (!_aimDevice.isValid)
            {
                var devices = new List<InputDevice>();
                InputDevices.GetDevicesAtXRNode(aimHand, devices);
                if (devices.Count > 0)
                    _aimDevice = devices[0];
            }
        }

        private void UpdateArc()
        {
            if (arcLine == null) return;

            Transform aimTransform = Camera.main?.transform;
            if (aimTransform == null) return;

            Vector3 startPos = aimTransform.position;
            Vector3 forward = aimTransform.forward;
            forward.y = 0;
            forward.Normalize();

            // Calculate arc points
            Vector3[] points = new Vector3[arcSegments];
            _validTarget = false;

            for (int i = 0; i < arcSegments; i++)
            {
                float t = (float)i / (arcSegments - 1);
                float x = t * maxDistance;
                float y = arcHeight * 4f * t * (1f - t); // Parabola

                points[i] = startPos + forward * x + Vector3.up * y;

                // Raycast to check for ground
                if (i > 0)
                {
                    Vector3 direction = points[i] - points[i - 1];
                    if (Physics.Raycast(points[i - 1], direction,
                        out RaycastHit hit, direction.magnitude, teleportMask))
                    {
                        _targetPosition = hit.point;
                        _validTarget = true;

                        // Truncate arc at hit point
                        points[i] = hit.point;
                        arcLine.positionCount = i + 1;
                        break;
                    }
                }
            }

            arcLine.SetPositions(points);
            arcLine.enabled = true;

            // Color
            Color color = _validTarget ? validColor : invalidColor;
            arcLine.startColor = color;
            arcLine.endColor = color;

            // Target indicator
            if (targetIndicator != null)
            {
                targetIndicator.SetActive(_validTarget);
                if (_validTarget)
                    targetIndicator.transform.position = _targetPosition;
            }
        }

        private void Teleport(Vector3 targetPosition)
        {
            if (xrRig == null) return;

            // Calculate offset (keep head position relative to rig)
            Vector3 headOffset = Camera.main.transform.position - xrRig.position;
            headOffset.y = 0;

            xrRig.position = targetPosition - headOffset;
            Debug.Log($"[Teleport] Moved to: {targetPosition}");
        }

        private void HideVisuals()
        {
            if (arcLine != null)
            {
                arcLine.enabled = false;
                arcLine.positionCount = arcSegments;
            }
            if (targetIndicator != null)
                targetIndicator.SetActive(false);
        }
    }
}
