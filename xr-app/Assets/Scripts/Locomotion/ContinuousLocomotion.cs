using UnityEngine;
using UnityEngine.XR;
using System.Collections.Generic;

namespace NexusXR.Locomotion
{
    /// <summary>
    /// Smooth continuous locomotion as an alternative to teleport.
    /// Uses the left thumbstick for movement and right thumbstick for snap-turn.
    /// Includes comfort options (vignette, speed limits) to reduce motion sickness.
    ///
    /// Recommended for seated Rift DK usage. For Quest 2 standalone,
    /// teleport is generally preferred.
    /// </summary>
    public class ContinuousLocomotion : MonoBehaviour
    {
        [Header("Movement")]
        [Tooltip("Movement speed (meters/second)")]
        [Range(0.5f, 5f)]
        public float moveSpeed = 2f;

        [Tooltip("XR node for movement input")]
        public XRNode moveHand = XRNode.LeftHand;

        [Tooltip("Thumbstick deadzone")]
        [Range(0.05f, 0.5f)]
        public float deadzone = 0.15f;

        [Header("Snap Turn")]
        [Tooltip("Enable snap turn on right stick")]
        public bool enableSnapTurn = true;

        [Tooltip("Snap turn angle (degrees)")]
        [Range(15f, 90f)]
        public float snapAngle = 45f;

        [Tooltip("Snap turn cooldown (seconds)")]
        [Range(0.1f, 0.5f)]
        public float snapCooldown = 0.25f;

        [Header("Comfort")]
        [Tooltip("Enable comfort vignette during movement")]
        public bool comfortVignette = true;

        [Tooltip("Vignette intensity during movement")]
        [Range(0f, 1f)]
        public float vignetteIntensity = 0.3f;

        [Header("References")]
        [Tooltip("The XR camera rig/origin to move")]
        public Transform xrRig;

        private InputDevice _moveDevice;
        private InputDevice _turnDevice;
        private float _snapTimer;
        private bool _isMoving;

        private void Update()
        {
            UpdateDevices();
            HandleMovement();
            HandleSnapTurn();
        }

        private void UpdateDevices()
        {
            if (!_moveDevice.isValid)
            {
                var devices = new List<InputDevice>();
                InputDevices.GetDevicesAtXRNode(moveHand, devices);
                if (devices.Count > 0)
                    _moveDevice = devices[0];
            }
            if (!_turnDevice.isValid)
            {
                var devices = new List<InputDevice>();
                InputDevices.GetDevicesAtXRNode(XRNode.RightHand, devices);
                if (devices.Count > 0)
                    _turnDevice = devices[0];
            }
        }

        private void HandleMovement()
        {
            if (xrRig == null || !_moveDevice.isValid) return;

            _moveDevice.TryGetFeatureValue(CommonUsages.primary2DAxis, out Vector2 stick);

            if (stick.magnitude < deadzone)
            {
                _isMoving = false;
                return;
            }

            _isMoving = true;

            // Move relative to head direction (horizontal only)
            Transform head = Camera.main?.transform;
            if (head == null) return;

            Vector3 forward = head.forward;
            forward.y = 0;
            forward.Normalize();

            Vector3 right = head.right;
            right.y = 0;
            right.Normalize();

            Vector3 movement = (forward * stick.y + right * stick.x) *
                               moveSpeed * Time.deltaTime;

            xrRig.position += movement;
        }

        private void HandleSnapTurn()
        {
            if (!enableSnapTurn || !_turnDevice.isValid || xrRig == null)
                return;

            _snapTimer -= Time.deltaTime;

            _turnDevice.TryGetFeatureValue(CommonUsages.primary2DAxis, out Vector2 stick);

            if (_snapTimer <= 0f && Mathf.Abs(stick.x) > 0.7f)
            {
                float direction = stick.x > 0 ? 1f : -1f;

                // Rotate around the head position (not rig center)
                Vector3 headPos = Camera.main?.transform.position ?? xrRig.position;
                xrRig.RotateAround(headPos, Vector3.up, snapAngle * direction);

                _snapTimer = snapCooldown;
            }
        }

        /// <summary>Whether the player is currently moving (for vignette).</summary>
        public bool IsMoving => _isMoving;
    }
}
