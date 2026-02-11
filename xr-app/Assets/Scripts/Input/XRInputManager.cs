using UnityEngine;
using UnityEngine.XR;
using System.Collections.Generic;

namespace NexusXR.Input
{
    /// <summary>
    /// Centralized XR input manager that abstracts controller input
    /// across Quest 2 (touch controllers) and Rift DK (also touch).
    /// Provides a unified input API for the rest of the application.
    ///
    /// Relies on Unity's XR Interaction Toolkit for low-level input.
    /// This manager adds application-level convenience methods.
    /// </summary>
    public class XRInputManager : MonoBehaviour
    {
        public static XRInputManager Instance { get; private set; }

        [Header("Input Configuration")]
        [Tooltip("Dominant hand (used for primary pointer)")]
        public XRNode dominantHand = XRNode.RightHand;

        [Tooltip("Enable hand tracking input (Quest 2 only)")]
        public bool enableHandTracking = true;

        [Tooltip("Enable gaze-based pointing as fallback")]
        public bool enableGazePointing;

        // ── State ──────────────────────────────────────────────────
        private InputDevice _leftController;
        private InputDevice _rightController;
        private bool _leftTrigger;
        private bool _rightTrigger;
        private bool _leftGrip;
        private bool _rightGrip;
        private Vector2 _leftStick;
        private Vector2 _rightStick;
        private bool _primaryButton; // A on Quest
        private bool _secondaryButton; // B on Quest

        // ── Public API ─────────────────────────────────────────────

        /// <summary>True if the dominant hand trigger is pressed.</summary>
        public bool IsTriggerPressed => dominantHand == XRNode.RightHand
            ? _rightTrigger : _leftTrigger;

        /// <summary>True if the dominant hand grip is pressed.</summary>
        public bool IsGripPressed => dominantHand == XRNode.RightHand
            ? _rightGrip : _leftGrip;

        /// <summary>Left thumbstick value.</summary>
        public Vector2 LeftStick => _leftStick;

        /// <summary>Right thumbstick value.</summary>
        public Vector2 RightStick => _rightStick;

        /// <summary>Primary action button (A on Quest).</summary>
        public bool PrimaryButton => _primaryButton;

        /// <summary>Secondary action button (B on Quest).</summary>
        public bool SecondaryButton => _secondaryButton;

        private void Awake()
        {
            if (Instance != null && Instance != this)
            {
                Destroy(gameObject);
                return;
            }
            Instance = this;
        }

        private void Update()
        {
            UpdateControllerState(XRNode.LeftHand, ref _leftController,
                ref _leftTrigger, ref _leftGrip, ref _leftStick);
            UpdateControllerState(XRNode.RightHand, ref _rightController,
                ref _rightTrigger, ref _rightGrip, ref _rightStick);

            // Primary/secondary buttons (right hand A/B)
            if (_rightController.isValid)
            {
                _rightController.TryGetFeatureValue(
                    CommonUsages.primaryButton, out _primaryButton);
                _rightController.TryGetFeatureValue(
                    CommonUsages.secondaryButton, out _secondaryButton);
            }
        }

        private void UpdateControllerState(
            XRNode node, ref InputDevice device,
            ref bool trigger, ref bool grip, ref Vector2 stick)
        {
            if (!device.isValid)
            {
                var devices = new List<InputDevice>();
                InputDevices.GetDevicesAtXRNode(node, devices);
                if (devices.Count > 0)
                    device = devices[0];
            }

            if (device.isValid)
            {
                device.TryGetFeatureValue(CommonUsages.triggerButton, out trigger);
                device.TryGetFeatureValue(CommonUsages.gripButton, out grip);
                device.TryGetFeatureValue(CommonUsages.primary2DAxis, out stick);
            }
        }

        /// <summary>Vibrate the specified controller.</summary>
        public void Haptic(XRNode hand, float amplitude = 0.3f, float duration = 0.1f)
        {
            var devices = new List<InputDevice>();
            InputDevices.GetDevicesAtXRNode(hand, devices);
            foreach (var device in devices)
            {
                device.SendHapticImpulse(0, amplitude, duration);
            }
        }
    }
}
