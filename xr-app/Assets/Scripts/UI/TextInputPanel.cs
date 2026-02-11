using UnityEngine;
using UnityEngine.UI;
using TMPro;
using NexusXR.Core;

namespace NexusXR.UI
{
    /// <summary>
    /// A world-space text input panel for XR environments.
    /// Since native keyboard input is limited in VR, this provides:
    /// - An on-screen virtual keyboard (stub, relies on system keyboard on Quest)
    /// - A text display area showing current input
    /// - Submit and clear buttons
    ///
    /// On Quest 2, Unity's TMP_InputField triggers the system keyboard overlay.
    /// On Rift, the desktop keyboard is used directly.
    /// </summary>
    public class TextInputPanel : MonoBehaviour
    {
        [Header("UI References")]
        public TMP_InputField inputField;
        public Button submitButton;
        public Button clearButton;
        public TextMeshProUGUI placeholderText;

        [Header("Configuration")]
        [Tooltip("Placeholder text shown when input is empty")]
        public string placeholder = "Type your question here...";

        [Tooltip("Maximum character limit")]
        public int maxCharacters = 500;

        /// <summary>Callback invoked when text is submitted.</summary>
        public System.Action<string> OnSubmit;

        private void Start()
        {
            if (inputField != null)
            {
                inputField.characterLimit = maxCharacters;
                inputField.onSubmit.AddListener(HandleSubmit);
            }

            if (submitButton != null)
                submitButton.onClick.AddListener(() => HandleSubmit(GetText()));

            if (clearButton != null)
                clearButton.onClick.AddListener(Clear);

            if (placeholderText != null)
                placeholderText.text = placeholder;
        }

        /// <summary>Get the current input text.</summary>
        public string GetText()
        {
            return inputField != null ? inputField.text.Trim() : "";
        }

        /// <summary>Set the input text programmatically.</summary>
        public void SetText(string text)
        {
            if (inputField != null)
                inputField.text = text;
        }

        /// <summary>Clear the input field.</summary>
        public void Clear()
        {
            if (inputField != null)
                inputField.text = "";
        }

        /// <summary>Focus the input field (triggers system keyboard on Quest).</summary>
        public void Focus()
        {
            if (inputField != null)
                inputField.ActivateInputField();
        }

        private void HandleSubmit(string text)
        {
            if (string.IsNullOrWhiteSpace(text)) return;
            Debug.Log($"[TextInputPanel] Submitted: {text}");
            OnSubmit?.Invoke(text.Trim());
        }
    }
}
