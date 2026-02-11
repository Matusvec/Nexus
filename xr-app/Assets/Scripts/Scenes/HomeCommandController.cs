using UnityEngine;
using UnityEngine.UI;
using TMPro;
using NexusXR.Core;

namespace NexusXR.Scenes
{
    /// <summary>
    /// Controller for the Home / Command scene.
    /// Provides the main entry point: text input for queries, and quick-action
    /// buttons for Search, Run Agent, and Documents.
    ///
    /// Layout: A floating panel at comfortable VR reading distance with
    /// a centered input field and three action buttons below.
    /// </summary>
    public class HomeCommandController : MonoBehaviour
    {
        [Header("UI References")]
        [Tooltip("Text input field for queries/commands")]
        public TMP_InputField inputField;

        [Tooltip("Search (retrieval) button")]
        public Button searchButton;

        [Tooltip("Run Agent button")]
        public Button agentButton;

        [Tooltip("Documents library button")]
        public Button docsButton;

        [Tooltip("Status text (shows mock mode, connection status)")]
        public TextMeshProUGUI statusText;

        [Tooltip("Welcome/title text")]
        public TextMeshProUGUI titleText;

        private void Start()
        {
            // Wire up button clicks
            if (searchButton != null)
                searchButton.onClick.AddListener(OnSearchClicked);
            if (agentButton != null)
                agentButton.onClick.AddListener(OnAgentClicked);
            if (docsButton != null)
                docsButton.onClick.AddListener(OnDocsClicked);

            UpdateStatusDisplay();
            SetDefaultInputText();

            Debug.Log("[HomeCommand] Scene ready");
        }

        private void SetDefaultInputText()
        {
            if (inputField != null)
            {
                inputField.text = "";
                inputField.placeholder.GetComponent<TextMeshProUGUI>().text =
                    "Ask a question or describe a task...";
            }

            if (titleText != null)
            {
                titleText.text = "NEXUS";
            }
        }

        private void UpdateStatusDisplay()
        {
            if (statusText == null) return;

            var manager = NexusXRManager.Instance;
            if (manager != null)
            {
                string mode = manager.useMockMode ? "MOCK MODE" : "LIVE";
                string ar = manager.arSimulationMode ? " | AR SIM" : "";
                statusText.text = $"v{manager.appVersion} | {mode}{ar}";
            }
            else
            {
                statusText.text = "Initializing...";
            }
        }

        // ── Button Handlers ────────────────────────────────────────

        private void OnSearchClicked()
        {
            string query = GetInputText();
            if (string.IsNullOrWhiteSpace(query))
            {
                query = "What is RAPTOR retrieval and how does it work?";
                Debug.Log("[HomeCommand] Using demo query");
            }

            Debug.Log($"[HomeCommand] Search: {query}");
            SceneNavigator.Instance?.GoToRetrievalResults(query);
        }

        private void OnAgentClicked()
        {
            string task = GetInputText();
            if (string.IsNullOrWhiteSpace(task))
            {
                task = "Summarize all documents and identify key themes";
                Debug.Log("[HomeCommand] Using demo task");
            }

            Debug.Log($"[HomeCommand] Agent task: {task}");
            SceneNavigator.Instance?.GoToAgentWorkspace(task);
        }

        private void OnDocsClicked()
        {
            Debug.Log("[HomeCommand] Opening docs library");
            SceneNavigator.Instance?.GoToDocsLibrary();
        }

        private string GetInputText()
        {
            return inputField != null ? inputField.text.Trim() : "";
        }
    }
}
