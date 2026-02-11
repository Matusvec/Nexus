using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;
using NexusXR.API;
using NexusXR.API.Models;
using NexusXR.Core;

namespace NexusXR.Scenes
{
    /// <summary>
    /// Controller for the Agent Workspace scene.
    /// Shows a streaming task timeline of agent events including thinking,
    /// tool calls, results, and generated tokens. Supports starting runs
    /// and viewing streaming progress in real time.
    ///
    /// Layout: A vertical timeline on the left side with event cards
    /// stacking downward. The right side shows the accumulated output.
    /// </summary>
    public class AgentWorkspaceController : MonoBehaviour
    {
        [Header("UI References")]
        [Tooltip("Text displaying the current task")]
        public TextMeshProUGUI taskText;

        [Tooltip("Container for timeline event cards")]
        public Transform timelineContainer;

        [Tooltip("Prefab for a timeline event card")]
        public GameObject eventCardPrefab;

        [Tooltip("Accumulated output text panel")]
        public TextMeshProUGUI outputText;

        [Tooltip("Status indicator text")]
        public TextMeshProUGUI statusText;

        [Tooltip("Start/stop run button")]
        public Button runButton;

        [Tooltip("Back button")]
        public Button backButton;

        [Tooltip("Run button label")]
        public TextMeshProUGUI runButtonText;

        [Header("Layout")]
        [Tooltip("Vertical spacing between timeline events")]
        public float eventSpacing = 0.12f;

        private string _currentTask;
        private bool _isRunning;
        private Coroutine _runCoroutine;
        private readonly List<GameObject> _eventCards = new List<GameObject>();
        private string _accumulatedOutput = "";

        private void Start()
        {
            if (backButton != null)
                backButton.onClick.AddListener(OnBackClicked);
            if (runButton != null)
                runButton.onClick.AddListener(OnRunClicked);

            // Get the task passed from the Home scene
            _currentTask = SceneNavigator.ConsumePendingAgentTask();
            if (string.IsNullOrEmpty(_currentTask))
            {
                _currentTask = "Summarize all documents and identify key themes";
                Debug.Log("[AgentWorkspace] No pending task, using default");
            }

            if (taskText != null)
                taskText.text = _currentTask;

            UpdateStatus("Ready");
            UpdateRunButton();

            // Auto-start the run
            StartAgentRun();
        }

        private void StartAgentRun()
        {
            if (_isRunning) return;

            _isRunning = true;
            _accumulatedOutput = "";
            ClearTimeline();
            UpdateStatus("Running...");
            UpdateRunButton();

            var request = new AgentRunRequest
            {
                task = _currentTask
            };

            _runCoroutine = StartCoroutine(NexusApiClient.Instance.RunAgent(
                request,
                OnAgentEvent,
                OnAgentError
            ));
        }

        private void StopAgentRun()
        {
            if (!_isRunning) return;

            if (_runCoroutine != null)
            {
                StopCoroutine(_runCoroutine);
                _runCoroutine = null;
            }

            _isRunning = false;
            UpdateStatus("Stopped");
            UpdateRunButton();
        }

        private void OnAgentEvent(AgentEvent evt)
        {
            AddTimelineEvent(evt);

            switch (evt.type)
            {
                case "token":
                    _accumulatedOutput += evt.content;
                    if (outputText != null)
                        outputText.text = _accumulatedOutput;
                    break;

                case "done":
                    _isRunning = false;
                    UpdateStatus("Completed");
                    UpdateRunButton();
                    break;

                case "error":
                    _isRunning = false;
                    UpdateStatus($"Error: {evt.content}");
                    UpdateRunButton();
                    break;
            }
        }

        private void OnAgentError(string error)
        {
            _isRunning = false;
            UpdateStatus($"Error: {error}");
            UpdateRunButton();
            Debug.LogError($"[AgentWorkspace] Agent error: {error}");
        }

        private void AddTimelineEvent(AgentEvent evt)
        {
            if (timelineContainer == null) return;

            GameObject card;
            if (eventCardPrefab != null)
            {
                card = Instantiate(eventCardPrefab, timelineContainer);
            }
            else
            {
                // Fallback: create a simple text object
                card = new GameObject($"Event_{_eventCards.Count}");
                card.transform.SetParent(timelineContainer, false);
                var text = card.AddComponent<TextMeshProUGUI>();
                text.fontSize = 14;
                text.enableWordWrapping = true;
            }

            // Position the card
            card.transform.localPosition = new Vector3(
                0, -_eventCards.Count * eventSpacing, 0);

            // Set content based on event type
            string icon = GetEventIcon(evt.type);
            string content = FormatEventContent(evt);

            var texts = card.GetComponentsInChildren<TextMeshProUGUI>();
            if (texts.Length > 0)
            {
                texts[0].text = $"{icon} {content}";
            }

            _eventCards.Add(card);
        }

        private string GetEventIcon(string eventType)
        {
            return eventType switch
            {
                "thinking" => "\u25cb", // ○ circle
                "tool_call" => "\u25b6", // ▶ play
                "tool_result" => "\u2714", // ✔ check
                "token" => "\u270e", // ✎ pencil
                "done" => "\u2605", // ★ star
                "error" => "\u2716", // ✖ cross
                _ => "\u2022" // • bullet
            };
        }

        private string FormatEventContent(AgentEvent evt)
        {
            return evt.type switch
            {
                "thinking" => $"<color=#88CCFF>Thinking:</color> {evt.content}",
                "tool_call" => $"<color=#FFCC44>Tool: {evt.toolName}</color>\n<size=80%>{evt.toolInput}</size>",
                "tool_result" => $"<color=#44CC88>Result:</color> {evt.toolOutput ?? evt.content}",
                "token" => $"<color=#CCCCCC>{evt.content}</color>",
                "done" => $"<color=#44FF44>{evt.content}</color>",
                "error" => $"<color=#FF4444>{evt.content}</color>",
                _ => evt.content
            };
        }

        private void ClearTimeline()
        {
            foreach (var card in _eventCards)
            {
                if (card != null) Destroy(card);
            }
            _eventCards.Clear();

            if (outputText != null)
                outputText.text = "";
        }

        private void UpdateStatus(string status)
        {
            if (statusText != null)
                statusText.text = $"Status: {status}";
        }

        private void UpdateRunButton()
        {
            if (runButtonText != null)
                runButtonText.text = _isRunning ? "Stop" : "Run Again";
        }

        // ── Button Handlers ────────────────────────────────────────

        private void OnRunClicked()
        {
            if (_isRunning)
                StopAgentRun();
            else
                StartAgentRun();
        }

        private void OnBackClicked()
        {
            StopAgentRun();
            SceneNavigator.Instance?.GoHome();
        }
    }
}
