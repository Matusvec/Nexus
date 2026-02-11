using System.Collections;
using UnityEngine;
using NexusXR.Core;
using NexusXR.API;

namespace NexusXR.Demo
{
    /// <summary>
    /// Deterministic demo script that walks through all core product flows.
    /// Attach to a GameObject in the Home scene and press 'D' (desktop)
    /// or the secondary button (B on Quest) to start the demo sequence.
    ///
    /// Demo sequence:
    /// 1. Home → type a question → search
    /// 2. Retrieval Results → browse results → back
    /// 3. Home → type a task → run agent
    /// 4. Agent Workspace → watch streaming progress → back
    /// 5. Home → open docs library
    /// 6. Docs Library → browse → back
    /// </summary>
    public class DemoWalkthrough : MonoBehaviour
    {
        [Header("Demo Settings")]
        [Tooltip("Auto-start the demo on scene load")]
        public bool autoStart;

        [Tooltip("Delay between demo steps (seconds)")]
        [Range(1f, 10f)]
        public float stepDelay = 3f;

        [Tooltip("Demo query for retrieval")]
        public string demoQuery = "What is RAPTOR retrieval and how does it work?";

        [Tooltip("Demo task for agent")]
        public string demoAgentTask = "Summarize all documents and identify key themes across the knowledge base";

        private bool _isRunning;
        private int _currentStep;

        private void Start()
        {
            // Ensure mock mode is on for demo
            NexusApiClient.Instance.useMockMode = true;

            if (autoStart)
            {
                StartDemo();
            }
        }

        private void Update()
        {
            // Desktop: press D to start demo
            // VR: press B button (secondary) to start demo
            if (!_isRunning)
            {
                if (UnityEngine.Input.GetKeyDown(KeyCode.D))
                {
                    StartDemo();
                }
            }
        }

        /// <summary>Start the full demo walkthrough.</summary>
        public void StartDemo()
        {
            if (_isRunning) return;
            _isRunning = true;
            _currentStep = 0;
            Debug.Log("[Demo] ========== STARTING NEXUS XR DEMO ==========");
            StartCoroutine(RunDemoSequence());
        }

        private IEnumerator RunDemoSequence()
        {
            // ── Step 1: Home screen ────────────────────────────────
            LogStep("Welcome to Nexus XR - Your AI Research Workspace in VR");
            yield return new WaitForSeconds(stepDelay);

            // ── Step 2: Search query ───────────────────────────────
            LogStep($"Searching: \"{demoQuery}\"");
            SceneNavigator.Instance?.GoToRetrievalResults(demoQuery);
            yield return new WaitForSeconds(stepDelay * 2f);

            // ── Step 3: Back to home ───────────────────────────────
            LogStep("Returning to Home");
            SceneNavigator.Instance?.GoHome();
            yield return new WaitForSeconds(stepDelay);

            // ── Step 4: Agent task ─────────────────────────────────
            LogStep($"Running agent task: \"{demoAgentTask}\"");
            SceneNavigator.Instance?.GoToAgentWorkspace(demoAgentTask);
            yield return new WaitForSeconds(stepDelay * 3f);

            // ── Step 5: Back to home ───────────────────────────────
            LogStep("Returning to Home");
            SceneNavigator.Instance?.GoHome();
            yield return new WaitForSeconds(stepDelay);

            // ── Step 6: Document library ───────────────────────────
            LogStep("Opening Document Library");
            SceneNavigator.Instance?.GoToDocsLibrary();
            yield return new WaitForSeconds(stepDelay * 2f);

            // ── Step 7: Back to home ───────────────────────────────
            LogStep("Returning to Home");
            SceneNavigator.Instance?.GoHome();
            yield return new WaitForSeconds(stepDelay);

            // ── Done ───────────────────────────────────────────────
            Debug.Log("[Demo] ========== DEMO COMPLETE ==========");
            Debug.Log("[Demo] All core flows demonstrated:");
            Debug.Log("[Demo]   1. Ask a question → retrieval results with citations");
            Debug.Log("[Demo]   2. Run an agent task → streamed progress + tool calls");
            Debug.Log("[Demo]   3. Document library → browse/manage documents");
            _isRunning = false;
        }

        private void LogStep(string message)
        {
            _currentStep++;
            Debug.Log($"[Demo] Step {_currentStep}: {message}");
        }
    }
}
