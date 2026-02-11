using UnityEngine;
using UnityEngine.SceneManagement;
using System.Collections;

namespace NexusXR.Core
{
    /// <summary>
    /// Handles navigation between XR scenes with smooth transitions.
    /// Provides a fade-to-black transition to avoid discomfort during scene changes.
    /// </summary>
    public class SceneNavigator : MonoBehaviour
    {
        public static SceneNavigator Instance { get; private set; }

        /// <summary>Scene name constants matching Unity scene assets.</summary>
        public static class Scenes
        {
            public const string Home = "HomeCommand";
            public const string RetrievalResults = "RetrievalResults";
            public const string AgentWorkspace = "AgentWorkspace";
            public const string DocsLibrary = "DocsLibrary";
        }

        [Header("Transition Settings")]
        [Tooltip("Duration of the fade transition in seconds")]
        [Range(0.1f, 1f)]
        public float fadeDuration = 0.3f;

        [Tooltip("Color of the fade overlay")]
        public Color fadeColor = Color.black;

        /// <summary>True while a scene transition is in progress.</summary>
        public bool IsTransitioning { get; private set; }

        /// <summary>Name of the currently loaded scene.</summary>
        public string CurrentScene => SceneManager.GetActiveScene().name;

        // Data passed between scenes
        private static string _pendingQuery;
        private static string _pendingAgentTask;

        private void Awake()
        {
            if (Instance != null && Instance != this)
            {
                Destroy(gameObject);
                return;
            }
            Instance = this;
            DontDestroyOnLoad(gameObject);
        }

        /// <summary>Navigate to the Home/Command scene.</summary>
        public void GoHome()
        {
            LoadScene(Scenes.Home);
        }

        /// <summary>Navigate to Retrieval Results with a query.</summary>
        public void GoToRetrievalResults(string query)
        {
            _pendingQuery = query;
            LoadScene(Scenes.RetrievalResults);
        }

        /// <summary>Navigate to Agent Workspace with a task.</summary>
        public void GoToAgentWorkspace(string task)
        {
            _pendingAgentTask = task;
            LoadScene(Scenes.AgentWorkspace);
        }

        /// <summary>Navigate to the Document Library.</summary>
        public void GoToDocsLibrary()
        {
            LoadScene(Scenes.DocsLibrary);
        }

        /// <summary>Retrieve and clear the pending query (used by RetrievalResults scene).</summary>
        public static string ConsumePendingQuery()
        {
            string q = _pendingQuery;
            _pendingQuery = null;
            return q;
        }

        /// <summary>Retrieve and clear the pending agent task (used by AgentWorkspace scene).</summary>
        public static string ConsumePendingAgentTask()
        {
            string t = _pendingAgentTask;
            _pendingAgentTask = null;
            return t;
        }

        private void LoadScene(string sceneName)
        {
            if (IsTransitioning) return;
            if (CurrentScene == sceneName) return;

            StartCoroutine(TransitionToScene(sceneName));
        }

        private IEnumerator TransitionToScene(string sceneName)
        {
            IsTransitioning = true;
            Debug.Log($"[SceneNavigator] Transitioning to: {sceneName}");

            // Fade out (in a full implementation, this would use a screen-space overlay)
            yield return new WaitForSeconds(fadeDuration);

            // Load scene
            AsyncOperation asyncLoad = SceneManager.LoadSceneAsync(sceneName);
            if (asyncLoad != null)
            {
                while (!asyncLoad.isDone)
                {
                    yield return null;
                }
            }

            // Fade in
            yield return new WaitForSeconds(fadeDuration);

            IsTransitioning = false;
            Debug.Log($"[SceneNavigator] Arrived at: {sceneName}");
        }
    }
}
