using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;
using NexusXR.API.Models;

namespace NexusXR.API
{
    /// <summary>
    /// Central API client for communicating with the Nexus backend.
    /// Supports both live HTTP mode and a local mock mode for development
    /// and demos without a running backend.
    /// 
    /// Usage:
    ///   NexusApiClient.Instance.UseMockMode = true; // toggle mock
    ///   StartCoroutine(NexusApiClient.Instance.Query(request, OnResult, OnError));
    /// </summary>
    public class NexusApiClient : MonoBehaviour
    {
        // ── Singleton ──────────────────────────────────────────────
        private static NexusApiClient _instance;
        public static NexusApiClient Instance
        {
            get
            {
                if (_instance == null)
                {
                    var go = new GameObject("[NexusApiClient]");
                    _instance = go.AddComponent<NexusApiClient>();
                    DontDestroyOnLoad(go);
                }
                return _instance;
            }
        }

        // ── Configuration ──────────────────────────────────────────
        [Header("Backend Connection")]
        [Tooltip("Base URL of the Nexus FastAPI backend")]
        public string baseUrl = "http://localhost:8000";

        [Tooltip("Enable mock mode (no backend required)")]
        public bool useMockMode = true;

        [Tooltip("Simulated network latency in mock mode (seconds)")]
        [Range(0.1f, 3f)]
        public float mockLatency = 0.5f;

        private MockDataProvider _mockProvider;

        private void Awake()
        {
            if (_instance != null && _instance != this)
            {
                Destroy(gameObject);
                return;
            }
            _instance = this;
            DontDestroyOnLoad(gameObject);
            _mockProvider = new MockDataProvider();
        }

        // ── Query ──────────────────────────────────────────────────

        /// <summary>Send a retrieval query to the Nexus backend.</summary>
        public IEnumerator Query(
            QueryRequest request,
            Action<QueryResponse> onSuccess,
            Action<string> onError)
        {
            if (useMockMode)
            {
                yield return new WaitForSeconds(mockLatency);
                onSuccess?.Invoke(_mockProvider.GetQueryResponse(request.question));
                yield break;
            }

            string json = JsonUtility.ToJson(request);
            using var webRequest = new UnityWebRequest($"{baseUrl}/query", "POST");
            byte[] bodyRaw = Encoding.UTF8.GetBytes(json);
            webRequest.uploadHandler = new UploadHandlerRaw(bodyRaw);
            webRequest.downloadHandler = new DownloadHandlerBuffer();
            webRequest.SetRequestHeader("Content-Type", "application/json");

            yield return webRequest.SendWebRequest();

            if (webRequest.result != UnityWebRequest.Result.Success)
            {
                onError?.Invoke(webRequest.error);
            }
            else
            {
                var response = JsonUtility.FromJson<QueryResponse>(
                    webRequest.downloadHandler.text);
                onSuccess?.Invoke(response);
            }
        }

        // ── Documents ──────────────────────────────────────────────

        /// <summary>List all documents in the knowledge base.</summary>
        public IEnumerator GetDocuments(
            Action<List<NexusDocument>> onSuccess,
            Action<string> onError)
        {
            if (useMockMode)
            {
                yield return new WaitForSeconds(mockLatency);
                onSuccess?.Invoke(_mockProvider.GetDocuments());
                yield break;
            }

            using var webRequest = UnityWebRequest.Get($"{baseUrl}/documents");
            yield return webRequest.SendWebRequest();

            if (webRequest.result != UnityWebRequest.Result.Success)
            {
                onError?.Invoke(webRequest.error);
            }
            else
            {
                string wrappedJson = "{\"items\":" + webRequest.downloadHandler.text + "}";
                var wrapper = JsonUtility.FromJson<DocumentListWrapper>(wrappedJson);
                onSuccess?.Invoke(wrapper.items);
            }
        }

        /// <summary>Delete a document by ID.</summary>
        public IEnumerator DeleteDocument(
            string documentId,
            Action<bool> onSuccess,
            Action<string> onError)
        {
            if (useMockMode)
            {
                yield return new WaitForSeconds(mockLatency);
                _mockProvider.RemoveDocument(documentId);
                onSuccess?.Invoke(true);
                yield break;
            }

            using var webRequest = UnityWebRequest.Delete(
                $"{baseUrl}/documents/{documentId}");
            yield return webRequest.SendWebRequest();

            if (webRequest.result != UnityWebRequest.Result.Success)
            {
                onError?.Invoke(webRequest.error);
            }
            else
            {
                onSuccess?.Invoke(true);
            }
        }

        // ── Agent Runs ─────────────────────────────────────────────

        /// <summary>
        /// Start an agent run and stream events via a callback.
        /// In mock mode, simulates a realistic event sequence.
        /// </summary>
        public IEnumerator RunAgent(
            AgentRunRequest request,
            Action<AgentEvent> onEvent,
            Action<string> onError)
        {
            if (useMockMode)
            {
                var events = _mockProvider.GetAgentRunEvents(request.task);
                foreach (var evt in events)
                {
                    yield return new WaitForSeconds(
                        Mathf.Max(0.2f, mockLatency * 0.5f));
                    onEvent?.Invoke(evt);
                }
                yield break;
            }

            // Live mode: POST to agent endpoint and read SSE stream
            string json = JsonUtility.ToJson(request);
            using var webRequest = new UnityWebRequest(
                $"{baseUrl}/agent/run", "POST");
            byte[] bodyRaw = Encoding.UTF8.GetBytes(json);
            webRequest.uploadHandler = new UploadHandlerRaw(bodyRaw);
            webRequest.downloadHandler = new DownloadHandlerBuffer();
            webRequest.SetRequestHeader("Content-Type", "application/json");

            yield return webRequest.SendWebRequest();

            if (webRequest.result != UnityWebRequest.Result.Success)
            {
                onError?.Invoke(webRequest.error);
            }
            else
            {
                // Parse SSE-style response (simplified for prototype)
                string responseText = webRequest.downloadHandler.text;
                string[] lines = responseText.Split('\n');
                foreach (string line in lines)
                {
                    if (line.StartsWith("data: "))
                    {
                        string data = line.Substring(6);
                        var evt = JsonUtility.FromJson<AgentEvent>(data);
                        if (evt != null)
                            onEvent?.Invoke(evt);
                    }
                }
            }
        }

        // ── Stats ──────────────────────────────────────────────────

        /// <summary>Get database statistics.</summary>
        public IEnumerator GetStats(
            Action<DatabaseStats> onSuccess,
            Action<string> onError)
        {
            if (useMockMode)
            {
                yield return new WaitForSeconds(mockLatency * 0.5f);
                onSuccess?.Invoke(_mockProvider.GetStats());
                yield break;
            }

            using var webRequest = UnityWebRequest.Get($"{baseUrl}/stats");
            yield return webRequest.SendWebRequest();

            if (webRequest.result != UnityWebRequest.Result.Success)
            {
                onError?.Invoke(webRequest.error);
            }
            else
            {
                var stats = JsonUtility.FromJson<DatabaseStats>(
                    webRequest.downloadHandler.text);
                onSuccess?.Invoke(stats);
            }
        }

        // ── Helpers ────────────────────────────────────────────────

        [Serializable]
        private class DocumentListWrapper
        {
            public List<NexusDocument> items;
        }
    }
}
