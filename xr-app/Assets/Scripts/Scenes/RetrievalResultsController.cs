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
    /// Controller for the Retrieval Results scene.
    /// Displays query results as a spatial list of cards with citations.
    /// Each card shows a source chunk with its document name, layer, and relevance score.
    /// A "why this result" panel (stub) explains relevance ranking.
    ///
    /// Layout: Results appear as floating cards arranged in a gentle arc
    /// in front of the user, with the answer panel on the left and source
    /// cards fanning out to the right.
    /// </summary>
    public class RetrievalResultsController : MonoBehaviour
    {
        [Header("UI References")]
        [Tooltip("Text displaying the user's query")]
        public TextMeshProUGUI queryText;

        [Tooltip("Panel showing the AI-generated answer")]
        public TextMeshProUGUI answerText;

        [Tooltip("Container for result source cards")]
        public Transform resultsContainer;

        [Tooltip("Prefab for a single result card")]
        public GameObject resultCardPrefab;

        [Tooltip("'Why this result' explanation panel text")]
        public TextMeshProUGUI whyPanelText;

        [Tooltip("Loading indicator")]
        public GameObject loadingIndicator;

        [Tooltip("Back button")]
        public Button backButton;

        [Header("Layout")]
        [Tooltip("Spacing between result cards (meters)")]
        public float cardSpacing = 0.35f;

        [Tooltip("Arc angle spread for result cards (degrees)")]
        public float arcAngle = 40f;

        [Tooltip("Distance of cards from center (meters)")]
        public float cardDistance = 1.2f;

        private string _currentQuery;
        private QueryResponse _currentResponse;

        private void Start()
        {
            if (backButton != null)
                backButton.onClick.AddListener(OnBackClicked);

            // Get the query passed from the Home scene
            _currentQuery = SceneNavigator.ConsumePendingQuery();
            if (string.IsNullOrEmpty(_currentQuery))
            {
                _currentQuery = "What is RAPTOR retrieval?";
                Debug.Log("[RetrievalResults] No pending query, using default");
            }

            if (queryText != null)
                queryText.text = _currentQuery;

            // Start the query
            StartCoroutine(ExecuteQuery());
        }

        private IEnumerator ExecuteQuery()
        {
            SetLoading(true);

            var request = new QueryRequest
            {
                question = _currentQuery,
                top_k = 5
            };

            yield return NexusApiClient.Instance.Query(
                request,
                OnQuerySuccess,
                OnQueryError
            );
        }

        private void OnQuerySuccess(QueryResponse response)
        {
            _currentResponse = response;
            SetLoading(false);

            // Display the answer
            if (answerText != null)
                answerText.text = response.answer;

            // Display source cards
            DisplaySourceCards(response.sources);

            // Set default "why" panel text
            if (whyPanelText != null)
            {
                whyPanelText.text =
                    "Result Ranking\n\n" +
                    $"Query type: {response.queryType}\n" +
                    $"Tokens used: {response.tokensUsed}\n" +
                    $"Sources found: {response.sources.Count}\n\n" +
                    "Results are ranked by cosine similarity between " +
                    "the query embedding and chunk embeddings across " +
                    "all RAPTOR tree layers. Higher layers provide " +
                    "broader context while lower layers provide details.";
            }

            Debug.Log($"[RetrievalResults] Displayed {response.sources.Count} results");
        }

        private void OnQueryError(string error)
        {
            SetLoading(false);
            if (answerText != null)
                answerText.text = $"Error: {error}\n\nCheck backend connection or enable mock mode.";
            Debug.LogError($"[RetrievalResults] Query error: {error}");
        }

        private void DisplaySourceCards(List<RetrievalSource> sources)
        {
            if (resultsContainer == null || resultCardPrefab == null) return;

            // Clear existing cards
            foreach (Transform child in resultsContainer)
            {
                Destroy(child.gameObject);
            }

            // Create cards in an arc arrangement
            for (int i = 0; i < sources.Count; i++)
            {
                var source = sources[i];
                var card = Instantiate(resultCardPrefab, resultsContainer);

                // Position cards in an arc
                float angle = (i - (sources.Count - 1) * 0.5f) * (arcAngle / Mathf.Max(1, sources.Count - 1));
                float rad = angle * Mathf.Deg2Rad;
                card.transform.localPosition = new Vector3(
                    Mathf.Sin(rad) * cardDistance,
                    -i * cardSpacing * 0.3f,
                    Mathf.Cos(rad) * cardDistance - cardDistance
                );
                card.transform.LookAt(resultsContainer.position);
                card.transform.Rotate(0, 180, 0);

                // Populate card content
                PopulateCard(card, source, i + 1);
            }
        }

        private void PopulateCard(GameObject card, RetrievalSource source, int index)
        {
            // Find child text components and populate
            var texts = card.GetComponentsInChildren<TextMeshProUGUI>();
            if (texts.Length >= 3)
            {
                texts[0].text = $"#{index} | {source.documentName}";
                texts[1].text = source.content;
                texts[2].text = $"Layer {source.layer} | Score: {source.relevanceScore:F2} | {source.chunkId}";
            }
            else if (texts.Length >= 1)
            {
                texts[0].text =
                    $"<b>#{index} {source.documentName}</b>\n" +
                    $"<size=80%>{source.content}</size>\n" +
                    $"<size=70%><color=#888>Layer {source.layer} | Score: {source.relevanceScore:F2}</color></size>";
            }
        }

        private void SetLoading(bool loading)
        {
            if (loadingIndicator != null)
                loadingIndicator.SetActive(loading);
        }

        private void OnBackClicked()
        {
            SceneNavigator.Instance?.GoHome();
        }
    }
}
