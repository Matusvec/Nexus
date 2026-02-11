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
    /// Controller for the Document Library scene.
    /// Shows a spatial list of documents with their status (indexed / processing / error).
    /// Supports adding and removing documents (stubbed file picker).
    ///
    /// Layout: Documents displayed as floating cards in a grid arrangement
    /// with status badges and action buttons per card.
    /// </summary>
    public class DocsLibraryController : MonoBehaviour
    {
        [Header("UI References")]
        [Tooltip("Container for document cards")]
        public Transform docsContainer;

        [Tooltip("Prefab for a document card")]
        public GameObject docCardPrefab;

        [Tooltip("Summary stats text")]
        public TextMeshProUGUI statsText;

        [Tooltip("Add document button")]
        public Button addButton;

        [Tooltip("Refresh button")]
        public Button refreshButton;

        [Tooltip("Back button")]
        public Button backButton;

        [Tooltip("Loading indicator")]
        public GameObject loadingIndicator;

        [Header("Layout")]
        [Tooltip("Number of columns in the grid")]
        public int gridColumns = 3;

        [Tooltip("Horizontal spacing between cards")]
        public float horizontalSpacing = 0.4f;

        [Tooltip("Vertical spacing between rows")]
        public float verticalSpacing = 0.35f;

        private List<NexusDocument> _documents = new List<NexusDocument>();
        private readonly List<GameObject> _docCards = new List<GameObject>();

        private void Start()
        {
            if (backButton != null)
                backButton.onClick.AddListener(OnBackClicked);
            if (addButton != null)
                addButton.onClick.AddListener(OnAddClicked);
            if (refreshButton != null)
                refreshButton.onClick.AddListener(OnRefreshClicked);

            StartCoroutine(LoadDocuments());
        }

        private IEnumerator LoadDocuments()
        {
            SetLoading(true);

            yield return NexusApiClient.Instance.GetDocuments(
                OnDocumentsLoaded,
                OnDocumentsError
            );
        }

        private void OnDocumentsLoaded(List<NexusDocument> documents)
        {
            _documents = documents;
            SetLoading(false);
            DisplayDocuments();
            UpdateStats();
            Debug.Log($"[DocsLibrary] Loaded {documents.Count} documents");
        }

        private void OnDocumentsError(string error)
        {
            SetLoading(false);
            if (statsText != null)
                statsText.text = $"Error loading documents: {error}";
            Debug.LogError($"[DocsLibrary] Error: {error}");
        }

        private void DisplayDocuments()
        {
            ClearCards();

            if (docsContainer == null) return;

            for (int i = 0; i < _documents.Count; i++)
            {
                var doc = _documents[i];
                CreateDocCard(doc, i);
            }
        }

        private void CreateDocCard(NexusDocument doc, int index)
        {
            GameObject card;
            if (docCardPrefab != null)
            {
                card = Instantiate(docCardPrefab, docsContainer);
            }
            else
            {
                card = new GameObject($"DocCard_{doc.id}");
                card.transform.SetParent(docsContainer, false);
                var text = card.AddComponent<TextMeshProUGUI>();
                text.fontSize = 12;
                text.enableWordWrapping = true;
            }

            // Grid position
            int col = index % gridColumns;
            int row = index / gridColumns;
            float centerOffset = (gridColumns - 1) * horizontalSpacing * 0.5f;
            card.transform.localPosition = new Vector3(
                col * horizontalSpacing - centerOffset,
                -row * verticalSpacing,
                0
            );

            // Populate card content
            PopulateDocCard(card, doc);
            _docCards.Add(card);
        }

        private void PopulateDocCard(GameObject card, NexusDocument doc)
        {
            string statusIcon = doc.status switch
            {
                "ready" => "\u2714", // ✔
                "processing" => "\u25cb", // ○
                "error" => "\u2716", // ✖
                _ => "\u2022" // •
            };

            string statusColor = doc.status switch
            {
                "ready" => "#44CC88",
                "processing" => "#FFCC44",
                "error" => "#FF4444",
                _ => "#888888"
            };

            string fileSize = FormatFileSize(doc.fileSize);
            string summary = string.IsNullOrEmpty(doc.summary)
                ? "(Processing...)"
                : (doc.summary.Length > 100 ? doc.summary.Substring(0, 100) + "..." : doc.summary);

            var texts = card.GetComponentsInChildren<TextMeshProUGUI>();
            if (texts.Length >= 1)
            {
                texts[0].text =
                    $"<b>{doc.filename}</b>\n" +
                    $"<color={statusColor}>{statusIcon} {doc.status}</color> | " +
                    $"{doc.chunkCount} chunks | {fileSize}\n" +
                    $"<size=80%>{summary}</size>";
            }

            // Wire up delete button if present
            var deleteButton = card.GetComponentInChildren<Button>();
            if (deleteButton != null)
            {
                string docId = doc.id;
                deleteButton.onClick.AddListener(() => OnDeleteClicked(docId));
            }
        }

        private void UpdateStats()
        {
            if (statsText == null) return;

            int ready = 0, processing = 0;
            long totalSize = 0;
            int totalChunks = 0;

            foreach (var doc in _documents)
            {
                if (doc.status == "ready") ready++;
                else if (doc.status == "processing") processing++;
                totalSize += doc.fileSize;
                totalChunks += doc.chunkCount;
            }

            statsText.text =
                $"Documents: {_documents.Count} | Ready: {ready} | " +
                $"Processing: {processing} | Total chunks: {totalChunks} | " +
                $"Size: {FormatFileSize(totalSize)}";
        }

        private void ClearCards()
        {
            foreach (var card in _docCards)
            {
                if (card != null) Destroy(card);
            }
            _docCards.Clear();
        }

        private void SetLoading(bool loading)
        {
            if (loadingIndicator != null)
                loadingIndicator.SetActive(loading);
        }

        private static string FormatFileSize(long bytes)
        {
            if (bytes < 1024) return $"{bytes} B";
            if (bytes < 1024 * 1024) return $"{bytes / 1024f:F1} KB";
            return $"{bytes / (1024f * 1024f):F1} MB";
        }

        // ── Button Handlers ────────────────────────────────────────

        private void OnAddClicked()
        {
            // Stub: in a full implementation this would open a file picker
            // or trigger a native dialog. For VR, this might use a virtual
            // file browser or accept drops from a companion app.
            Debug.Log("[DocsLibrary] Add document (stub - would open file picker)");

            if (statsText != null)
                statsText.text = "Add Document: Use companion web app or drop files via ADB sideload.";
        }

        private void OnDeleteClicked(string documentId)
        {
            Debug.Log($"[DocsLibrary] Deleting document: {documentId}");
            StartCoroutine(DeleteDocument(documentId));
        }

        private IEnumerator DeleteDocument(string documentId)
        {
            yield return NexusApiClient.Instance.DeleteDocument(
                documentId,
                success =>
                {
                    Debug.Log($"[DocsLibrary] Deleted: {documentId}");
                    StartCoroutine(LoadDocuments());
                },
                error =>
                {
                    Debug.LogError($"[DocsLibrary] Delete failed: {error}");
                }
            );
        }

        private void OnRefreshClicked()
        {
            StartCoroutine(LoadDocuments());
        }

        private void OnBackClicked()
        {
            SceneNavigator.Instance?.GoHome();
        }
    }
}
