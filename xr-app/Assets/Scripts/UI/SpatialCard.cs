using UnityEngine;
using UnityEngine.EventSystems;
using TMPro;

namespace NexusXR.UI
{
    /// <summary>
    /// A selectable spatial card used for displaying individual results,
    /// documents, or events. Responds to XR pointer interactions with
    /// hover highlights and click actions.
    /// </summary>
    public class SpatialCard : MonoBehaviour,
        IPointerEnterHandler, IPointerExitHandler, IPointerClickHandler
    {
        [Header("Content")]
        [Tooltip("Primary title text")]
        public TextMeshProUGUI titleText;

        [Tooltip("Body/description text")]
        public TextMeshProUGUI bodyText;

        [Tooltip("Footer/metadata text")]
        public TextMeshProUGUI footerText;

        [Header("Visual Feedback")]
        [Tooltip("Background renderer for hover effect")]
        public Renderer backgroundRenderer;

        [Tooltip("Normal background color")]
        public Color normalColor = new Color(0.1f, 0.1f, 0.15f, 0.85f);

        [Tooltip("Highlighted background color on hover")]
        public Color hoverColor = new Color(0.15f, 0.2f, 0.3f, 0.95f);

        [Tooltip("Scale multiplier on hover")]
        public float hoverScale = 1.03f;

        private Vector3 _originalScale;
        private bool _isHovered;

        /// <summary>Callback invoked when the card is clicked.</summary>
        public System.Action<SpatialCard> OnClicked;

        private void Start()
        {
            _originalScale = transform.localScale;
            SetColor(normalColor);
        }

        public void OnPointerEnter(PointerEventData eventData)
        {
            _isHovered = true;
            SetColor(hoverColor);
            transform.localScale = _originalScale * hoverScale;
        }

        public void OnPointerExit(PointerEventData eventData)
        {
            _isHovered = false;
            SetColor(normalColor);
            transform.localScale = _originalScale;
        }

        public void OnPointerClick(PointerEventData eventData)
        {
            OnClicked?.Invoke(this);
        }

        /// <summary>Set the card's content fields.</summary>
        public void SetContent(string title, string body, string footer = "")
        {
            if (titleText != null) titleText.text = title;
            if (bodyText != null) bodyText.text = body;
            if (footerText != null) footerText.text = footer;
        }

        private void SetColor(Color color)
        {
            if (backgroundRenderer != null)
            {
                var block = new MaterialPropertyBlock();
                block.SetColor("_Color", color);
                backgroundRenderer.SetPropertyBlock(block);
            }
        }
    }
}
