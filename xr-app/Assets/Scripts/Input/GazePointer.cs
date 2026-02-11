using UnityEngine;

namespace NexusXR.Input
{
    /// <summary>
    /// Head-gaze based pointer for fallback input when controllers
    /// are unavailable or for AR glasses-style interaction.
    /// Casts a ray from the camera center and shows a reticle at the hit point.
    ///
    /// In AR mode, gaze pointing is the primary selection method,
    /// combined with dwell-to-select or pinch gestures.
    /// </summary>
    public class GazePointer : MonoBehaviour
    {
        [Header("Configuration")]
        [Tooltip("Maximum raycast distance")]
        public float maxDistance = 10f;

        [Tooltip("Layer mask for interactable UI elements")]
        public LayerMask uiLayerMask = -1;

        [Tooltip("Reticle GameObject shown at the gaze hit point")]
        public GameObject reticle;

        [Tooltip("Default reticle scale")]
        public float reticleScale = 0.01f;

        [Tooltip("Reticle scale when hovering over an interactable")]
        public float reticleHoverScale = 0.015f;

        [Header("Dwell Selection (AR Mode)")]
        [Tooltip("Enable dwell-to-select (gaze at target for N seconds)")]
        public bool enableDwell;

        [Tooltip("Dwell time to trigger selection (seconds)")]
        [Range(0.5f, 3f)]
        public float dwellTime = 1.5f;

        [Tooltip("Visual indicator for dwell progress")]
        public UnityEngine.UI.Image dwellProgressImage;

        private Camera _camera;
        private float _dwellTimer;
        private GameObject _currentTarget;

        private void Start()
        {
            _camera = Camera.main;
            if (reticle != null)
                reticle.SetActive(true);
        }

        private void Update()
        {
            if (_camera == null) return;

            Ray ray = new Ray(_camera.transform.position, _camera.transform.forward);
            bool hit = Physics.Raycast(ray, out RaycastHit hitInfo, maxDistance, uiLayerMask);

            if (hit)
            {
                // Position reticle at hit point
                if (reticle != null)
                {
                    reticle.SetActive(true);
                    reticle.transform.position = hitInfo.point;
                    reticle.transform.rotation = Quaternion.LookRotation(hitInfo.normal);
                    reticle.transform.localScale = Vector3.one * reticleHoverScale;
                }

                // Dwell selection
                if (enableDwell)
                {
                    if (hitInfo.collider.gameObject == _currentTarget)
                    {
                        _dwellTimer += Time.deltaTime;
                        UpdateDwellProgress(_dwellTimer / dwellTime);

                        if (_dwellTimer >= dwellTime)
                        {
                            TriggerSelect(hitInfo.collider.gameObject);
                            _dwellTimer = 0f;
                        }
                    }
                    else
                    {
                        _currentTarget = hitInfo.collider.gameObject;
                        _dwellTimer = 0f;
                        UpdateDwellProgress(0f);
                    }
                }
            }
            else
            {
                if (reticle != null)
                {
                    reticle.transform.position = ray.GetPoint(maxDistance);
                    reticle.transform.localScale = Vector3.one * reticleScale;
                }

                _currentTarget = null;
                _dwellTimer = 0f;
                UpdateDwellProgress(0f);
            }
        }

        private void UpdateDwellProgress(float progress)
        {
            if (dwellProgressImage != null)
            {
                dwellProgressImage.fillAmount = Mathf.Clamp01(progress);
                dwellProgressImage.gameObject.SetActive(progress > 0f);
            }
        }

        private void TriggerSelect(GameObject target)
        {
            Debug.Log($"[GazePointer] Dwell selected: {target.name}");
            // Send a click event to the target's event handler
            var handler = target.GetComponent<UnityEngine.EventSystems.IPointerClickHandler>();
            if (handler != null)
            {
                handler.OnPointerClick(
                    new UnityEngine.EventSystems.PointerEventData(
                        UnityEngine.EventSystems.EventSystem.current));
            }
        }
    }
}
