const API_BASE = 'http://localhost:8000';

async function apiPost(endpoint, body) {
  const res = await fetch(`${API_BASE}${endpoint}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

async function apiGet(endpoint) {
  const res = await fetch(`${API_BASE}${endpoint}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

// Store last prediction result in sessionStorage for results page
function savePrediction(data, formData) {
  sessionStorage.setItem('heartPrediction', JSON.stringify(data));
  sessionStorage.setItem('heartFormData', JSON.stringify(formData));
}

function loadPrediction() {
  const pred = sessionStorage.getItem('heartPrediction');
  const form = sessionStorage.getItem('heartFormData');
  return {
    prediction: pred ? JSON.parse(pred) : null,
    formData: form ? JSON.parse(form) : null
  };
}
