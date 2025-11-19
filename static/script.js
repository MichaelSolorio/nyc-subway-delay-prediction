/**
 * ATLAS Frontend JavaScript
 * Handles API calls and UI updates
 */

// Configuration
const API_BASE_URL = 'http://localhost:5000/api';
let autoRefreshInterval = null;
let isAutoRefresh = false;

// DOM Elements
const statusDisplay = document.getElementById('status-display');
const predictionsContainer = document.getElementById('predictions-container');
const refreshBtn = document.getElementById('refresh-btn');
const autoRefreshBtn = document.getElementById('auto-refresh-btn');

/**
 * Fetch predictions from API
 */
async function fetchPredictions() {
    try {
        statusDisplay.innerHTML = '<p class="loading">🔄 Fetching predictions...</p>';

        const response = await fetch(`${API_BASE_URL}/predict`);
        const data = await response.json();

        if (data.success) {
            displayPredictions(data);
            updateStatus('✅ Predictions updated', 'success');
        } else {
            updateStatus('❌ Error: ' + data.error, 'error');
        }

    } catch (error) {
        console.error('Error fetching predictions:', error);
        updateStatus('❌ Could not connect to API. Is the Flask server running?', 'error');
    }
}

/**
 * Display predictions in the UI
 */
function displayPredictions(data) {
    if (!data.predictions || data.predictions.length === 0) {
        predictionsContainer.innerHTML = '<p class="placeholder">No predictions available yet. Train the model first!</p>';
        return;
    }

    predictionsContainer.innerHTML = '';

    data.predictions.forEach(pred => {
        const card = document.createElement('div');
        card.className = `prediction-card ${pred.prediction}`;

        card.innerHTML = `
            <div class="station-name">${pred.station}</div>
            <div class="prediction-status ${pred.prediction}">
                ${pred.prediction === 'on-time' ? '✅ On-Time' : '⚠️ Delayed'}
            </div>
            <div class="confidence">
                Confidence: ${(pred.confidence * 100).toFixed(1)}%
            </div>
            ${pred.estimated_delay_minutes ?
                `<div class="delay-time">~${pred.estimated_delay_minutes} min delay</div>`
                : ''}
        `;

        predictionsContainer.appendChild(card);
    });
}

/**
 * Update status message
 */
function updateStatus(message, type) {
    const className = type === 'success' ? 'success' : 'error';
    statusDisplay.innerHTML = `<p class="${className}">${message}</p>`;
}

/**
 * Toggle auto-refresh
 */
function toggleAutoRefresh() {
    isAutoRefresh = !isAutoRefresh;

    if (isAutoRefresh) {
        // Enable auto-refresh (every 30 seconds)
        fetchPredictions(); // Fetch immediately
        autoRefreshInterval = setInterval(fetchPredictions, 30000);
        autoRefreshBtn.textContent = 'Auto-Refresh: ON';
        autoRefreshBtn.classList.add('active');
    } else {
        // Disable auto-refresh
        if (autoRefreshInterval) {
            clearInterval(autoRefreshInterval);
            autoRefreshInterval = null;
        }
        autoRefreshBtn.textContent = 'Auto-Refresh: OFF';
        autoRefreshBtn.classList.remove('active');
    }
}

/**
 * Check API health
 */
async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        console.log('API Health:', data);
        return data.status === 'healthy';
    } catch (error) {
        console.error('API health check failed:', error);
        return false;
    }
}

// Event Listeners
refreshBtn.addEventListener('click', fetchPredictions);
autoRefreshBtn.addEventListener('click', toggleAutoRefresh);

// Initialize on page load
window.addEventListener('DOMContentLoaded', async () => {
    console.log('ATLAS Frontend initialized');

    // Check if API is available
    const isHealthy = await checkAPIHealth();

    if (isHealthy) {
        updateStatus('✅ Connected to ATLAS API', 'success');
    } else {
        updateStatus('⚠️ API not available. Start Flask server with: python -m src.api.app', 'error');
    }
});
