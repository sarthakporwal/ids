// CANShield IDS - Frontend JavaScript

// API Configuration
const API_URL = 'http://localhost:8000';

// State
let currentData = null;
let predictions = null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
    setupEventListeners();
    checkAPIHealth();
});

function initializeApp() {
    console.log('🛡️ CANShield IDS initialized');
    
    // Initialize threshold display
    const thresholdInput = document.getElementById('threshold');
    const thresholdValue = document.getElementById('thresholdValue');
    
    thresholdInput.addEventListener('input', (e) => {
        thresholdValue.textContent = parseFloat(e.target.value).toFixed(4);
    });
    
    // Sample data visualization
    createSampleCharts();
}

function setupEventListeners() {
    // File upload
    const fileInput = document.getElementById('fileInput');
    const uploadArea = document.getElementById('uploadArea');
    
    fileInput.addEventListener('change', handleFileSelect);
    
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = '#667eea';
        uploadArea.style.background = 'rgba(102, 126, 234, 0.05)';
    });
    
    uploadArea.addEventListener('dragleave', (e) => {
        uploadArea.style.borderColor = '';
        uploadArea.style.background = '';
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = '';
        uploadArea.style.background = '';
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });
    
    // Run detection button
    document.getElementById('runDetectionBtn').addEventListener('click', runDetection);
}

async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_URL}/health`);
        const data = await response.json();
        
        if (data.status === 'healthy') {
            updateSystemStatus('Active', 'positive');
            console.log('✅ API connected');
        }
    } catch (error) {
        console.warn('⚠️ API not available. Using demo mode.');
        updateSystemStatus('Demo Mode', 'warning');
    }
}

function updateSystemStatus(status, type) {
    const statusElement = document.getElementById('systemStatus');
    statusElement.textContent = status;
    statusElement.className = type;
}

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    const uploadStatus = document.getElementById('uploadStatus');
    
    if (!file.name.endsWith('.csv')) {
        uploadStatus.className = 'upload-status error';
        uploadStatus.textContent = '❌ Please upload a CSV file';
        return;
    }
    
    uploadStatus.className = 'upload-status success';
    uploadStatus.innerHTML = `
        ✅ File uploaded successfully<br>
        📄 <strong>${file.name}</strong> (${(file.size / 1024).toFixed(2)} KB)
    `;
    
    // Read file
    const reader = new FileReader();
    reader.onload = (e) => {
        const text = e.target.result;
        parseCSV(text);
    };
    reader.readAsText(file);
}

function parseCSV(text) {
    const lines = text.split('\\n');
    const data = [];
    
    for (let i = 1; i < Math.min(lines.length, 1000); i++) {
        const values = lines[i].split(',');
        if (values.length > 0) {
            data.push(values);
        }
    }
    
    currentData = data;
    updateMetric('samplesProcessed', data.length);
    console.log(`📊 Loaded ${data.length} samples`);
}

async function runDetection() {
    const btn = document.getElementById('runDetectionBtn');
    const threshold = parseFloat(document.getElementById('threshold').value);
    
    if (!currentData) {
        alert('Please upload a dataset first');
        return;
    }
    
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Detecting...';
    
    // Simulate detection (replace with actual API call)
    setTimeout(() => {
        simulateDetection(threshold);
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-play"></i> Run Detection';
    }, 2000);
}

function simulateDetection(threshold) {
    const numSamples = currentData.length;
    
    // Generate synthetic predictions
    const errors = [];
    const attackTypes = ['Flooding', 'Suppress', 'Plateau', 'Continuous', 'Playback'];
    
    for (let i = 0; i < numSamples; i++) {
        const baseError = 0.002 + Math.random() * 0.002;
        const isAttack = Math.random() > 0.85;
        const error = isAttack ? baseError + Math.random() * 0.01 : baseError;
        
        errors.push({
            sampleId: i,
            error: error,
            isAttack: error > threshold,
            attackType: error > threshold ? attackTypes[Math.floor(Math.random() * attackTypes.length)] : 'Normal'
        });
    }
    
    predictions = errors;
    
    // Update UI
    const attacksDetected = errors.filter(e => e.isAttack).length;
    updateMetric('attacksDetected', attacksDetected);
    updateMetric('attackRate', `${(attacksDetected / numSamples * 100).toFixed(1)}%`);
    
    // Update charts
    updateErrorChart(errors, threshold);
    updateAttackTable(errors.filter(e => e.isAttack).slice(0, 10));
    updateAttackDistribution(errors);
    updateErrorHistogram(errors, threshold);
    
    console.log(`✅ Detection complete: ${attacksDetected} attacks found`);
}

function updateMetric(id, value) {
    const element = document.getElementById(id);
    if (element) {
        element.textContent = value;
    }
}

function createSampleCharts() {
    // Sample error chart
    const sampleData = {
        x: Array.from({length: 100}, (_, i) => i),
        y: Array.from({length: 100}, () => 0.002 + Math.random() * 0.001),
        type: 'scatter',
        mode: 'lines',
        name: 'Reconstruction Error',
        line: {color: '#667eea', width: 2},
        fill: 'tozeroy'
    };
    
    const layout = {
        title: 'Reconstruction Error Timeline',
        xaxis: {title: 'Sample Index'},
        yaxis: {title: 'Reconstruction Error'},
        hovermode: 'x unified'
    };
    
    Plotly.newPlot('errorChart', [sampleData], layout, {responsive: true});
}

function updateErrorChart(errors, threshold) {
    const x = errors.map(e => e.sampleId);
    const y = errors.map(e => e.error);
    
    const trace1 = {
        x: x,
        y: y,
        type: 'scatter',
        mode: 'lines',
        name: 'Reconstruction Error',
        line: {color: '#667eea', width: 2},
        fill: 'tozeroy'
    };
    
    const trace2 = {
        x: [0, errors.length],
        y: [threshold, threshold],
        type: 'scatter',
        mode: 'lines',
        name: 'Threshold',
        line: {color: 'red', width: 2, dash: 'dash'}
    };
    
    const attackPoints = errors.filter(e => e.isAttack);
    const trace3 = {
        x: attackPoints.map(e => e.sampleId),
        y: attackPoints.map(e => e.error),
        type: 'scatter',
        mode: 'markers',
        name: 'Attack Detected',
        marker: {color: 'red', size: 8, symbol: 'x'}
    };
    
    const layout = {
        title: 'Reconstruction Error Timeline',
        xaxis: {title: 'Sample Index'},
        yaxis: {title: 'Reconstruction Error'},
        hovermode: 'x unified'
    };
    
    Plotly.newPlot('errorChart', [trace1, trace2, trace3], layout, {responsive: true});
}

function updateAttackTable(attacks) {
    const tbody = document.getElementById('attackTableBody');
    
    if (attacks.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="5" style="text-align: center; color: #4bb543;">
                    ✅ No attacks detected - Traffic is normal
                </td>
            </tr>
        `;
        return;
    }
    
    tbody.innerHTML = attacks.map(attack => `
        <tr>
            <td>${attack.sampleId}</td>
            <td><strong>${attack.attackType}</strong></td>
            <td>${attack.error.toFixed(6)}</td>
            <td>${attack.error > 0.01 ? 'High' : 'Medium'}</td>
            <td><span class="status-badge status-attack">ATTACK</span></td>
        </tr>
    `).join('');
}

function updateAttackDistribution(errors) {
    const attackCounts = {};
    
    errors.forEach(e => {
        if (e.isAttack) {
            attackCounts[e.attackType] = (attackCounts[e.attackType] || 0) + 1;
        }
    });
    
    const data = [{
        values: Object.values(attackCounts),
        labels: Object.keys(attackCounts),
        type: 'pie',
        marker: {
            colors: ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#fa709a']
        }
    }];
    
    const layout = {
        title: 'Attack Type Distribution'
    };
    
    Plotly.newPlot('attackPieChart', data, layout, {responsive: true});
}

function updateErrorHistogram(errors, threshold) {
    const data = [{
        x: errors.map(e => e.error),
        type: 'histogram',
        nbinsx: 50,
        marker: {color: '#667eea'}
    }];
    
    const layout = {
        title: 'Error Distribution',
        xaxis: {title: 'Reconstruction Error'},
        yaxis: {title: 'Count'},
        shapes: [{
            type: 'line',
            x0: threshold,
            x1: threshold,
            y0: 0,
            y1: 1,
            yref: 'paper',
            line: {color: 'red', width: 2, dash: 'dash'}
        }]
    };
    
    Plotly.newPlot('errorHistogram', data, layout, {responsive: true});
}

// Smooth scrolling for navigation
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({behavior: 'smooth'});
        }
    });
});

console.log('🚀 CANShield IDS ready');

