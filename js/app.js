/**
 * Real-Time CNN Visualizer - Main Application Orchestrator
 * Connects Webcam, executes ONNX Runtime Web via WebGPU/WASM,
 * and renders interactive multi-layer feature maps and Grad-CAM overlays.
 */

import { CNNInferenceEngine } from './inference.js';
import { COLORMAPS, getRGB } from './colormaps.js';

export const MODEL_CONFIGS = {
  RESNET18_INT8: {
    id: 'resnet18_int8',
    name: 'ResNet18 (Caltech-256 Fine-Tuned, INT8 ~11MB - Recommended)',
    url: 'models/resnet18_caltech256_int8.onnx',
    description: 'Fine-tuned ResNet-18 on Caltech-256 with high top-1 accuracy. Compact ~11MB INT8 model with 5 multi-layer activation taps.'
  },
  MOBILENET_INT8: {
    id: 'mobilenetv2_int8',
    name: 'MobileNetV2 (Caltech-256, INT8 ~2.6MB - Fastest)',
    url: 'models/mobilenetv2_caltech256_int8.onnx',
    description: 'Ultra-lightweight 2.6MB model optimized for instant loading and 60 FPS WebGPU inference.'
  },
  RESNET50_INT8: {
    id: 'resnet50_int8',
    name: 'ResNet50 (Caltech-256, INT8 ~23MB - Deep Features)',
    url: 'models/resnet50_caltech256_int8.onnx',
    description: 'Full ResNet-50 23MB quantized model fitting under GitHub limits with 5 deep residual blocks.'
  }
};

export const SAMPLE_IMAGES = [
  { id: 'bear', name: '🐻 Bear (Wild Animal)', path: 'sample_images/bear.jpg' },
  { id: 'ak47', name: '🔫 AK-47 (Object)', path: 'sample_images/ak47.jpg' },
  { id: 'toad', name: '🐸 Toad (Amphibian)', path: 'sample_images/toad.jpg' },
  { id: 'american_flag', name: '🇺🇸 American Flag (Symbol)', path: 'sample_images/american_flag.jpg' },
  { id: 'greyhound', name: '🐕 Greyhound (Dog)', path: 'sample_images/greyhound.jpg' },
  { id: 'backpack', name: '🎒 Backpack (Accessory)', path: 'sample_images/backpack.jpg' }
];

class RealtimeCNNApp {
  constructor() {
    this.engine = new CNNInferenceEngine({
      onStatusChange: this.handleEngineStatus.bind(this)
    });

    // State
    this.selectedModelKey = 'RESNET18_INT8';
    this.activeLayerKey = 'act_layer1';
    this.selectedChannel = 0;
    this.colormap = COLORMAPS.JET;
    this.showGradCam = true;
    this.gradCamAlpha = 0.55;
    this.isPaused = false;
    this.useWebcam = true;
    this.activeMediaElement = null;
    this.mediaStream = null;

    // Performance tracking
    this.fps = 0;
    this.lastFrameTime = performance.now();
    this.frameCount = 0;
    this.fpsUpdateTime = performance.now();

    // Montage configuration
    this.montageCols = 6;
    this.montageRows = 6;

    // Cache DOM
    this.dom = {};
  }

  async init() {
    this.cacheDOMElements();
    this.bindEvents();

    // 1. Load Caltech-256 class labels
    await this.engine.loadLabels('caltech_labels.json');

    // 2. Load fine-tuned ResNet-18 model (~11MB)
    const initialConfig = MODEL_CONFIGS[this.selectedModelKey] || MODEL_CONFIGS.RESNET18_INT8;
    try {
      await this.engine.loadModel(initialConfig.url);
    } catch (err) {
      console.warn('Initial model load failed, prompt for file or fallback:', err);
    }

    // 3. Start Webcam
    await this.startWebcam();

    // 4. Start main animation & inference loop
    requestAnimationFrame(this.renderLoop.bind(this));
  }

  cacheDOMElements() {
    this.dom = {
      // Header & Performance
      backendBadge: document.getElementById('backend-badge'),
      fpsBadge: document.getElementById('fps-badge'),
      latencyBadge: document.getElementById('latency-badge'),
      statusBanner: document.getElementById('status-banner'),
      modelSelect: document.getElementById('model-select'),
      customFilePicker: document.getElementById('custom-model-file'),
      btnLoadUrl: document.getElementById('btn-load-url'),
      modelUrlInput: document.getElementById('model-url-input'),

      // Input / Video
      videoElement: document.getElementById('webcam-video'),
      sampleImgElement: document.getElementById('sample-image'),
      canvasGradCam: document.getElementById('gradcam-canvas'),
      btnToggleCam: document.getElementById('btn-toggle-cam'),
      btnPauseStream: document.getElementById('btn-pause-stream'),
      btnColorOrder: document.getElementById('btn-color-order'),
      sampleSelect: document.getElementById('sample-select'),
      customImgUpload: document.getElementById('custom-img-upload'),

      // Grad-CAM controls
      toggleGradCam: document.getElementById('toggle-gradcam'),
      sliderGradCamAlpha: document.getElementById('gradcam-alpha'),
      alphaDisplay: document.getElementById('alpha-display'),

      // Predictions
      predictionsList: document.getElementById('predictions-list'),
      topClassDisplay: document.getElementById('top-class-display'),

      // Layer tabs
      layerTabsContainer: document.getElementById('layer-tabs'),

      // Feature Map Montage
      montageCanvas: document.getElementById('montage-canvas'),
      montageTitle: document.getElementById('montage-title'),
      montageDimsBadge: document.getElementById('montage-dims-badge'),
      colormapSelect: document.getElementById('colormap-select'),

      // Single Channel Zoom Inspector
      singleChannelCanvas: document.getElementById('single-channel-canvas'),
      channelIndexSlider: document.getElementById('channel-index-slider'),
      channelIndexDisplay: document.getElementById('channel-index-display'),
      statChannelMin: document.getElementById('stat-channel-min'),
      statChannelMax: document.getElementById('stat-channel-max'),
      statChannelMean: document.getElementById('stat-channel-mean'),
      statChannelSparsity: document.getElementById('stat-channel-sparsity')
    };
  }

  bindEvents() {
    // Model Selection
    if (this.dom.modelSelect) {
      this.dom.modelSelect.addEventListener('change', async (e) => {
        const val = e.target.value;
        if (val === 'custom_file') {
          if (this.dom.customFilePicker) this.dom.customFilePicker.click();
        } else if (val === 'custom_url') {
          const urlModal = document.getElementById('url-modal');
          if (urlModal) urlModal.classList.remove('hidden');
        } else if (MODEL_CONFIGS[val]) {
          await this.loadModelFromConfig(MODEL_CONFIGS[val]);
        }
      });
    }

    // Custom File Picker
    if (this.dom.customFilePicker) {
      this.dom.customFilePicker.addEventListener('change', async (e) => {
        const file = e.target.files && e.target.files[0];
        if (!file) return;
        const arrayBuffer = await file.arrayBuffer();
        await this.engine.loadModel(arrayBuffer);
      });
    }

    // Custom URL modal
    if (this.dom.btnLoadUrl) {
      this.dom.btnLoadUrl.addEventListener('click', async () => {
        const url = this.dom.modelUrlInput.value.trim();
        if (url) {
          await this.engine.loadModel(url);
          const urlModal = document.getElementById('url-modal');
          if (urlModal) urlModal.classList.add('hidden');
        }
      });
    }

    // Camera Controls
    if (this.dom.btnToggleCam) {
      this.dom.btnToggleCam.addEventListener('click', () => {
        if (this.useWebcam) {
          this.stopWebcam();
          this.useWebcam = false;
          this.dom.btnToggleCam.innerHTML = '📹 Enable Camera';
          this.loadSampleImage(SAMPLE_IMAGES[0].path);
        } else {
          this.startWebcam();
          this.dom.btnToggleCam.innerHTML = '⏹️ Stop Camera';
        }
      });
    }

    if (this.dom.btnPauseStream) {
      this.dom.btnPauseStream.addEventListener('click', () => {
        this.isPaused = !this.isPaused;
        this.dom.btnPauseStream.innerHTML = this.isPaused ? '▶️ Resume' : '⏸️ Pause';
      });
    }

    // Color Format Toggle (RGB vs BGR)
    if (this.dom.btnColorOrder) {
      this.dom.btnColorOrder.addEventListener('click', () => {
        const newOrder = this.engine.colorOrder === 'RGB' ? 'BGR' : 'RGB';
        this.engine.setColorOrder(newOrder);
        this.dom.btnColorOrder.innerHTML = `🎨 Format: ${newOrder}`;
      });
    }

    // Sample Image Selector
    if (this.dom.sampleSelect) {
      this.dom.sampleSelect.addEventListener('change', (e) => {
        const path = e.target.value;
        if (path) {
          this.stopWebcam();
          this.useWebcam = false;
          if (this.dom.btnToggleCam) this.dom.btnToggleCam.innerHTML = '📹 Enable Camera';
          this.loadSampleImage(path);
        }
      });
    }

    // Custom Image Upload
    if (this.dom.customImgUpload) {
      this.dom.customImgUpload.addEventListener('change', (e) => {
        const file = e.target.files && e.target.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = (re) => {
          this.stopWebcam();
          this.useWebcam = false;
          if (this.dom.btnToggleCam) this.dom.btnToggleCam.innerHTML = '📹 Enable Camera';
          this.loadSampleImage(re.target.result);
        };
        reader.readAsDataURL(file);
      });
    }

    // Grad-CAM Controls
    if (this.dom.toggleGradCam) {
      this.dom.toggleGradCam.addEventListener('change', (e) => {
        this.showGradCam = e.target.checked;
      });
    }

    if (this.dom.sliderGradCamAlpha) {
      this.dom.sliderGradCamAlpha.addEventListener('input', (e) => {
        this.gradCamAlpha = parseFloat(e.target.value);
        if (this.dom.alphaDisplay) {
          this.dom.alphaDisplay.textContent = `${Math.round(this.gradCamAlpha * 100)}%`;
        }
      });
    }

    // Colormap Selector
    if (this.dom.colormapSelect) {
      this.dom.colormapSelect.addEventListener('change', (e) => {
        this.colormap = e.target.value;
      });
    }

    // Channel Index Slider
    if (this.dom.channelIndexSlider) {
      this.dom.channelIndexSlider.addEventListener('input', (e) => {
        this.selectedChannel = parseInt(e.target.value, 10);
        if (this.dom.channelIndexDisplay) {
          this.dom.channelIndexDisplay.textContent = `Channel #${this.selectedChannel}`;
        }
      });
    }

    // Clicking inside Montage Canvas selects channel!
    if (this.dom.montageCanvas) {
      this.dom.montageCanvas.addEventListener('click', (e) => {
        const rect = this.dom.montageCanvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        const col = Math.floor((x / rect.width) * this.montageCols);
        const row = Math.floor((y / rect.height) * this.montageRows);
        const clickedIdx = row * this.montageCols + col;
        
        if (this.lastActivations && this.lastActivations[this.activeLayerKey]) {
          const maxChannels = this.lastActivations[this.activeLayerKey].channels;
          if (clickedIdx < maxChannels) {
            this.selectedChannel = clickedIdx;
            if (this.dom.channelIndexSlider) this.dom.channelIndexSlider.value = clickedIdx;
            if (this.dom.channelIndexDisplay) this.dom.channelIndexDisplay.textContent = `Channel #${clickedIdx}`;
          }
        }
      });
    }
  }

  handleEngineStatus({ status, message, backend }) {
    if (this.dom.statusBanner) {
      this.dom.statusBanner.textContent = message;
      this.dom.statusBanner.className = `status-banner status-${status}`;
    }

    if (backend && this.dom.backendBadge) {
      if (backend === 'webgpu') {
        this.dom.backendBadge.textContent = '⚡ WebGPU Active';
        this.dom.backendBadge.className = 'badge-hardware badge-webgpu';
      } else {
        this.dom.backendBadge.textContent = '🧩 WASM SIMD Active';
        this.dom.backendBadge.className = 'badge-hardware badge-wasm';
      }
    }
  }

  async loadModelFromConfig(config) {
    try {
      await this.engine.loadModel(config.url);
    } catch (err) {
      console.error('Failed to load model:', err);
    }
  }

  async startWebcam() {
    try {
      this.stopWebcam();
      const constraints = {
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user'
        },
        audio: false
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      this.mediaStream = stream;
      this.dom.videoElement.srcObject = stream;
      await this.dom.videoElement.play();

      this.dom.videoElement.classList.remove('hidden');
      this.dom.sampleImgElement.classList.add('hidden');
      this.activeMediaElement = this.dom.videoElement;
      this.useWebcam = true;
    } catch (err) {
      console.warn('Webcam permission denied or unavailable, loading sample image:', err);
      this.useWebcam = false;
      if (this.dom.btnToggleCam) this.dom.btnToggleCam.innerHTML = '📹 Enable Camera';
      this.loadSampleImage(SAMPLE_IMAGES[0].path);
    }
  }

  stopWebcam() {
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach(track => track.stop());
      this.mediaStream = null;
    }
  }

  loadSampleImage(src) {
    this.dom.videoElement.classList.add('hidden');
    this.dom.sampleImgElement.classList.remove('hidden');
    this.dom.sampleImgElement.src = src;
    if (this.dom.sampleImgElement.complete && this.dom.sampleImgElement.naturalWidth > 0) {
      this.activeMediaElement = this.dom.sampleImgElement;
    }
    this.dom.sampleImgElement.onload = () => {
      this.activeMediaElement = this.dom.sampleImgElement;
    };
  }

  /**
   * Main Real-Time Inference & Rendering Loop
   */
  async renderLoop() {
    if (!this.isPaused && this.activeMediaElement && this.engine.session) {
      try {
        const results = await this.engine.runInference(this.activeMediaElement);
        this.lastActivations = results.activations;

        // 1. Render Top-5 Predictions
        this.renderPredictions(results.topPredictions);

        // 2. Render Grad-CAM Heatmap Overlay
        this.renderGradCamOverlay(results.gradCamHeatmap);

        // 3. Render Feature Map Montage & Single Channel
        this.renderActivationMaps(results.activations);

        // 4. Update Latency & FPS
        this.updatePerformanceMetrics(results.timings);
      } catch (err) {
        console.error('Inference render error:', err);
      }
    }

    requestAnimationFrame(this.renderLoop.bind(this));
  }

  renderPredictions(topPredictions) {
    if (!this.dom.predictionsList || !topPredictions.length) return;

    if (this.dom.topClassDisplay) {
      const top = topPredictions[0];
      this.dom.topClassDisplay.textContent = `${top.className} (${(top.probability * 100).toFixed(1)}%)`;
    }

    this.dom.predictionsList.innerHTML = topPredictions.map((pred, idx) => {
      const percent = (pred.probability * 100).toFixed(1);
      const isTop = idx === 0;
      return `
        <div class="prediction-row ${isTop ? 'prediction-top' : ''}">
          <div class="pred-label-row">
            <span class="pred-rank">#${idx + 1}</span>
            <span class="pred-name">${pred.className}</span>
            <span class="pred-pct">${percent}%</span>
          </div>
          <div class="pred-bar-track">
            <div class="pred-bar-fill" style="width: ${percent}%;"></div>
          </div>
        </div>
      `;
    }).join('');
  }

  renderGradCamOverlay(heatmap) {
    const canvas = this.dom.canvasGradCam;
    const media = this.activeMediaElement;
    if (!canvas || !media) return;

    const ctx = canvas.getContext('2d');
    const displayW = media.videoWidth || media.naturalWidth || canvas.width;
    const displayH = media.videoHeight || media.naturalHeight || canvas.height;

    if (canvas.width !== displayW || canvas.height !== displayH) {
      canvas.width = displayW;
      canvas.height = displayH;
    }

    ctx.clearRect(0, 0, displayW, displayH);

    if (!this.showGradCam || !heatmap) return;

    // Create offscreen image of the low-res heatmap (e.g. 7x7)
    const offCanvas = document.createElement('canvas');
    offCanvas.width = heatmap.width;
    offCanvas.height = heatmap.height;
    const offCtx = offCanvas.getContext('2d');
    const imgData = offCtx.createImageData(heatmap.width, heatmap.height);

    for (let i = 0; i < heatmap.data.length; i++) {
      const val = heatmap.data[i];
      const [r, g, b] = getRGB(val, this.colormap);
      imgData.data[i * 4] = r;
      imgData.data[i * 4 + 1] = g;
      imgData.data[i * 4 + 2] = b;
      imgData.data[i * 4 + 3] = Math.round(val * 255);
    }
    offCtx.putImageData(imgData, 0, 0);

    // Smoothly upscale heatmap onto video canvas with alpha
    ctx.save();
    ctx.globalAlpha = this.gradCamAlpha;
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(offCanvas, 0, 0, displayW, displayH);
    ctx.restore();
  }

  renderActivationMaps(activations) {
    if (!activations) return;

    // 1. Build layer tabs if not yet built
    this.ensureLayerTabs(activations);

    const layer = activations[this.activeLayerKey];
    if (!layer) return;

    // 2. Render Montage Grid
    this.renderMontage(layer);

    // 3. Render Single Channel Inspector
    this.renderSingleChannel(layer);
  }

  ensureLayerTabs(activations) {
    const container = this.dom.layerTabsContainer;
    if (!container || container.dataset.built === 'true') return;

    container.innerHTML = Object.keys(activations).map((key, idx) => {
      const layer = activations[key];
      const isActive = key === this.activeLayerKey;
      return `
        <button class="layer-tab ${isActive ? 'active' : ''}" data-layer="${key}" type="button">
          <span class="tab-badge">L${idx + 1}</span>
          <span>${layer.name.split('(')[0].trim()}</span>
          <span class="tab-sub">${layer.width}×${layer.height}</span>
        </button>
      `;
    }).join('');

    container.querySelectorAll('.layer-tab').forEach(btn => {
      btn.addEventListener('click', () => {
        container.querySelectorAll('.layer-tab').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        this.activeLayerKey = btn.dataset.layer;
        this.selectedChannel = 0;
        
        // Update slider max
        const layer = activations[this.activeLayerKey];
        if (layer && this.dom.channelIndexSlider) {
          this.dom.channelIndexSlider.max = layer.channels - 1;
          this.dom.channelIndexSlider.value = 0;
        }
      });
    });

    container.dataset.built = 'true';
  }

  renderMontage(layer) {
    const canvas = this.dom.montageCanvas;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const cols = this.montageCols;
    const rows = this.montageRows;
    const totalVisible = Math.min(layer.channels, cols * rows);

    const cellW = canvas.width / cols;
    const cellH = canvas.height / rows;
    const planeSize = layer.width * layer.height;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (let idx = 0; idx < totalVisible; idx++) {
      const col = idx % cols;
      const row = Math.floor(idx / cols);
      const x = col * cellW;
      const y = row * cellH;

      // Extract channel data
      const channelOffset = idx * planeSize;
      
      // Find min/max for local normalization
      let minVal = Infinity, maxVal = -Infinity;
      for (let i = 0; i < planeSize; i++) {
        const v = layer.data[channelOffset + i];
        if (v < minVal) minVal = v;
        if (v > maxVal) maxVal = v;
      }
      const range = maxVal - minVal > 1e-4 ? maxVal - minVal : 1.0;

      // Draw pixels to cell
      const pW = cellW / layer.width;
      const pH = cellH / layer.height;

      for (let py = 0; py < layer.height; py++) {
        for (let px = 0; px < layer.width; px++) {
          const raw = layer.data[channelOffset + py * layer.width + px];
          const norm = (raw - minVal) / range;
          const [r, g, b] = getRGB(norm, this.colormap);
          ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
          ctx.fillRect(x + px * pW, y + py * pH, pW + 0.5, pH + 0.5);
        }
      }

      // Border around active channel
      if (idx === this.selectedChannel) {
        ctx.strokeStyle = '#00f5d4';
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, cellW, cellH);
      } else {
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
        ctx.lineWidth = 1;
        ctx.strokeRect(x, y, cellW, cellH);
      }
    }

    if (this.dom.montageDimsBadge) {
      this.dom.montageDimsBadge.textContent = `${layer.channels} Ch • ${layer.width}×${layer.height}`;
    }
  }

  renderSingleChannel(layer) {
    const canvas = this.dom.singleChannelCanvas;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const ch = Math.min(this.selectedChannel, layer.channels - 1);
    const planeSize = layer.width * layer.height;
    const channelOffset = ch * planeSize;

    let minVal = Infinity, maxVal = -Infinity, sum = 0, zeroCount = 0;
    for (let i = 0; i < planeSize; i++) {
      const v = layer.data[channelOffset + i];
      if (v < minVal) minVal = v;
      if (v > maxVal) maxVal = v;
      sum += v;
      if (Math.abs(v) < 1e-4) zeroCount++;
    }
    const mean = sum / planeSize;
    const sparsity = (zeroCount / planeSize) * 100;
    const range = maxVal - minVal > 1e-4 ? maxVal - minVal : 1.0;

    // Draw magnified channel
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const pW = canvas.width / layer.width;
    const pH = canvas.height / layer.height;

    for (let py = 0; py < layer.height; py++) {
      for (let px = 0; px < layer.width; px++) {
        const raw = layer.data[channelOffset + py * layer.width + px];
        const norm = (raw - minVal) / range;
        const [r, g, b] = getRGB(norm, this.colormap);
        ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
        ctx.fillRect(px * pW, py * pH, pW + 0.5, pH + 0.5);
      }
    }

    // Update stats readouts
    if (this.dom.statChannelMin) this.dom.statChannelMin.textContent = minVal.toFixed(2);
    if (this.dom.statChannelMax) this.dom.statChannelMax.textContent = maxVal.toFixed(2);
    if (this.dom.statChannelMean) this.dom.statChannelMean.textContent = mean.toFixed(2);
    if (this.dom.statChannelSparsity) this.dom.statChannelSparsity.textContent = `${sparsity.toFixed(1)}%`;
  }

  updatePerformanceMetrics(timings) {
    const now = performance.now();
    this.frameCount++;

    if (now - this.fpsUpdateTime > 500) {
      this.fps = Math.round((this.frameCount * 1000) / (now - this.fpsUpdateTime));
      this.frameCount = 0;
      this.fpsUpdateTime = now;

      if (this.dom.fpsBadge) {
        this.dom.fpsBadge.textContent = `${this.fps} FPS`;
      }
    }

    if (this.dom.latencyBadge && timings) {
      this.dom.latencyBadge.textContent = `${timings.inferenceMs.toFixed(0)}ms infer (${timings.totalMs.toFixed(0)}ms total)`;
    }
  }
}

// Bootstrap
document.addEventListener('DOMContentLoaded', () => {
  const app = new RealtimeCNNApp();
  app.init();
  window.__cnnApp = app;
});
