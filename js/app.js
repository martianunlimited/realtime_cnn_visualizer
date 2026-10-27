/**
 * Real-Time CNN Visualizer - Main Application Orchestrator
 * Connects Webcam, executes ONNX Runtime Web via WebGPU/WASM,
 * and renders:
 *  - 8 Rows × 5 Columns Convolutional Feature Hierarchy (deeper layers from left to right)
 *  - Final Dense Layer 256-Neuron Activation Matrix & Softmax Class Probabilities
 *  - Grad-CAM Attention Heatmap & Single Kernel Zoom Inspector
 */

import { CNNInferenceEngine } from './inference.js';
import { COLORMAPS, getRGB } from './colormaps.js';

export const LAYER_KEYS = ['act_layer1', 'act_layer2', 'act_layer3', 'act_layer4', 'act_layer5'];
export const LAYER_TITLES = [
  'Layer 1 (Stem)',
  'Layer 2 (Block 1)',
  'Layer 3 (Block 2)',
  'Layer 4 (Block 3)',
  'Layer 5 (Deep)'
];

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

    // Model & Stream State
    this.selectedModelKey = 'RESNET18_INT8';
    this.colormap = COLORMAPS.JET;
    this.showGradCam = true;
    this.gradCamAlpha = 0.55;
    this.isPaused = false;
    this.useWebcam = true;
    this.activeMediaElement = null;
    this.mediaStream = null;

    // 8 Rows × 5 Columns Grid State
    this.gridCols = 5;
    this.gridRows = 8;
    this.bankOffset = 0; // Starts at channel 0 (channels 0..7)
    this.selectedCol = 0; // Selected column in 8x5 grid (0..4)
    this.selectedRow = 0; // Selected row in 8x5 grid (0..7)

    // Dense Layer State
    this.hoverDenseIndex = -1;
    this.lastDenseLayer = null;
    this.lastActivations = null;

    // Performance tracking
    this.fps = 0;
    this.frameCount = 0;
    this.fpsUpdateTime = performance.now();

    // Cached DOM
    this.dom = {};
  }

  async init() {
    this.cacheDOMElements();
    this.bindEvents();

    // 1. Load Caltech-256 class labels
    await this.engine.loadLabels('caltech_labels.json');

    // 2. Load default fine-tuned ResNet-18 model (~11MB)
    const initialConfig = MODEL_CONFIGS[this.selectedModelKey] || MODEL_CONFIGS.RESNET18_INT8;
    try {
      await this.engine.loadModel(initialConfig.url);
    } catch (err) {
      console.warn('Initial model load failed, fallback available:', err);
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

      // Single Kernel Zoom Inspector
      singleChannelCanvas: document.getElementById('single-channel-canvas'),
      zoomKernelLayerDisplay: document.getElementById('zoom-kernel-layer-display'),
      zoomKernelDims: document.getElementById('zoom-kernel-dims'),
      statChannelMin: document.getElementById('stat-channel-min'),
      statChannelMax: document.getElementById('stat-channel-max'),
      statChannelMean: document.getElementById('stat-channel-mean'),
      statChannelSparsity: document.getElementById('stat-channel-sparsity'),

      // 8x5 Convolutional Grid
      convGridCanvas: document.getElementById('conv-grid-canvas'),
      colormapSelect: document.getElementById('colormap-select'),
      btnBankPrev: document.getElementById('btn-bank-prev'),
      btnBankNext: document.getElementById('btn-bank-next'),
      bankLabel: document.getElementById('bank-label'),

      // Final Dense Layer
      denseMatrixCanvas: document.getElementById('dense-matrix-canvas'),
      denseNeuronHover: document.getElementById('dense-neuron-hover'),
      topClassDisplay: document.getElementById('top-class-display'),
      metricTopConf: document.getElementById('metric-top-conf'),
      metricMargin: document.getElementById('metric-margin'),
      metricMaxLogit: document.getElementById('metric-max-logit'),
      metricMinLogit: document.getElementById('metric-min-logit'),
      predictionsList: document.getElementById('predictions-list')
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

    // Bank Stepper Controls (Previous & Next Bank of 8 channels)
    if (this.dom.btnBankPrev) {
      this.dom.btnBankPrev.addEventListener('click', () => {
        this.bankOffset = Math.max(0, this.bankOffset - 8);
        this.updateBankLabel();
      });
    }

    if (this.dom.btnBankNext) {
      this.dom.btnBankNext.addEventListener('click', () => {
        this.bankOffset = Math.min(56, this.bankOffset + 8);
        this.updateBankLabel();
      });
    }

    // 8x5 Conv Grid Click Selection
    if (this.dom.convGridCanvas) {
      this.dom.convGridCanvas.addEventListener('click', (e) => {
        const rect = this.dom.convGridCanvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        const col = Math.floor((x / rect.width) * this.gridCols);
        const row = Math.floor((y / rect.height) * this.gridRows);

        if (col >= 0 && col < this.gridCols && row >= 0 && row < this.gridRows) {
          this.selectedCol = col;
          this.selectedRow = row;
          if (this.lastActivations) {
            this.renderSingleKernelZoom(this.lastActivations);
          }
        }
      });
    }

    // Final Dense Layer Matrix Hover & Mouseleave
    if (this.dom.denseMatrixCanvas) {
      this.dom.denseMatrixCanvas.addEventListener('mousemove', (e) => {
        const rect = this.dom.denseMatrixCanvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        const col = Math.floor((x / rect.width) * 16);
        const row = Math.floor((y / rect.height) * 16);
        const idx = row * 16 + col;

        if (idx >= 0 && idx < 256) {
          this.hoverDenseIndex = idx;
          this.updateDenseHoverReadout(idx);
        }
      });

      this.dom.denseMatrixCanvas.addEventListener('mouseleave', () => {
        this.hoverDenseIndex = -1;
        if (this.dom.denseNeuronHover) {
          this.dom.denseNeuronHover.textContent = 'Hover over any neuron to inspect class probability & logit';
        }
      });
    }
  }

  updateBankLabel() {
    if (this.dom.bankLabel) {
      const bankNum = Math.floor(this.bankOffset / 8) + 1;
      this.dom.bankLabel.textContent = `Showing Channels ${this.bankOffset} – ${this.bankOffset + 7} (Bank ${bankNum} of 8)`;
    }
  }

  updateDenseHoverReadout(classId) {
    if (!this.dom.denseNeuronHover) return;

    const className = (this.engine.labels && this.engine.labels[classId]) || `Class #${classId}`;
    let probStr = '--%';
    let logitStr = '--';

    if (this.lastDenseLayer && this.lastDenseLayer.probabilities) {
      const p = this.lastDenseLayer.probabilities[classId];
      const logit = this.lastDenseLayer.logits[classId];
      probStr = `${(p * 100).toFixed(2)}%`;
      logitStr = logit !== undefined ? logit.toFixed(2) : '--';
    }

    this.dom.denseNeuronHover.innerHTML = `
      <strong>Neuron #${classId} (${className})</strong>: Prob <strong>${probStr}</strong> • Logit: <strong>${logitStr}</strong>
    `;
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
        this.lastDenseLayer = results.denseLayer;

        // 1. Render 8 Rows × 5 Columns Convolutional Feature Grid
        this.renderConvGrid(results.activations);

        // 2. Render Single Kernel Zoom Inspector
        this.renderSingleKernelZoom(results.activations);

        // 3. Render Final Dense Layer & Top Class Probabilities
        this.renderDenseLayer(results.denseLayer);

        // 4. Render Grad-CAM Heatmap Overlay
        this.renderGradCamOverlay(results.gradCamHeatmap);

        // 5. Update Latency & FPS Performance Metrics
        this.updatePerformanceMetrics(results.timings);
      } catch (err) {
        console.error('Inference render error:', err);
      }
    }

    requestAnimationFrame(this.renderLoop.bind(this));
  }

  /**
   * Render 8 Rows × 5 Columns Convolutional Feature Hierarchy Canvas
   * Columns 0..4 = Progressively deeper layers (Layer 1 Stem -> Layer 5 Deep)
   * Rows 0..7    = 8 feature channels (bankOffset + row)
   */
  renderConvGrid(activations) {
    const canvas = this.dom.convGridCanvas;
    if (!canvas || !activations) return;

    const ctx = canvas.getContext('2d');
    const cols = this.gridCols; // 5
    const rows = this.gridRows; // 8
    const cellW = canvas.width / cols;
    const cellH = canvas.height / rows;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (let c = 0; c < cols; c++) {
      const layerKey = LAYER_KEYS[c];
      const layer = activations[layerKey];
      if (!layer) continue;

      const planeSize = layer.width * layer.height;

      for (let r = 0; r < rows; r++) {
        const channelIdx = (this.bankOffset + r) % layer.channels;
        const channelOffset = channelIdx * planeSize;

        const cellX = c * cellW;
        const cellY = r * cellH;

        // Find min and max for local normalization
        let minVal = Infinity;
        let maxVal = -Infinity;
        for (let i = 0; i < planeSize; i++) {
          const v = layer.data[channelOffset + i];
          if (v < minVal) minVal = v;
          if (v > maxVal) maxVal = v;
        }
        const range = maxVal - minVal > 1e-4 ? maxVal - minVal : 1.0;

        // Draw feature map pixels
        const pW = cellW / layer.width;
        const pH = cellH / layer.height;

        for (let py = 0; py < layer.height; py++) {
          for (let px = 0; px < layer.width; px++) {
            const raw = layer.data[channelOffset + py * layer.width + px];
            const norm = (raw - minVal) / range;
            const [red, green, blue] = getRGB(norm, this.colormap);
            ctx.fillStyle = `rgb(${red}, ${green}, ${blue})`;
            ctx.fillRect(cellX + px * pW, cellY + py * pH, pW + 0.5, pH + 0.5);
          }
        }

        // Selected Cell Highlight (Luminous Cyan border)
        const isSelected = c === this.selectedCol && r === this.selectedRow;
        if (isSelected) {
          ctx.strokeStyle = '#00f5d4';
          ctx.lineWidth = 3;
          ctx.strokeRect(cellX + 1.5, cellY + 1.5, cellW - 3, cellH - 3);

          // Subtle inner glow
          ctx.fillStyle = 'rgba(0, 245, 212, 0.15)';
          ctx.fillRect(cellX, cellY, cellW, cellH);
        } else {
          ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
          ctx.lineWidth = 1;
          ctx.strokeRect(cellX + 0.5, cellY + 0.5, cellW - 1, cellH - 1);
        }

        // Draw Channel Badge in top-left
        ctx.fillStyle = 'rgba(0, 0, 0, 0.65)';
        ctx.fillRect(cellX + 3, cellY + 3, 24, 12);
        ctx.fillStyle = isSelected ? '#00f5d4' : '#cbd5e1';
        ctx.font = '600 9px monospace';
        ctx.fillText(`#${channelIdx}`, cellX + 5, cellY + 12);
      }
    }
  }

  /**
   * Render Selected Kernel in the Single Kernel Zoom Inspector Panel
   */
  renderSingleKernelZoom(activations) {
    const canvas = this.dom.singleChannelCanvas;
    if (!canvas || !activations) return;

    const layerKey = LAYER_KEYS[this.selectedCol];
    const layer = activations[layerKey];
    if (!layer) return;

    const channelIdx = (this.bankOffset + this.selectedRow) % layer.channels;
    const planeSize = layer.width * layer.height;
    const channelOffset = channelIdx * planeSize;

    // Update Header Displays
    if (this.dom.zoomKernelLayerDisplay) {
      this.dom.zoomKernelLayerDisplay.textContent = `${LAYER_TITLES[this.selectedCol]} • Ch #${channelIdx}`;
    }
    if (this.dom.zoomKernelDims) {
      this.dom.zoomKernelDims.textContent = `${layer.width}×${layer.height}`;
    }

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

    // Draw enlarged 224x224 kernel
    const ctx = canvas.getContext('2d');
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

    // Update Statistics Readouts
    if (this.dom.statChannelMin) this.dom.statChannelMin.textContent = minVal.toFixed(2);
    if (this.dom.statChannelMax) this.dom.statChannelMax.textContent = maxVal.toFixed(2);
    if (this.dom.statChannelMean) this.dom.statChannelMean.textContent = mean.toFixed(2);
    if (this.dom.statChannelSparsity) this.dom.statChannelSparsity.textContent = `${sparsity.toFixed(1)}%`;
  }

  /**
   * Render Final Dense Layer 256-Neuron Activation Matrix & Class Probabilities
   */
  renderDenseLayer(denseLayer) {
    if (!denseLayer) return;

    // 1. Draw 16×16 Output Neurons Heatmap
    const canvas = this.dom.denseMatrixCanvas;
    if (canvas && denseLayer.probabilities) {
      const ctx = canvas.getContext('2d');
      const probs = denseLayer.probabilities;
      const cellW = canvas.width / 16;
      const cellH = canvas.height / 16;

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const winnerId = denseLayer.topPredictions[0]?.classId ?? -1;

      for (let idx = 0; idx < 256; idx++) {
        const col = idx % 16;
        const row = Math.floor(idx / 16);
        const x = col * cellW;
        const y = row * cellH;

        const p = probs[idx] || 0.0;
        // Non-linear energy stretch: Math.pow(p, 0.45) highlights both subtle runner-ups and dominant peaks
        const energy = Math.min(1.0, Math.pow(p, 0.42));
        const [r, g, b] = getRGB(energy, this.colormap);

        ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
        ctx.fillRect(x + 1, y + 1, cellW - 2, cellH - 2);

        // Highlight Winning Neuron with glowing indicator
        if (idx === winnerId) {
          ctx.strokeStyle = '#00f5d4';
          ctx.lineWidth = 2.5;
          ctx.strokeRect(x + 1, y + 1, cellW - 2, cellH - 2);
        } else if (idx === this.hoverDenseIndex) {
          ctx.strokeStyle = '#ffffff';
          ctx.lineWidth = 1.5;
          ctx.strokeRect(x + 1, y + 1, cellW - 2, cellH - 2);
        } else {
          ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
          ctx.lineWidth = 1;
          ctx.strokeRect(x + 0.5, y + 0.5, cellW - 1, cellH - 1);
        }
      }
    }

    // 2. Update Decision Metrics
    if (denseLayer.topPredictions && denseLayer.topPredictions.length > 0) {
      const top = denseLayer.topPredictions[0];
      if (this.dom.topClassDisplay) {
        this.dom.topClassDisplay.textContent = `${top.className} (${(top.probability * 100).toFixed(1)}%)`;
      }
      if (this.dom.metricTopConf) {
        this.dom.metricTopConf.textContent = `${(top.probability * 100).toFixed(1)}%`;
      }
      if (this.dom.metricMargin) {
        this.dom.metricMargin.textContent = `+${(denseLayer.top1Margin * 100).toFixed(1)}%`;
      }
      if (this.dom.metricMaxLogit) {
        this.dom.metricMaxLogit.textContent = denseLayer.maxLogit.toFixed(1);
      }
      if (this.dom.metricMinLogit) {
        this.dom.metricMinLogit.textContent = denseLayer.minLogit.toFixed(1);
      }
    }

    // 3. Render Top-Ranked Prediction Bars
    this.renderPredictionsList(denseLayer.topPredictions);
  }

  renderPredictionsList(topPredictions) {
    if (!this.dom.predictionsList || !topPredictions || !topPredictions.length) return;

    this.dom.predictionsList.innerHTML = topPredictions.map((pred, idx) => {
      const percent = (pred.probability * 100).toFixed(1);
      const isTop = idx === 0;
      return `
        <div class="prediction-row ${isTop ? 'prediction-top' : ''}">
          <div class="pred-label-row">
            <span class="pred-rank">#${idx + 1}</span>
            <span class="pred-name" title="${pred.className} (Class ${pred.classId})">${pred.className}</span>
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
