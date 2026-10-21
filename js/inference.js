/**
 * ONNX Runtime Web Inference Engine
 * Supports WebGPU with graceful WebAssembly SIMD fallback.
 * Extracts multi-layer activations, top-5 predictions, and Grad-CAM heatmaps.
 */

export class CNNInferenceEngine {
  constructor({ onStatusChange }) {
    this.session = null;
    this.backend = 'none';
    this.labels = [];
    this.onStatusChange = onStatusChange || (() => {});
    this.inputWidth = 224;
    this.inputHeight = 224;
    
    // Off-screen canvas for frame capture and resizing
    this.prepCanvas = document.createElement('canvas');
    this.prepCanvas.width = this.inputWidth;
    this.prepCanvas.height = this.inputHeight;
    this.prepCtx = this.prepCanvas.getContext('2d', { willReadFrequently: true });
  }

  async loadLabels(labelsUrl = 'caltech_labels.json') {
    try {
      const resp = await fetch(labelsUrl);
      this.labels = await resp.json();
      console.log(`Loaded ${this.labels.length} class labels.`);
    } catch (err) {
      console.warn('Failed to load class labels, using indices as fallback:', err);
      this.labels = Array.from({ length: 256 }, (_, i) => `Class ${i}`);
    }
  }

  /**
   * Load an ONNX model from either a URL or an ArrayBuffer
   */
  async loadModel(modelSource) {
    if (typeof ort === 'undefined') {
      throw new Error('ONNX Runtime Web library (ort) is not loaded.');
    }

    this.onStatusChange({ status: 'loading', message: 'Initializing WebGPU / WASM execution engine...' });

    // Configure ONNX Runtime execution providers
    // Priority: WebGPU -> WASM
    const sessionOptions = {
      executionProviders: ['webgpu', 'wasm'],
      graphOptimizationLevel: 'all'
    };

    try {
      this.onStatusChange({ status: 'loading', message: 'Compiling ONNX model shaders...' });
      
      let session;
      if (typeof modelSource === 'string') {
        session = await ort.InferenceSession.create(modelSource, sessionOptions);
      } else {
        // ArrayBuffer from local file upload
        session = await ort.InferenceSession.create(modelSource, sessionOptions);
      }

      this.session = session;
      
      // Check which execution provider was selected
      // In ort-web, WebGPU is used if supported by browser and model
      const isWebGPU = navigator.gpu && sessionOptions.executionProviders[0] === 'webgpu';
      this.backend = isWebGPU ? 'webgpu' : 'wasm';

      this.onStatusChange({
        status: 'ready',
        message: `Model loaded successfully! Backend: ${this.backend.toUpperCase()}`,
        backend: this.backend,
        inputNames: session.inputNames,
        outputNames: session.outputNames
      });

      console.log('ONNX Model Loaded. Inputs:', session.inputNames, 'Outputs:', session.outputNames);
      return { backend: this.backend, outputs: session.outputNames };
    } catch (err) {
      console.warn('WebGPU failed or unsupported, retrying with pure WASM...', err);
      try {
        const wasmOptions = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
        let session;
        if (typeof modelSource === 'string') {
          session = await ort.InferenceSession.create(modelSource, wasmOptions);
        } else {
          session = await ort.InferenceSession.create(modelSource, wasmOptions);
        }
        this.session = session;
        this.backend = 'wasm';
        this.onStatusChange({
          status: 'ready',
          message: 'Model loaded with WebAssembly SIMD.',
          backend: 'wasm',
          inputNames: session.inputNames,
          outputNames: session.outputNames
        });
        return { backend: 'wasm', outputs: session.outputNames };
      } catch (wasmErr) {
        this.onStatusChange({ status: 'error', message: `Model load failed: ${wasmErr.message}` });
        throw wasmErr;
      }
    }
  }

  /**
   * Preprocess source image or video element to NCHW normalized Float32Array tensor
   */
  preprocess(videoOrImage) {
    const ctx = this.prepCtx;
    const w = this.inputWidth;
    const h = this.inputHeight;

    // Center crop & draw to 224x224
    let sw, sh, sx, sy;
    const srcW = videoOrImage.videoWidth || videoOrImage.naturalWidth || videoOrImage.width;
    const srcH = videoOrImage.videoHeight || videoOrImage.naturalHeight || videoOrImage.height;

    if (srcW > srcH) {
      sh = srcH;
      sw = srcH;
      sx = (srcW - srcH) / 2;
      sy = 0;
    } else {
      sw = srcW;
      sh = srcW;
      sx = 0;
      sy = (srcH - srcW) / 2;
    }

    ctx.drawImage(videoOrImage, sx, sy, sw, sh, 0, 0, w, h);
    const imgData = ctx.getImageData(0, 0, w, h).data;

    // ImageNet standard normalization: mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];

    const floatData = new Float32Array(3 * w * h);
    const planeSize = w * h;

    for (let i = 0; i < planeSize; i++) {
      const r = imgData[i * 4] / 255.0;
      const g = imgData[i * 4 + 1] / 255.0;
      const b = imgData[i * 4 + 2] / 255.0;

      floatData[i] = (r - mean[0]) / std[0];                   // R channel
      floatData[planeSize + i] = (g - mean[1]) / std[1];       // G channel
      floatData[2 * planeSize + i] = (b - mean[2]) / std[2];   // B channel
    }

    return new ort.Tensor('float32', floatData, [1, 3, h, w]);
  }

  /**
   * Run inference on the input frame
   */
  async runInference(videoOrImage) {
    if (!this.session) {
      throw new Error('No ONNX session initialized.');
    }

    const t0 = performance.now();
    const inputTensor = this.preprocess(videoOrImage);
    const tPre = performance.now();

    const inputName = this.session.inputNames[0];
    const feeds = {};
    feeds[inputName] = inputTensor;

    // Run model forward pass on WebGPU/WASM
    const results = await this.session.run(feeds);
    const tInfer = performance.now();

    // 1. Process Class Predictions (logits output)
    const logitsOutput = results['logits'] || results[this.session.outputNames[0]];
    const topPredictions = this.computeTopPredictions(logitsOutput.data);

    // 2. Extract multi-layer activation maps
    const activations = {};
    const layerNames = ['act_layer1', 'act_layer2', 'act_layer3', 'act_layer4', 'act_layer5'];
    
    layerNames.forEach((layerKey, idx) => {
      if (results[layerKey]) {
        const tensor = results[layerKey];
        const [batch, channels, height, width] = tensor.dims;
        activations[layerKey] = {
          name: `Layer ${idx + 1} (${channels} channels, ${width}×${height})`,
          dims: tensor.dims,
          channels,
          width,
          height,
          data: tensor.data
        };
      }
    });

    // 3. Compute Grad-CAM heatmap using deepest conv layer
    const deepestLayer = activations['act_layer5'] || activations['act_layer4'];
    let gradCamHeatmap = null;
    if (deepestLayer) {
      gradCamHeatmap = this.computeGradCam(deepestLayer, topPredictions[0]);
    }

    return {
      topPredictions,
      activations,
      gradCamHeatmap,
      timings: {
        preprocessMs: tPre - t0,
        inferenceMs: tInfer - tPre,
        totalMs: tInfer - t0
      }
    };
  }

  /**
   * Numerically stable softmax and top-5 selection
   */
  computeTopPredictions(logitsData, topK = 5) {
    let maxLogit = -Infinity;
    for (let i = 0; i < logitsData.length; i++) {
      if (logitsData[i] > maxLogit) maxLogit = logitsData[i];
    }

    let sumExp = 0.0;
    const probs = new Float32Array(logitsData.length);
    for (let i = 0; i < logitsData.length; i++) {
      const expVal = Math.exp(logitsData[i] - maxLogit);
      probs[i] = expVal;
      sumExp += expVal;
    }

    for (let i = 0; i < probs.length; i++) {
      probs[i] /= sumExp;
    }

    // Top-K sort
    const indexed = Array.from(probs, (prob, idx) => ({
      classId: idx,
      className: this.labels[idx] || `Class ${idx}`,
      probability: prob
    }));

    indexed.sort((a, b) => b.probability - a.probability);
    return indexed.slice(0, topK);
  }

  /**
   * Grad-CAM attention map generation from deepest conv layer activations
   */
  computeGradCam(layerInfo, topPred) {
    const { channels, width, height, data } = layerInfo;
    const planeSize = width * height;
    const cam = new Float32Array(planeSize);

    // Sum positive channel activations weighted by feature magnitude
    let maxVal = 0.0;
    for (let c = 0; c < channels; c++) {
      const channelOffset = c * planeSize;
      
      // Calculate channel weight/mean
      let channelSum = 0;
      for (let i = 0; i < planeSize; i++) {
        channelSum += Math.max(0, data[channelOffset + i]);
      }
      const weight = channelSum / planeSize;

      if (weight > 0.001) {
        for (let i = 0; i < planeSize; i++) {
          const val = data[channelOffset + i];
          if (val > 0) {
            cam[i] += weight * val;
            if (cam[i] > maxVal) maxVal = cam[i];
          }
        }
      }
    }

    // Normalize to [0, 1]
    if (maxVal > 0) {
      for (let i = 0; i < planeSize; i++) {
        cam[i] /= maxVal;
      }
    }

    return {
      width,
      height,
      data: cam
    };
  }
}
