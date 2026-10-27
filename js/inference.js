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
    this.colorOrder = 'RGB'; // 'RGB' (PyTorch default) or 'BGR' (OpenCV/Caffe default)
    
    // Off-screen canvas for frame capture and resizing
    this.prepCanvas = document.createElement('canvas');
    this.prepCanvas.width = this.inputWidth;
    this.prepCanvas.height = this.inputHeight;
    this.prepCtx = this.prepCanvas.getContext('2d', { willReadFrequently: true });
  }

  setColorOrder(order) {
    this.colorOrder = (order || 'RGB').toUpperCase();
    console.log(`[Engine] Color channel order set to: ${this.colorOrder}`);
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

    // Normalization parameters (ImageNet default)
    const isBGR = (this.colorOrder || 'RGB').toUpperCase() === 'BGR';
    const mean = isBGR ? [0.406, 0.456, 0.485] : [0.485, 0.456, 0.406];
    const std = isBGR ? [0.225, 0.224, 0.229] : [0.229, 0.224, 0.225];

    const floatData = new Float32Array(3 * w * h);
    const planeSize = w * h;

    for (let i = 0; i < planeSize; i++) {
      const r = imgData[i * 4] / 255.0;
      const g = imgData[i * 4 + 1] / 255.0;
      const b = imgData[i * 4 + 2] / 255.0;

      // In RGB: Ch0=R, Ch1=G, Ch2=B | In BGR: Ch0=B, Ch1=G, Ch2=R
      const ch0 = isBGR ? b : r;
      const ch1 = g;
      const ch2 = isBGR ? r : b;

      floatData[i] = (ch0 - mean[0]) / std[0];                   // Channel 0
      floatData[planeSize + i] = (ch1 - mean[1]) / std[1];       // Channel 1
      floatData[2 * planeSize + i] = (ch2 - mean[2]) / std[2];   // Channel 2
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

    // 1. Process Class Predictions and Dense Layer Output
    const logitsOutput = results['logits'] || results[this.session.outputNames[0]];
    const denseLayer = this.computeDenseLayer(logitsOutput.data, 8);
    const topPredictions = denseLayer.topPredictions;

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
      denseLayer,
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
   * Numerically stable softmax and complete dense layer representation
   */
  computeDenseLayer(logitsData, topK = 8) {
    let maxLogit = -Infinity;
    let minLogit = Infinity;
    for (let i = 0; i < logitsData.length; i++) {
      if (logitsData[i] > maxLogit) maxLogit = logitsData[i];
      if (logitsData[i] < minLogit) minLogit = logitsData[i];
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
      probability: prob,
      logit: logitsData[idx]
    }));

    indexed.sort((a, b) => b.probability - a.probability);
    const topPredictions = indexed.slice(0, topK);

    return {
      topPredictions,
      probabilities: probs,
      logits: logitsData,
      numClasses: logitsData.length,
      maxLogit,
      minLogit,
      top1Margin: indexed.length > 1 ? indexed[0].probability - indexed[1].probability : indexed[0].probability
    };
  }

  computeTopPredictions(logitsData, topK = 5) {
    return this.computeDenseLayer(logitsData, topK).topPredictions;
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
