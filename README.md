# Real-Time CNN Visualizer (WebGPU / WASM ONNX)

An interactive, in-browser platform for **real-time visualization of Convolutional Neural Networks (CNNs)**. It connects to your webcam, extracts intermediate convolutional layer activation maps at up to **60 FPS**, and renders **Grad-CAM attention heatmaps** and classification predictions—**100% client-side with zero dedicated server required**.

Originally built as a Python/PyQt/TensorFlow application, this project is now ported to modern **ONNX Runtime Web** with **WebGPU** hardware acceleration and **WebAssembly SIMD** fallback.

🌐 **Live Demo (GitHub Pages):** [https://martianunlimited.github.io/realtime_cnn_visualizer/](https://martianunlimited.github.io/realtime_cnn_visualizer/)

---

## 🌟 Key Features

1. **In-Browser WebGPU & WebAssembly ONNX Execution:**
   - Powered by Microsoft's `onnxruntime-web`.
   - Runs neural network inference directly on the client's GPU via WebGPU shaders (delivering 30–60 FPS on standard laptops/desktops).
   - Graceful fallback to multi-threaded WebAssembly SIMD for browsers without WebGPU.
   - **Zero server / 100% private**: No video or camera frames are ever sent over the network.

2. **Multi-Layer Convolutional Activation Maps:**
   - Visualizes the hierarchical representations across 5 distinct CNN stages:
     - **Layer 1 (Stem):** Low-level edges, color contrasts, and spatial gradients ($112 \times 112$).
     - **Layer 2 (Block 1):** Corners, textures, and simple geometric shapes ($56 \times 56$).
     - **Layer 3 (Block 2):** Repeating patterns, motifs, and textures ($28 \times 28$).
     - **Layer 4 (Block 3):** Object parts, boundaries, and components ($14 \times 14$).
     - **Layer 5 (Deep Block):** High-level semantic features ($7 \times 7$).
   - **Interactive Montage Grid:** Displays channels in an interactive matrix (like `mapsToGrid`). Click any channel to zoom into it!
   - **Single Channel Inspector:** Enlarged view with live statistics (Min, Max, Mean, Sparsity % of zero activations).
   - **Multiple Colormaps:** Jet (MATLAB/OpenCV standard), Turbo (Google uniform), Viridis, Inferno, Neon Cyan, and Monochrome.

3. **Real-Time Grad-CAM Attention:**
   - Overlays class-activation heatmaps directly onto the live webcam video with adjustable opacity.
   - Highlights exactly where the CNN is focusing to recognize objects.

4. **Caltech-256 Classification:**
   - Live top-5 predictions with animated confidence bars and percentage scores.

5. **Flexible Input Pipeline:**
   - Live Webcam stream with front/back camera switching and pause/resume.
   - Pre-loaded benchmark sample images (Cat & Dog, etc.) for testing without a camera.
   - Drag-and-drop or file upload for custom photos and videos.

---

## 📦 Solving the 25MB Model Hosting Challenge

Original Keras / TensorFlow `.h5` ResNet50 models in FP32 format are ~98 MB, which exceeds GitHub's 25MB web upload limit and requires Git LFS. We provide three robust solutions:

### Solution 1: INT8 Dynamic Quantization (<25MB - Bundled Directly in Repo!)
We quantized both MobileNetV2 and ResNet50 to INT8 using ONNX Runtime quantization:
- **MobileNetV2 INT8:** **`2.59 MB`** (loads in <0.5s, 60 FPS on any laptop or phone).
- **ResNet50 INT8:** **`23.08 MB`** (**strictly under the 25MB GitHub limit!**).

Both quantized models are included directly in the `models/` directory and can be committed and hosted on GitHub Pages with zero external dependencies!

### Solution 2: GitHub Releases (Up to 2GB, Free Global CDN)
For full FP32 models (e.g. `resnet50_caltech256.onnx`, 91.6 MB):
1. In your GitHub repository, click **Releases** → **Draft a new release**.
2. Tag it (e.g., `v1.0.0`) and attach `resnet50_caltech256.onnx` (GitHub allows up to 2 GB per asset).
3. Copy the download URL:
   `https://github.com/martianunlimited/realtime_cnn_visualizer/releases/download/v1.0.0/resnet50_caltech256.onnx`
4. In the web app, choose **Load from Remote URL** and paste this link!

### Solution 3: Hugging Face Model Hub (Industry Standard for Web ML)
1. Upload your `.onnx` model to a free Hugging Face Model repo:
   `https://huggingface.co/martianunlimited/realtime-cnn-caltech256`
2. Direct download URL with built-in CORS:
   `https://huggingface.co/martianunlimited/realtime-cnn-caltech256/resolve/main/resnet50_caltech256.onnx`
3. Paste the URL into the app's remote loader.

### Solution 4: Local File Picker (Zero Network Upload)
Users can click **Load Local .ONNX File...** and select any `.onnx` model on their computer. It loads straight into browser memory in milliseconds with zero network transfer!

---

## 🛠️ Exporting Models from PyTorch / Keras

We have replaced legacy TensorFlow 1.x with modern PyTorch 2.x and standard ONNX tools:

```bash
# Run the included model exporter
python export_models.py
```

This exports:
- `models/mobilenetv2_caltech256.onnx` (9.7 MB) & `_int8.onnx` (2.59 MB)
- `models/resnet50_caltech256.onnx` (91.6 MB) & `_int8.onnx` (23.08 MB)

All models include multi-layer activation taps: `act_layer1`, `act_layer2`, `act_layer3`, `act_layer4`, `act_layer5`, and classification `logits`.

---

## 🚀 Running Locally

```bash
# Start a local static server
python -m http.server 8000
```
Then open `http://localhost:8000` in your browser.

---

## 🌐 Deploying to GitHub Pages

1. Push this repository to GitHub:
   ```bash
   git add .
   git commit -m "feat: WebGPU in-browser CNN visualizer with multi-layer activations and Grad-CAM"
   git push origin main
   ```
2. In your GitHub repository:
   - Go to **Settings** → **Pages**.
   - Under **Build and deployment** → **Source**, select **Deploy from a branch**.
   - Select branch **`main`** and folder **`/ (root)`**, then click **Save**.
3. Your web visualizer will be live at:
   `https://martianunlimited.github.io/realtime_cnn_visualizer/`!

---

## 📜 Legacy PyQt / Python Docker Setup

The original desktop Python Qt implementation is preserved in:
- `main.py` / `main_v2.py`
- `Dockerfile` / `Dockerfile.cpu`
- `docker-compose.yml`
