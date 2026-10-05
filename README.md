# Neural Network X-Ray

Interactive CNN interpretability web app for handwritten OCR across Latin and Bengali characters.

Draw a character and watch each layer process it in real time: input pixels, convolutions, activations, pooling, dense features, and softmax output. The app also includes a training timeline (epoch checkpoints) and a Model Lab to design/train custom architectures.

## Live Features

- Real-time in-browser inference with ONNX Runtime Web (WASM)
- Per-layer visual sections explaining the full CNN pipeline
- 2D network view + optional 3D architecture view
- Epoch playback to inspect how predictions evolve during training
- Model Lab to build/train/export custom CNNs (browser, HF CPU, or GPU backend)
- Shareable canvas state via compressed URL hash

## Tech Stack

- Next.js 16 + React 19 + TypeScript
- Tailwind CSS 4 + Radix UI + Framer Motion
- Zustand for global state
- ONNX Runtime Web for inference
- TensorFlow.js (Model Lab browser training)
- Three.js / React Three Fiber (3D view)

## Project Structure

- `app/page.tsx` — main long-form interactive page
- `components/sections/*` — scroll-driven explanatory sections
- `components/canvas/*` — drawing, upload, and network canvases
- `components/model-lab/*` — architecture builder, training controls, exports
- `lib/model/*` — preprocess, model loading, inference, epoch checkpoints
- `stores/*` — inference/UI/model-lab Zustand stores

## Run Locally

Install dependencies:

```bash
pnpm install
```

Start dev server:

```bash
pnpm dev
```

Open `http://localhost:3000`.

## Scripts

```bash
pnpm dev
pnpm lint
pnpm build
pnpm start
```

## Model Assets

Primary ONNX model is loaded from:

- `public/models/combined-cnn/model.onnx`

Training visual data is loaded from:

- `public/training/history.json`
- `public/training/weight-snapshots.json`

Browser training subsets:

- `public/data/emnist-subset.bin`
- `public/data/bangla-subset.bin`

## Environment Variables

- `NEXT_PUBLIC_MODEL_BASE_URL` (optional): custom base URL for epoch checkpoint models
- `MODAL_ENDPOINT_URL` (optional): backend endpoint used by `/api/gpu-train`

## Training / Data Scripts

Useful scripts in `scripts/`:

- `train_combined.py` — full EMNIST+Bangla training/export pipeline
- `prepare_browser_data.py` — creates compact `.bin` datasets for browser training
- `generate_demo_data.py` — synthetic fallback dataset generation

## Notes

- Inference is client-side in the browser.
- Model checkpoints are prefetched to keep epoch scrubbing responsive.
- Some heavy visual sections are deferred until near viewport for smoother load.
