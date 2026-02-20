# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
pnpm dev          # Start dev server (localhost:3000)
pnpm build        # Production build (Turbopack)
pnpm lint         # ESLint
pnpm start        # Run production server
```

**Do NOT run `pnpm build` after every change** — only when explicitly asked or verifying critical changes.

## Architecture

### Stack
Next.js 16.1.6, React 19, TypeScript, Tailwind CSS 4 (@theme inline in globals.css), Zustand 5, Framer Motion 12, ONNX Runtime Web (WASM, single-threaded), Three.js + R3F for optional 3D view.

### Inference Pipeline
1. **Draw** → `DrawingCanvas` captures strokes on 280×280 internal canvas
2. **Preprocess** (`lib/model/preprocess.ts`) → grayscale, bilinear resize to 28×28, transpose for EMNIST format, output Float32Array [1,1,28,28]
3. **Infer** (`lib/model/predict.ts`) → ONNX model at `/public/models/emnist-cnn/model.onnx` extracts 11 intermediate layer activations + softmax output (62 classes, 47 valid — ByMerge masks 15 untrained indices)
4. **Store** → `inferenceStore` holds activations, prediction, input tensor globally
5. **Debounce** → `useInference` hook debounces at 150ms

### Hero Stage State Machine (`uiStore.heroStage`)
- `"drawing"` → full-size centered canvas with hero text
- `"shrinking"` → triggered on first draw, canvas animates to floating position
- `"revealed"` → neuron network visualization appears, canvas is draggable

### Page Structure (`app/page.tsx`)
`NeuronNetworkSection` renders first (contains hero + network canvas + floating input). Nine more sections follow, each wrapped in `LazySection` (deferred via IntersectionObserver, 300px rootMargin) + `SectionWrapper` (consistent padding/fade-in).

### Scroll Tracking (`hooks/useScrollSection.ts`)
Scroll listener checks `getBoundingClientRect().top` for each section ID against 40% viewport threshold. Updates `uiStore.activeSection` for the header nav indicator.

### NeuronNetworkSection (~1400 lines)
The most complex component. Key optimizations:
- **No React state in RAF loop** — hover state + wave progress stored in refs
- **Struct-of-arrays** for connections (flat Uint8Array, cache-friendly)
- **8 alpha buckets** to batch draw calls
- **Sqrt-compressed activation normalization** across all layers

### Epoch Checkpoints
50 epoch models hosted on HuggingFace (`soyeb-jim285/ocr-visualization-models`). Loaded by `lib/model/epochModels.ts` with prefetch strategy (±3 around current + background batch of all 50).

## Stores

**`inferenceStore`**: inputTensor, layerActivations, prediction, topPrediction, selectedNeuron, isInferring
**`uiStore`**: viewMode (2d/3d), activeSection, scrollProgress, modelLoaded, heroStage

## Theming (globals.css)

Tailwind v4 with CSS variables. Dark theme: background `#0a0a0f`, surface `#13131a`, foreground `#e8e8ed`. Accents: indigo `#6366f1`, purple `#8b5cf6`, cyan `#06b6d4`. Fonts: Geist Sans + Geist Mono.

## Key Patterns
- Model sessions cached as singletons (`loadModel.ts`, `epochModels.ts`)
- Use `getState()` not subscriptions in non-React contexts (e.g., `ImageUploader`)
- Framer Motion springs for canvas transitions (stiffness 260, damping 28, mass 0.6)
- Canvas elements use `image-rendering: pixelated` for 28×28 grid display
- CORS headers in `next.config.ts` enable WASM multi-threading (`Cross-Origin-Opener-Policy: same-origin`)
