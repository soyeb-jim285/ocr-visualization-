---
title: OCR Model Lab - GPU Training
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
---

GPU training backend for [ocr-visualization](https://github.com/soyeb-jim285/ocr-visualization).

Accepts a CNN architecture config, trains on the [ocr-handwriting-data](https://huggingface.co/datasets/soyeb-jim285/ocr-handwriting-data) dataset, and returns epoch metrics + a multi-output ONNX model.
