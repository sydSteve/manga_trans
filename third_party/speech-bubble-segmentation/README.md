---
license: gpl-3.0
library_name: candle
base_model: kitsumed/yolov8m_seg-speech-bubble
tags:
- candle
- yolo
- image-segmentation
- comic
- manga
- speech-bubble
---

# mayocream/yolov8m_seg-speech-bubble

This repository contains a Candle-compatible `safetensors` conversion of
[`kitsumed/yolov8m_seg-speech-bubble`](https://huggingface.co/kitsumed/yolov8m_seg-speech-bubble).

Files:

- `model.safetensors`: converted floating-point checkpoint with the original Ultralytics tensor names
- `config.json`: Candle loader metadata for `koharu-ml`
- `config.yaml`: original upstream Ultralytics config

Model metadata:

- Variant: `YOLOv8m-seg`
- Input size: `640`
- Classes:
- `speech bubble`


Upstream revision: `da4efccf35a15c8a8c2564431a4b7e121d3e0d99`
