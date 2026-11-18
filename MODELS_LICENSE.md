# Models and data: licences

The **source code** in this repository is MIT-licensed (see `LICENSE`).

The **trained weights and language models** are a different matter: they were trained on, or
counted from, datasets that allow **non-commercial research use only**. Use them for
research, teaching and personal, non-commercial projects. For commercial use, retrain on data
you have the rights to (the training scripts are included).

| File(s) | Trained on / derived from | Terms |
|---|---|---|
| `web/models/en.onnx`, `fields.onnx` | IAM Handwriting Database (+ MNIST digits for `fields`) | IAM: non-commercial research, citation required (Marti & Bunke, 2002). MNIST: CC BY-SA 3.0 |
| `web/models/ar.onnx` | KHATT | Non-commercial research (Mahmoud et al., 2014) |
| `web/models/joint.onnx` | IAM, RIMES, KHATT, MNIST | Most restrictive of the above: non-commercial research |
| `web/models/lm_en.json`, `lm_mixed.json`, `models/lm_*.json` | Character counts from IAM / RIMES / KHATT training transcripts | Same as the datasets they come from |
| `web/models/ppocr_det_v4.onnx` | PaddleOCR PP-OCRv4 text detector, ONNX conversion by RapidOCR | Apache-2.0 (PaddlePaddle / RapidAI) |
| `web/vendor/*` (fetched at build time) | ONNX Runtime Web | MIT (Microsoft) |
| `lexicons/*.txt` | Written for this project | MIT |

No dataset images or transcripts are included in this repository; `scripts/download_data.py`
fetches them for training.
