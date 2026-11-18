# Handwriting reader: small, quantized, runs in the browser

Reads handwritten **English, French and Arabic** from photos of notes, whiteboards and simple
forms, entirely on the device (phone or laptop). The images never leave it.

- **Try it:** the web demo in `web/`, deployed on Vercel. Take a photo; it finds the page and
  the lines, reads them and lets you fix the text.
- **Send text, not photos, to an AI assistant:** "Send the text" exports the reading with
  line positions, confidence and alternative readings, plus small crops of only the uncertain
  lines. That is about 900 tokens instead of about 1,500 for the photo, and nothing is stored.
- **History on the device:** readings are saved in the browser (text only, never photos), so
  you can reopen, fix, re-send, export or delete them. No server or account.
- **Teach it your handwriting:** correct a few pages, download the training set, and run
  `scripts/finetune.py`.
- **Small:** 1.3M-parameter joint model, 5 MB (1.4 MB int8). About 10 ms per line and about 1 s per
  photo in the browser.

**Good at:** handwritten notes, letters, whiteboards (about 5 to 14% character error on real
photos with the language model). **Not built for:** printed text, invoices and receipts, or
codes and IDs where every character matters. See "Results" below.

Code: MIT. Model weights: non-commercial research only, because of the training datasets. See
`MODELS_LICENSE.md`.

---

# Quantized OCR Model for Low CPU Usage

Handwritten text recognition (HTR) for English and Arabic, on line images or whole pages, built to run on a CPU after int8 quantization.

- **Recognizer:** CRNN with a CTC head.
  - `vgg` + BiLSTM: 4.1M params, the baseline and distillation teacher.
  - `lite` + depthwise-separable convs + temporal convs: 0.6M params, **0.7 MB int8, ~1.8 ms/line**.
- **Scripts:**
  - English (`latin`): trained on IAM.
  - Arabic: trained on KHATT. Right-to-left lines are mirrored, so the model reads in reading order, and digit and Latin runs are re-ordered back.
- **Pages:** a classical line segmenter (no neural net, ~50 ms/page) feeds the line recognizer.
- **Browser:** `web/` runs the same pipeline 100% client-side (ONNX Runtime Web + a JS port of preprocessing, decoding and segmentation, checked against Python). See `web/README.md`.
- **Deployment:** ONNX Runtime on CPU. Inference needs `numpy`, `pillow`, `onnxruntime`, `opencv-python-headless` and `rapidfuzz`, with no torch. Each `.onnx` file carries its charset, script and direction in its metadata.

> IAM and KHATT are licensed for non-commercial research. For commercial use, replace them with synthetic and self-collected data.

## Setup

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .
.venv/bin/python scripts/download_data.py iam khatt iam_pages khatt_pages   # ~650 MB into data/raw/
```

## Pipeline

```bash
# Train. Uses Apple MPS / CUDA if available.
.venv/bin/python scripts/train.py --out runs/crnn_lstm                                     # English teacher
# Light models. (Spell the flags out: zsh does not word-split a "$LITE" variable.)
.venv/bin/python scripts/train.py --out runs/lite_en --arch lite --separable --seq conv \
    --channels 16 32 64 128 --hidden 128 --teacher runs/crnn_lstm/best.pt              # distilled
.venv/bin/python scripts/train.py --out runs/lite_ar --arch lite --separable --seq conv \
    --channels 16 32 64 128 --hidden 128 --script arabic --data-dir data/raw/khatt --epochs 100
.venv/bin/python scripts/download_data.py mnist                                          # for number fields
.venv/bin/python scripts/train.py --out runs/lite_fields --arch lite --separable --seq conv \
    --channels 16 32 64 128 --hidden 128 --init runs/lite_en/best.pt --synth-fields 6000 --epochs 25 --lr 5e-4
# One joint English + French + Arabic model (needs data/words from scripts/build_word_bank.py)
.venv/bin/python scripts/train.py --out runs/joint --arch lite --separable --seq conv \
    --channels 16 32 64 128 --hidden 192 --script mixed --data-dir data/raw/iam_line data/raw/rimes data/raw/khatt \
    --synth-fields 6000 --synth-mixed 20000 --words-dir data/words --epochs 40

# Export, quantize, evaluate. Use --data-dir data/raw/khatt for the Arabic model.
.venv/bin/python scripts/export_onnx.py runs/lite_en/best.pt
.venv/bin/python scripts/quantize.py runs/lite_en/model_fp32.onnx --mode static
.venv/bin/python scripts/evaluate.py runs/lite_en/model_*.onnx --threads 1

# Whole pages: segmentation + recognition
.venv/bin/python scripts/eval_pages.py --set iam --model runs/lite_en/model_int8_static.onnx
sips -s format jpeg photo.heic --out photo.jpg   # macOS: convert iPhone HEIC first
.venv/bin/python scripts/recognize.py photo.jpg --model runs/lite_en/model_int8_static.onnx --debug-dir runs/demo
# Photos: the sheet is found and flattened, and lines come from a learned text detector
# (PP-OCRv4 DBNet, models/ppocr_det_v4.onnx); --method projection is the classic segmenter.

# Language model (character 6-gram from training transcripts) and domain word lists
.venv/bin/python scripts/build_lm.py --data-dir data/raw/iam_line --script latin --out models/lm_en.json
.venv/bin/python scripts/tune_lm.py runs/lite_en/model_int8_static.onnx --lm models/lm_en.json --data-dir data/raw/iam_line
.venv/bin/python scripts/recognize.py notes.jpg --model runs/joint/model_int8_dynamic.onnx \
    --lm models/lm_mixed.json --lexicon lexicons/cisco_ios.txt
```

## Invoices and forms (field extraction)

For a known layout (a "template": blank form image + named, typed field boxes):

1. **Register** the scan or photo to the template (ORB + RANSAC homography: undoes rotation, scale, shift, mild perspective).
2. **Subtract the template**: printed labels, boxes and rules are masked out of each field, leaving only the handwriting.
3. **Read with constrained decoding**: CTC prefix beam search that only keeps prefixes that can still match the field's pattern
   (`date`, `amount`, `number`, `digits`, `phone`, `code`, `name`, `text`) or a lexicon (e.g. your customer list).
   `check` fields (tick boxes) are decided from the amount of ink, without reading.
4. **Normalize** (ISO dates with a sliding century, decimal amounts with `1.250,00` / `1,250.00` handled) and
   **flag for review** when the reading is invalid or its confidence is under `--review-below`. Confidence is measured
   on the *value*: hypotheses that differ only in spacing (`11 / 05 / 2003`, `11/05/2003`) count together.

```bash
# 1. Propose field boxes from a blank form (scan or photo), then edit names/types in the JSON
.venv/bin/python scripts/propose_template.py blank_invoice.jpg --name my_invoice --out templates/
# 2. Extract
.venv/bin/python scripts/extract_fields.py scans/*.jpg --template templates/my_invoice.json \
    --model latin=runs/lite_fields/model_int8_static.onnx --out results.json
# Benchmarks: synthetic invoices (data/forms_synth, from scripts/make_forms.py) and line-level number fields
.venv/bin/python scripts/extract_fields.py data/forms_synth/*.jpg --template templates/invoice_demo.json \
    --model latin=runs/lite_fields/model_int8_static.onnx --truth data/forms_synth/truth.json
.venv/bin/python scripts/eval_fields.py runs/lite_en/model_int8_static.onnx runs/lite_fields/model_int8_static.onnx
```

IAM has almost no handwritten digits, so `runs/lite_fields` fine-tunes the light English model on IAM plus synthetic
number fields built from MNIST digits (`--synth-fields`, `src/htr/synth.py`). MNIST's test writers are held out for evaluation.

## Adapting to your own documents

- **Template from filled samples** (no blank form needed): `scripts/template_from_samples.py a.jpg b.jpg c.jpg --name my_form`.
  - It aligns the samples, keeps the lightest value per pixel (handwriting differs between copies, print does not), deskews, then proposes fields.
  - Proposals: empty boxes, rows of digit cells (one `digits` field), tick boxes (`check`), and solid or dotted write-on lines.
  - The result is a skeleton: name the fields and fix the types in the JSON.
  - Measured on CNAM with `--truth-template`: the real blank gives 12 of 17 fields; a scan-degraded blank gives 7–11; blanks recovered from 5–8 synthetic fills give 6–7.
- **Correction mode** (browser page): fix the text under each detected line, click "Add page to training set" for each page, then download a `.zip` of line images plus `labels.jsonl`.
- **Fine-tune on those corrections**: `scripts/finetune.py --base runs/lite_en/best.pt --data corrections.zip --out runs/ft_mine`.
  1. Holds out 20% of your lines.
  2. Cuts the rest into words and recombines them into about 20x more lines in the same handwriting.
  3. Mixes in replay lines from the original data.
  4. Adds any new characters to the output layer.
  5. Keeps the result only if it beats the base model on your held-out lines. Only epochs whose error on the original data rose by at most `--max-forgetting` (default 1.5 points) can be picked.

  It writes `best.pt`, `model_fp32.onnx` with its browser sidecar, and `report.json`.

`scripts/bench_arch.py` times candidate architectures on CPU before training them. Latency does not depend on the weights.

## Results (Apple M5, 1 CPU thread)

Sources: line-level test numbers are in `runs/results.jsonl` (`evaluate.py`), page numbers in `runs/page_results.jsonl`,
number fields in `runs/field_line_results.jsonl`, synthetic invoices in `runs/lite_fields/forms.json`, synthetic CNAM forms in
`runs/cnam_forms.jsonl`. Each row below is the latest entry for that model in those logs.

**English baseline (BiLSTM, 4.06M params, 60 epochs, IAM test = 2,915 lines):**

| Model | Size | CER | WER | ms/line p50 |
|---|---|---|---|---|
| fp32 | 16.4 MB | 6.54% | 21.0% | 13.8 |
| int8 dynamic | 4.2 MB | 6.51% | 21.0% | 20.1 |
| int8 static (LSTM stays fp32) | 12.0 MB | 6.58% | 21.1% | 16.2 |
| int8 hybrid | 4.1 MB | 6.57% | 21.1% | 18.0 |

Here int8 costs no accuracy but is *slower*, because the convolutional backbone takes 75% of the time and the int8 LSTM path is slow on ARM.

**Light models (0.6M params, trained overnight; int8 static is the one to deploy):**

| Model | Test set | Size | CER | WER | ms/line p50 |
|---|---|---|---|---|---|
| lite_en fp32 (distilled from the BiLSTM, 60 epochs) | IAM, 2,915 lines | 2.51 MB | 11.05% | 34.8% | 3.06 |
| lite_en int8 static | IAM | 0.72 MB | 11.23% | 35.5% | 1.75 |
| lite_ar fp32 (100 epochs) | KHATT, 1,004 lines | 2.49 MB | 9.00% | 20.0% | 3.36 |
| lite_ar int8 static | KHATT | 0.71 MB | 9.55% | 21.0% | 1.93 |
| lite_fields int8 static (lite_en + synthetic number fields) | IAM | 0.72 MB | 12.08% | 38.6% | 1.58 |

**Joint English + French + Arabic model** (`runs/joint`, 1.27M params, 142 characters, 40 epochs on IAM + RIMES + KHATT
train lines, plus synthetic number fields and synthetic code-switched lines; reads mixed lines in one pass, no script picker).
Int8 **dynamic** is the one to deploy: static int8 is faster but costs Arabic a point, because its activation ranges are
calibrated on one script (calibrating on all three, `--data-dir a b c`, was worse still with minmax).

| Test set | fp32 CER | int8 dynamic CER | int8 static CER | specialist int8 static |
|---|---|---|---|---|
| IAM (English), 2,915 lines | 12.34% | 12.53% | 12.86% | lite_en 11.23% |
| RIMES (French) | 8.47% | 8.71% | 8.96% | — |
| KHATT (Arabic), 1,004 lines | 10.20% | 10.43% | 11.22% | lite_ar 9.55% |
| Mixed EN/FR/AR lines (600, synthetic from test-split words) | 6.67% | 6.42% | 6.83% | — |
| Size / ms per line p50 | 5.17 MB / 5.1 | 1.35 MB / 3.4 | 1.39 MB / 2.2 | 0.72 MB / 1.8 |

The specialists stay about one point better on their own script; the joint model is the one to use when the language
of a line is not known in advance, and in the browser demo (default model).

**Whole pages (segmentation + recognition, int8 static unless noted):**

| Model | Page set | Page CER | ms/page |
|---|---|---|---|
| BiLSTM fp32 | IAM paragraphs (214) | 9.1% | 201 |
| lite_en | IAM paragraphs (214) | 15.7% | 81 |
| lite_ar | synthetic Arabic pages (50) | 19.3% | 52 |

**Page segmentation methods** (`eval_pages.py`, lite_en / lite_ar int8 static, greedy decoding):

| Page set | `db` (paper + text detector, default) | `projection` |
|---|---|---|
| IAM paragraphs, all 214 pages (with ruled-line handling) | 15.7% | 15.7% |
| IAM paragraphs, pages 51–214, before ruled-line handling (not used to tune the line grouping) | 16.2% | 15.7% |
| Synthetic Arabic pages: stacked KHATT test lines, a segmentation check, not a real-page CER (50) | 5.6% | 19.3% |
| Phone photo of notes with a diagram, a desk and a folder in view | 34 lines found | 2 lines found |
| Phone photo of a filled job sheet (labels, underlines, a table), handwritten values (`eval_photo.py`) | **14.9%** value CER, 64% word recall | 61.9%, 2% |
| Segmentation time (IAM, 1 thread) | 273 ms | 55 ms |

On clean scans the two are level. On photos only `db` works, so it is the default.

**Forms and ruled paper.** Underlines, table borders and box edges touch the handwriting. Unhandled, they glue words to the grid (cut letters, merged rows) and put label and value on one line. `find_rules` in `page.py` handles them:
- **What counts as a rule:** a straight run at least 3 word heights long horizontally (2.5 vertically), thin, and mostly unbroken. Short vertical edges count only as the sides of a box.
- **Removal:** rules are taken out of the ink and the line crops, keeping the pixels where a stroke crosses them.
- **Separators:** a vertical rule, or an underline that starts or ends between two pieces of text, splits them into separate fields. Detector regions that run across such a boundary are cut.

On the job sheet photo this took value CER from 51% to 14.9% and word recall from 4% to 64%. Pages without ruled lines are unaffected (0 of 12 IAM pages changed).

The remaining errors there come from the recognizer: this writer's capitals ("1euton" for "Newton"), and "£" read as "2".

The language model does not help on such forms: 16.2% with it vs 14.9% without. It "corrects" codes, e.g. "M40 2AB" to "40 LAB". Use it for running text, and use `extract_fields.py` field types for codes.

**Language model** (`tune_lm.py`: fusion weights picked on validation lines, then test; beam 16):

| Model | Test set | beam CER | + LM CER | beam WER | + LM WER | decode ms/line |
|---|---|---|---|---|---|---|
| lite_en int8 static, `lm_en` | IAM (2,915 lines) | 10.79% | **7.62%** | 34.2% | **20.9%** | 7.3 |
| joint int8 dynamic, `lm_mixed` | RIMES (French) | 8.43% | **4.65%** | 32.2% | **14.7%** | 8.7 |
| joint int8 dynamic, `lm_mixed` | IAM | 12.10% | **8.46%** | 38.3% | **23.5%** | 11.1 |

The models are trained only on training lines that share no 20-character passage with any validation or test line. KHATT and RIMES reuse the same texts across writers: 99% and 69% of their test lines share such a passage with training text. This leaves no clean Arabic text (57 KHATT lines), so there is no Arabic language model yet, and KHATT test CERs of the recognizers themselves are probably flattered by these repeats.

With a language model, confidence stays image-only: the language model picks the text, and a reading the image does not support is flagged for review. On the 40 synthetic CNAM forms (lite_fields, LM on name/text fields), fields exact goes from 66.8% to **70.7%** and names from 32.5% to 46.2%. At the 0.7 threshold, 52.8% of fields are auto-accepted, 97.2% of them correct (98.0% without the LM).

**Handwritten number fields** (line level, `eval_fields.py`, unseen MNIST test writers, exact match, constrained decoding):

| Model | date | amount | digits | phone |
|---|---|---|---|---|
| BiLSTM (IAM only) | 21% | 32% | 27% | 2% |
| lite_en (IAM only) | 8.4% | 18.0% | 5.2% | 0.0% |
| **lite_fields int8 static** | **97.6%** | **94.0%** | **91.6%** | **84.0%** |

Constrained decoding adds 2 to 5 points over greedy for lite_fields.

**Synthetic invoices** (`extract_fields.py`, 60 forms, 840 fields, exact match after normalization):

| Recognizer(s) | all fields | numbers | text | ms/form |
|---|---|---|---|---|
| BiLSTM | 30.1% | 1.7–36% | 37.9% | 167 |
| lite_en | 17.5% | 0–17% | 32.9% | 93 |
| **lite_fields** | **68.8%** | **81–88%** | 28.7% | 84 |
| lite_fields + BiLSTM for text (`--model latin:text=...`) | 71.5% | 81–88% | 37.9% | 108 |

Review threshold with lite_fields, all fields:

| Threshold | Auto-accepted | Correct when accepted |
|---|---|---|
| 0.5 | 61.4% | 87.4% |
| **0.7** (recommended) | 46.9% | 93.7% |
| 0.9 | 26.1% | 98.2% |

Text fields never reach that level, so keep a human in the loop for free text.

**CNAM form, 40 synthetic fills** (`scripts/fill_template.py`: held-out IAM/RIMES test writers and MNIST test digits on the
real blank, with scan effects; 680 fields):

| Recognizer(s) | all exact | tick boxes | dates | IDs / codes / phones | names | auto-accepted at 0.7, correct |
|---|---|---|---|---|---|---|
| lite_fields | 66.8% | 100% | 87.5% | 42–60% | 32.5% | **52.2%, 98.0%** |
| lite_fields + BiLSTM for names/text | 70.7% | 100% | 87.5% | 42–60% | 48.8% | 60.7%, 94.4% |
| joint int8 dynamic alone | 67.6% | 100% | 90.0% | 39–68% | 37.5% | 55.6%, 96.8% |

The BiLSTM reads more names but is overconfident about them, so its threshold would have to be set separately.

**Real form: CNAM "Bulletin de remboursement".** `templates/cnam_bulletin.json` (17 fields, built from the blank form)
on a filled copy: 16 of 17 values correct (the address reads `NO 12` for `N° 12`, since the charset has no `°`),
with lite_fields for numbers and the BiLSTM for names and text, 108 ms. Every auto-accepted field was right;
the date was correct but flagged, because the writer's upright slashes read as `I` (synthetic dates now include upright slashes).

**Architecture speed (untrained, `bench_arch.py`, 200 IAM lines):**

| Candidate | Params | int8 size | fp32 ms | int8 static ms |
|---|---|---|---|---|
| vgg + BiLSTM 256 (baseline) | 4.06M | 4.14 MB | 15.0 | 16.7 |
| lite + BiLSTM 128 | 1.03M | 1.08 MB | 5.4 | 5.2 |
| lite + conv 128 | 0.85M | 0.91 MB | 5.9 | 4.2 |
| **lite separable + conv 128** | **0.60M** | **0.68 MB** | **3.3** | **1.8** |
| lite separable + conv 192 | 1.35M | 1.42 MB | 6.1 | 3.1 |

**Page segmentation (exact line-count rate):**

| Page set | projection (default) | blobs |
|---|---|---|
| IAM test paragraphs (214) | **94.4%** | 74.3% |
| Synthetic Arabic pages from KHATT test lines (50) | **60.0%** | 40.0% |

Projection takes about 50 ms per page. The KHATT paragraph counts are unreliable as ground truth, because several writers copied the same texts.

## Layout

```
src/htr/
  charset.py     character set, CTC greedy decoding (index 0 = blank)
  text.py        per-script normalization, RTL run handling, display detokenization
  preprocess.py  line image -> (64, W) float array (mirrored for RTL); shared by train and inference
  data.py        line datasets, augmentation, width-bucketed batching, CTC feasibility filter
  model.py       CRNN: arch vgg|lite, seq lstm|conv, optional depthwise-separable convs
  export.py      ONNX export + int8 quantization (dynamic / static / hybrid)
  inference.py   torch-free ONNX Runtime recognizer
  segment.py     page -> lines (projection or multi-orientation ridge "blobs")
  metrics.py     CER / WER
  decode.py      CTC greedy + prefix beam search, optionally constrained (regex / lexicon), with confidence
  fields.py      field types, patterns, value normalization
  forms.py       templates, registration, template subtraction, FormExtractor
  synth.py       synthetic handwritten number fields from MNIST digits
scripts/         download_data, train, export_onnx, quantize, evaluate, bench_arch, eval_pages, recognize,
                 propose_template, make_forms, extract_fields, eval_fields
```

## Notes

- **Training speed on Apple Silicon:** MPS recompiles kernels for every new input shape, so batch widths are padded to a multiple of 256 (`--pad-to`). That made training about 3x faster.
- **Distillation:** teacher and student emit one frame per 4 input columns, so the student matches the teacher's per-frame distribution (KL divergence) alongside CTC. In practice it skips the early all-blank CTC plateau.
- **KHATT:** about 12% of training "lines" are diagrams or lists that are far too narrow for their label at 64 px height. `ctc_feasible()` drops them from training only.
- **ONNX Runtime 1.32 calibration bug:** static calibration raises "No data is collected" when the number of samples is a multiple of the flush size. `export._flush_chunk` avoids it.
- **Known limits:**
  - The segmenter assumes text flows in lines across regions. Free-form layouts (mixed columns, 45-degree lines, rotated words) need a learned line detector.
  - KHATT is Modern Standard Arabic from scanned forms. Dialect text and ballpoint photos are a domain gap.
    On a phone photo of Tunisian-dialect text (`IMG_2953`), lite_ar reads almost nothing, even though it reaches 9.6% CER on KHATT.
    A few hundred labelled lines in the target style would be needed to fine-tune it.
  - Text fields on forms: the light model is about 5 CER points behind the BiLSTM on prose.
    Route text fields to a bigger model with `--model latin:text=...` when latency allows.
