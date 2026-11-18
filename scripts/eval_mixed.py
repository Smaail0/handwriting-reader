"""CER on 600 synthetic code-switched EN/FR/AR lines built from TEST-split words (data/words) and MNIST test digits.

  python scripts/eval_mixed.py runs/joint/model_fp32.onnx runs/joint/model_int8_dynamic.onnx
"""
import json
import sys
from pathlib import Path

import numpy as np

from htr.inference import OnnxRecognizer
from htr.metrics import cer
from htr.mixed import MixedLineGenerator, WordBank
from htr.synth import DigitBank
from htr.text import normalize

gen = MixedLineGenerator(WordBank(Path("data/words"), "test"), DigitBank(Path("data/raw/mnist/test.parquet")))
lines = [gen(np.random.default_rng((999, i))) for i in range(600)]
refs = [normalize(t, "mixed") for _, t in lines]
for m in sys.argv[1:]:
    rec = OnnxRecognizer(m)
    hyp = [normalize(rec(img), "mixed") for img, _ in lines]
    c = cer(refs, hyp)
    print(m, "mixed_test CER", round(c, 4), flush=True)
    with open("runs/results.jsonl", "a") as f:
        f.write(json.dumps({"model": m, "data_dir": "synthetic mixed (test words)", "split": "test",
                            "lines": 600, "cer": c}) + "\n")
