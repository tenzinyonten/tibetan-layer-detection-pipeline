# tibetan-layer-detection-pipeline

Detects Tsawa, Sabche, Chapter, Quotation and Yigchung layers in Tibetan
text with per-layer mmBERT models, and returns spans as character offsets.

## Models used

| Layer | Model | What it detects |
|---|---|---|
| Tsawa | [BDRC/Bo-Tsawa-Detection](https://huggingface.co/BDRC/Bo-Tsawa-Detection) | Root text (verse quoted and explained) |
| Sabche | [BDRC/Bo-Sabche-detection](https://huggingface.co/BDRC/Bo-Sabche-detection) | Outline headings |
| Chapter | [BDRC/Bo-Chapter-Detection](https://huggingface.co/BDRC/Bo-Chapter-Detection) | Chapter and section titles |
| Quotation | [BDRC/Bo-Quotation-Detection](https://huggingface.co/BDRC/Bo-Quotation-Detection) | Citations from other works |
| Yigchung | [BDRC/Bo-Yigchung-Detection](https://huggingface.co/BDRC/Bo-Yigchung-Detection) | Small-print notes |

## Installation

```bash
pip install .
export HF_TOKEN=your_token_here  # needed for private models (Sabche, Chapter, Quotation)
```

Installs the `tibetan_layer_detection` package plus four console scripts
(`tibetan-detect`, `tibetan-preprocess`, `tibetan-infer`, `tibetan-postprocess`).
Dependencies (`transformers`, `torch`, `pyyaml`, `numpy`) are pulled in
automatically; `requirements-tibetan-layer-detection.txt` pins the exact
versions this was last verified against.

## Pipeline

Three independently runnable stages, each also a console script:

```
text/.opf --[preprocess]--> preprocessed/*.json --[infer]--> predictions/*.json --[postprocess]--> JSON/HTML/.opf + logs
```

- **preprocess** -- text extraction (handles a `.txt` file, a raw OpenPecha
  book folder, or a folder of either; no model involved).
- **infer** -- windowing + model inference + Viterbi decoding + confidence.
  Checkpointed: a book already in the output folder is skipped, so a
  crashed batch resumes where it left off.
- **postprocess** -- joins Stage 1's text with Stage 2's spans, computes
  `review_needed` (confidence below `--review-threshold`, default 0.7), and
  writes JSON/HTML/`.opf` output, a batch summary, and a run log. Cheap to
  re-run with a different threshold without re-running inference.

`tibetan-detect` runs all three in sequence, for one-command use.

## Usage

```bash
# single document, all layers, both outputs
tibetan-detect --text book.txt --all --json --html --out results/

# folder of books (renamed .txt files or raw OpenPecha layout), specific layers
tibetan-detect --dir books/ --layers tsawa sabche --json --out results/

# also write predictions back as OpenPecha layers (layers/predicted/, never layers/v001/)
tibetan-detect --text data/raw_opf/P000201.opf/P000201.opf/base/v001.txt \
    --all --json --opf --out results/

# run one stage at a time (e.g. resume a crashed batch, or re-render with a new threshold)
tibetan-preprocess --input books/ --out results/preprocessed/
tibetan-infer --input results/preprocessed/ --out results/predictions/ --all
tibetan-postprocess --input results/predictions/ --source results/preprocessed/ \
    --out results/output/ --json --review-threshold 0.6
```

Python API:

```python
from tibetan_layer_detection import detect

result = detect("book.txt", layers=["all"])
result["layers"]["sabche"]
# [{"start": 176, "end": 189, "confidence": 0.9974, "review_needed": False}, ...]
```

Runs all three stages in-process (a temp directory, auto-cleaned, unless you
pass `write_json=True` / `write_html=True` / `write_opf=True` and/or
`out_dir=...`). For a large batch you might need to resume, use the
console scripts instead -- this function has no checkpointing.

Each span carries a `confidence` (mean softmax probability of its
Viterbi-decoded labels) and `review_needed` (`confidence < review_threshold`).

## Notes

- GPU recommended for books over 50K characters (CPU works but is slow).
- Yigchung positive detection not yet validated on real data; its own model
  card says to cite a test F1 of 0.431, well below the other four layers.

## REST API + job queue

A minimal FastAPI + Celery + Redis job queue on top of the package above --
no auth, no database; Redis (via Celery's own result backend) holds all job
state.

```bash
export HF_TOKEN=your_token_here   # needed for the private models
docker-compose up
```

Books are read from `./data/` on the host (bind-mounted read-only into the
worker container at `/data`); `book_path` in a request is resolved relative
to that folder.

```bash
curl -X POST http://localhost:8000/process \
  -H "Content-Type: application/json" \
  -d '{"book_path": "P000201.opf"}'
# {"job_id": "..."}

curl http://localhost:8000/status/<job_id>
# {"job_id": "...", "status": "running", "stage": "infer", "book_id": "P000201"}

curl http://localhost:8000/results/<job_id>
# the same result shape as tibetan_layer_detection.detect(): book_id,
# text_length, layers (with confidence/review_needed per span), errors
```

`layers` in the POST body is optional (defaults to every configured layer):
`{"book_path": "P000201.opf", "layers": ["tsawa", "sabche"]}`.

`status` is one of `queued` / `running` / `complete` / `failed`; while
`running`, `stage` is `preprocess` / `infer` / `postprocess`.

To run without Docker: `pip install -r api/requirements.txt`, a local Redis
(`redis-server`), then `celery -A api.worker worker --loglevel=info` and
`uvicorn api.main:app --host 0.0.0.0 --port 8000` as two separate processes,
with `REDIS_URL` and `TIBETAN_DATA_DIR` set (see `api/config.py`).
