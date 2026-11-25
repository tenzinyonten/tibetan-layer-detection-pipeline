# tibetan-layer-detection-pipeline

Finds seven kinds of structure in Tibetan text and returns each as character
offsets: **Tsawa**, **Sabche**, **Chapter**, **Quotation**, **Yigchung**, plus the
book's **Title** and **Author**. Each layer has its own mmBERT model; the pipeline
reads the text, runs the models, cleans the results and writes JSON, HTML and
OpenPecha `.opf` files.

## Contents

- [Layers and models](#layers-and-models)
- [Installation](#installation)
- [Pipeline](#pipeline)
- [What post-processing does](#what-post-processing-does)
- [Usage](#usage)
- [Title and author](#title-and-author)
- [Speed and hardware](#speed-and-hardware)
- [Repository layout](#repository-layout)
- [Notes](#notes)
- [REST API + job queue](#rest-api--job-queue)

## Layers and models

| Layer | Model | What it detects | How it reads the text |
|---|---|---|---|
| Tsawa | [BDRC/Bo-Tsawa-Detection](https://huggingface.co/BDRC/Bo-Tsawa-Detection) | Root text (verse quoted and explained) | windows over the whole text |
| Sabche | [BDRC/Bo-Sabche-Detection](https://huggingface.co/BDRC/Bo-Sabche-Detection) | Outline headings | windows over the whole text |
| Chapter | [BDRC/Bo-Chapter-Detection](https://huggingface.co/BDRC/Bo-Chapter-Detection) | Chapter and section titles | windows over the whole text |
| Quotation | [BDRC/Bo-Quotation-Detection](https://huggingface.co/BDRC/Bo-Quotation-Detection) | Citations from other works | windows over the whole text |
| Yigchung | [BDRC/Bo-Yigchung-Detection](https://huggingface.co/BDRC/Bo-Yigchung-Detection) | Small-print notes | windows over the whole text |
| Title | [BDRC/Bo-Title-Detection](https://huggingface.co/BDRC/Bo-Title-Detection) (private) | The title of the work | first and last 3,000 characters; one span |
| Author | [BDRC/Bo-Author-Detection](https://huggingface.co/BDRC/Bo-Author-Detection) (private) | The author of the work | first and last 3,000 characters; one span |

Title and author are separate single-field models. A joint model that finds both
in one pass, `Yontenn/mmbert-title-author-run2` (private), also works: pass it as
both `--title-model` and `--author-model`. On the 2,384-work test split the
separate models score a little higher (share of works with the right span at
IoU >= 0.5: title 0.842 and author 0.734, against 0.833 and 0.711 for the joint
model), so they are the default.

Window models read up to 8,190 tokens at a time, so a long book is cut into
overlapping windows (each layer has its own stride). All settings per layer are in
[`layer_config.py`](tibetan_layer_detection/layer_config.py).

## Installation

Requires Python 3.9 or newer.

```bash
git clone https://github.com/tenzinyonten/tibetan-layer-detection-pipeline
cd tibetan-layer-detection-pipeline
python3 -m venv .venv && source .venv/bin/activate
pip install .            # or `pip install -e .` while developing
```

The models are downloaded from Hugging Face the first time they are used (roughly
1 GB each) and kept in the Hugging Face cache. They are private, so log in
first, or export a token:

```bash
huggingface-cli login            # or: export HF_TOKEN=your_token_here
```

No key or token is ever stored by this package.

`pip install .` adds the `tibetan_layer_detection` package and four commands:
`tibetan-detect`, `tibetan-preprocess`, `tibetan-infer` and `tibetan-postprocess`.
The dependencies (`transformers`, `torch`, `pyyaml`, `numpy`) come with it;
`requirements-tibetan-layer-detection.txt` pins the exact versions this was last
verified against. The REST API needs more: see [REST API + job queue](#rest-api--job-queue).

**GPU:** it works on CPU, but a GPU is much faster. Install a CUDA build of PyTorch
first (see pytorch.org) and the pipeline uses it automatically. For Docker, see the
GPU option below.

**Check the install** (offline, no model downloads):

```bash
tibetan-detect --help
python -m unittest discover -s tests
```

## Pipeline

Three stages, each also a command you can run on its own:

```
text/.opf --[preprocess]--> preprocessed/*.json --[infer]--> predictions/*.json --[postprocess]--> JSON/HTML/.opf + logs
```

- **preprocess**: reads a `.txt` file, a raw OpenPecha book folder, or a folder of
  either. No model is involved.
- **infer**: windows, model inference, Viterbi decoding, confidence, and the
  per-layer cleanup rules (extending spans to the tsheg, merging small gaps,
  Sabche fragment repair and shad rule, Chapter bracket extension). Tsawa also runs
  the quotation model, so post-processing can drop quoted tsawa. Title and author
  run on the first and last 3,000 characters. It is checkpointed: a book already
  in the output folder is skipped, so a crashed batch resumes where it stopped.
- **postprocess**: joins Stage 1's text with Stage 2's spans, filters them
  (below), marks `review_needed`, and writes JSON, HTML and `.opf` output, a batch
  summary and a run log. It is cheap to re-run with a different threshold without
  re-running inference.

`tibetan-detect` runs all three in order, for one-command use.

Flowcharts of the stages and of post-processing, step by step:
[docs/postprocessing_flow.md](docs/postprocessing_flow.md) (clean) and
[docs/postprocessing_flow_fun.md](docs/postprocessing_flow_fun.md) (fun).

## What post-processing does

Stage 3 applies these filters to each book, in this order:

| Step | What it removes | Layers | Turn off (`tibetan-postprocess` only) |
|---|---|---|---|
| Quoted tsawa | a tsawa span that is 50% or more covered by predicted quotations | tsawa | `--keep-quoted-tsawa` |
| Short spans | spans under 5 characters | every layer except title and author | none |
| Non-Tibetan | spans with no Tibetan character (for example scraped English text) | all | none |
| Low confidence | spans below confidence 0.85 | sabche | `--keep-low-confidence` |

Then every remaining span gets `review_needed`: true when its confidence is below
`--review-threshold` (default 0.7, one number for all layers). Title and author can
use their own bar with `--title-author-review-threshold`. What was dropped is listed
in the batch summary. `tibetan-detect` and `detect()` always apply the filters; to switch one
off, run the stages separately and pass the flag to `tibetan-postprocess`.

## Usage

```bash
# one document, all seven layers, JSON and HTML
tibetan-detect --text book.txt --all --json --html --out results/

# a folder of books (renamed .txt files or raw OpenPecha layout), chosen layers
tibetan-detect --dir books/ --layers tsawa sabche --json --out results/

# also write the predictions as OpenPecha layers
# (layers/v001/ for a plain .txt; layers/predicted/ for a real OpenPecha book, never over its gold layers/v001/)
tibetan-detect --text data/raw_opf/P000201.opf/P000201.opf/base/v001.txt \
    --all --json --opf --out results/

# include title and author in the .opf (written as BookTitle.yml and Author.yml)
tibetan-detect --text book.txt --all --json --opf --title-author-opf --out results/

# run one stage at a time (resume a crashed batch, or re-render with a new threshold)
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
result["layers"]["title"]
# [{"start": 0, "end": 52, "confidence": 1.0, "block": "head", "model": "Yontenn/mmbert-title-only", "review_needed": False}]
```

It runs all three stages in one process (a temporary folder, cleaned up, unless you
pass `write_json=True`, `write_html=True`, `write_opf=True` or `out_dir=...`). It
has no checkpointing, so for a large batch that may need resuming use the commands.

Every span has a `confidence` (the mean softmax probability of its decoded labels)
and `review_needed`.

## Title and author

Title and author are not found by sliding windows. The model reads the first and
last 3,000 characters of the file (the whole text if it is shorter) as one sequence
and returns **one span per field**, with `block` saying where it was found
(`head`, `tail` or `whole`).

- **Title** is the first span the model finds.
- **Author** is the last span by default, which suits a single work whose
  colophon names the author. `--author-span first` keeps the first span instead.
  For a whole book that is a compilation, the authors are usually on the title page,
  so `first` often fits better; see Notes.
- `--title-model` and `--author-model` choose other models (a Hugging Face repo or a
  local folder).

One span per file means a compilation of several works gets only one title and one
author, even though each work has its own.

## Speed and hardware

CPU time grows in a straight line with text length: about **8 ms per character**
for all seven layers on an Apple M5 Pro (18 cores, 48 GB, PyTorch on 6 threads).
Measured on one real book, one run each:

| Text | Time on CPU |
|---|---|
| 20,000 characters | about 3.9 minutes |
| 140,805 characters (full book) | 18.7 minutes, about 95 CPU-minutes, peak memory 5.5 GB |

Title and author add only about 13 seconds, whatever the length. Quotation is the
slowest layer because it uses a smaller stride and so more windows. A GPU is
recommended for batches and for long books.

## Repository layout

```
tibetan_layer_detection/
  preprocess.py        Stage 1: read .txt / OpenPecha books
  infer.py             Stage 2: windows, model inference, Viterbi, per-layer cleanup
  windows.py, decode.py   windowing and span decoding used by Stage 2
  title_author.py      title and author (first and last 3,000 characters)
  postprocess/         Stage 3
    filters.py           short / non-Tibetan / low-confidence / quoted-tsawa drops
    review.py            review thresholds and review_needed flags
    json_out.py, html_out.py, opf_out.py   the three outputs
    cli.py               the tibetan-postprocess command and the per-book loop
  detect_layers.py     tibetan-detect: runs the three stages in order
  stage_args.py        the argument lists detect_layers and detect() pass to the stages
  layer_config.py      every layer's model, decoding settings and rules
api/                   REST API and job queue
docs/                  flowcharts of the pipeline and of post-processing
tests/                 unit tests (offline: no model downloads)
```

## Notes

- GPU recommended for books over 50K characters (CPU works but is slow).
- Yigchung positive detection is not yet validated on real data; its own model
  card says to cite a test F1 of 0.431, well below the other layers.
- Title and author were measured on a test split of single works. On whole books
  taking the last author span often misses the title-page authors; take
  `--author-span first` there. The title/author review threshold (0.7) is a
  placeholder that has not been tuned.
- The Sabche confidence cutoff (0.85) and the shad rule are Sabche-only precision
  rules; their settings are in `layer_config.py`.

## REST API + job queue

A minimal FastAPI + Celery + Redis job queue on top of the package above --
no auth, no database; Redis (via Celery's own result backend) holds all job
state.

```bash
export HF_TOKEN=your_token_here   # needed to download the models
docker-compose up
```

This runs on CPU. To run the worker on an NVIDIA GPU, add the GPU override
(builds torch with CUDA 12.4 and gives the worker the GPU; needs an NVIDIA
driver and the NVIDIA Container Toolkit on the host):

```bash
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
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
