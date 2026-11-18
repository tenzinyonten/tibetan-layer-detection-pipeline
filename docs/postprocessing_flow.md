# Post-processing flow

How a book goes from raw text to final layer files, and which script runs when.
Package folder: `tibetan_layer_detection/`.

## 1. The three stages

```mermaid
flowchart LR
    A["Entry point<br/>tibetan-detect or detect()<br/>detect_layers.py"] --> S1
    S1["Stage 1: preprocess.py<br/>read .txt or OpenPecha book<br/>write preprocessed/ID.json"] --> S2
    S2["Stage 2: infer.py<br/>cut text into windows<br/>run each layer model<br/>Viterbi decode and per-layer cleanup<br/>write predictions/ID.json"] --> S3
    S3["Stage 3: postprocess.py<br/>filter spans, flag for review<br/>write JSON, HTML, .opf"]
    S2 -. "tsawa also runs the quotation model" .-> S3
```

Stage 2 runs the quotation model together with tsawa (`layers_for_run` in `infer.py`),
so Stage 3 can drop quoted tsawa. Title and author read only the first and last
3,000 characters and give one span each.

## 2. Stage 3 step by step (`postprocess.py`, `main()`)

```mermaid
flowchart TD
    IN["Inputs<br/>predictions/ID.json from Stage 2<br/>preprocessed/ID.json from Stage 1"] --> LOAD["load_pair<br/>join spans with the source text"]
    LOAD --> OK{"Stage 1 file found?"}
    OK -- no --> SKIP["warn and skip this book"]
    OK -- yes --> F1
    F1["1. split_quoted_tsawa<br/>Drop tsawa inside quotations<br/>tsawa covered 50 percent or more by predicted quotations"]
    F2["2. drop_short_spans<br/>Drop spans under 5 characters<br/>title and author are exempt"]
    F3["3. drop_non_tibetan_spans<br/>Drop spans with no Tibetan text<br/>English nav text and other scraping junk"]
    F4["4. drop_low_confidence<br/>Drop low-confidence spans<br/>Sabche only, confidence below 0.85"]
    F1 --> F2 --> F3 --> F4
    F4 --> OUT{"Which outputs were asked for?"}
    OUT --> J["--json: write_json<br/>adds review_needed to every span"]
    OUT --> H["--html: write_html<br/>colour-coded page"]
    OUT --> O["--opf: write_opf_layers"]
    O --> OQ{"Real OpenPecha book that already has layers/v001?"}
    OQ -- yes --> OP["write to layers/predicted/<br/>gold layers/v001 is never touched"]
    OQ -- no --> OV["plain .txt: write_plain_opf<br/>then layers/v001/"]
    J --> SUM
    H --> SUM
    OP --> SUM
    OV --> SUM
    SUM["_batch_summary.json<br/>run_DATE.json"]
```

## 3. Options that change the steps

| Option | Effect |
|---|---|
| `--keep-quoted-tsawa` | skip the quoted-tsawa drop |
| `--keep-low-confidence` | skip the low-confidence drop |
| `--review-threshold` | one number for every layer (default 0.7); a span below it is flagged review_needed |
| `--title-author-review-threshold` | review threshold for title and author only |
| `--title-author-opf` | also write title and author to `.opf` as `BookTitle.yml` and `Author.yml` |

## 4. Where each step lives

| Step | Function | File |
|---|---|---|
| Quoted tsawa drop | `split_quoted_tsawa` | `tibetan_layer_detection/postprocess.py` |
| Short-span drop | `drop_short_spans` | `tibetan_layer_detection/postprocess.py` |
| Non-Tibetan drop | `drop_non_tibetan_spans` | `tibetan_layer_detection/postprocess.py` |
| Low-confidence drop | `drop_low_confidence` | `tibetan_layer_detection/postprocess.py` |
| Review flags | `needs_review` | `tibetan_layer_detection/postprocess.py` |
| Layer settings | `LAYERS` | `tibetan_layer_detection/layer_config.py` |
