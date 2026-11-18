# Post-processing flow, the fun version 🎉

The same journey as the clean diagram, told like a day in the life of a book. 📖
Package folder: `tibetan_layer_detection/`.

## 🚂 The three stops

```mermaid
flowchart LR
    A(["🏁 tibetan-detect<br/>press go"]):::start --> S1
    S1["📥 Stage 1: preprocess.py<br/>Welcome, book! Let me read you."]:::stage --> S2
    S2["🧠 Stage 2: infer.py<br/>Cut into windows, ask the 7 models,<br/>Viterbi picks the best path"]:::stage --> S3
    S3["🧼 Stage 3: postprocess.py<br/>The big clean-up and send-off"]:::stage
    S2 -. "🔗 tsawa brings its friend quotation along" .-> S3
    classDef start fill:#ffe08a,stroke:#b8860b,color:#222
    classDef stage fill:#cfe8ff,stroke:#2b6cb0,color:#222
```

## 🧼 Stage 3: the clean-up crew

```mermaid
flowchart TD
    IN(["📦 predictions plus source text arrive"]):::start --> LOAD["🤝 load_pair<br/>match spans with their text"]:::step
    LOAD --> OK{"❓ Did Stage 1 leave a file?"}
    OK -- "no 😬" --> SKIP["⏭️ warn and skip the book"]:::bad
    OK -- "yes 🎉" --> F1
    F1["🧹 Sweep out tsawa hiding inside quotes<br/>split_quoted_tsawa"]:::filter
    F2["✂️ Snip the tiny crumbs, under 5 chars<br/>drop_short_spans"]:::filter
    F3["🔤 Boot out anything with no Tibetan<br/>drop_non_tibetan_spans"]:::filter
    F4["🎯 Sabche: only the confident ones stay<br/>drop_low_confidence"]:::filter
    F1 --> F2 --> F3 --> F4
    F4 --> OUT{"🎁 What shall we make?"}
    OUT --> J["🧾 JSON<br/>every span gets a review_needed flag 🚩"]:::out
    OUT --> H["🎨 HTML<br/>pretty colours"]:::out
    OUT --> O["🗂️ .opf layers"]:::out
    O --> OQ{"📚 Real OpenPecha book with gold layers?"}
    OQ -- "yes 🏅" --> OP["🛡️ layers/predicted/<br/>gold is never touched"]:::safe
    OQ -- "no" --> OV["🆕 plain .txt: fresh opf<br/>then layers/v001/"]:::out
    J --> SUM
    H --> SUM
    OP --> SUM
    OV --> SUM
    SUM(["🏆 _batch_summary.json + run log<br/>All done!"]):::start
    classDef start fill:#ffe08a,stroke:#b8860b,color:#222
    classDef step fill:#e6e6fa,stroke:#6a5acd,color:#222
    classDef filter fill:#ffd6d6,stroke:#c0392b,color:#222
    classDef out fill:#d4f5d4,stroke:#2f855a,color:#222
    classDef safe fill:#d4f5f0,stroke:#2c7a7b,color:#222
    classDef bad fill:#eeeeee,stroke:#888,color:#222
```

## 🎛️ Little switches

- 🙈 `--keep-quoted-tsawa`: keep the tsawa that hide in quotes
- 🙉 `--keep-low-confidence`: keep the unsure Sabche spans
- 🚩 `--review-threshold`: one number for every layer (default 0.7); a span below it is flagged review_needed
- 🎓 `--title-author-review-threshold`: a separate bar for title and author
- 🏷️ `--title-author-opf`: also write `BookTitle.yml` and `Author.yml`

## 🔎 Who does what

| Crew member | Function | File |
|---|---|---|
| 🧹 | `split_quoted_tsawa` | `tibetan_layer_detection/postprocess.py` |
| ✂️ | `drop_short_spans` (title and author are safe) | `tibetan_layer_detection/postprocess.py` |
| 🔤 | `drop_non_tibetan_spans` | `tibetan_layer_detection/postprocess.py` |
| 🎯 | `drop_low_confidence` (Sabche, 0.85) | `tibetan_layer_detection/postprocess.py` |
| 🚩 | `needs_review` | `tibetan_layer_detection/postprocess.py` |
