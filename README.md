# tibetan-layer-detection-pipeline

Unified inference pipeline that runs all trained Tibetan layer detection
models on any Tibetan text and returns detected spans as character offsets.

## Models used

| Layer | Model | What it detects |
|---|---|---|
| Tsawa | [BDRC/Bo-Tsawa-Detection](https://huggingface.co/BDRC/Bo-Tsawa-Detection) | Root text (verse quoted and explained) |
| Sabche | [BDRC/Bo-Sabche-detection](https://huggingface.co/BDRC/Bo-Sabche-detection) | Outline headings |
| Chapter | [BDRC/Bo-Chapter-Detection](https://huggingface.co/BDRC/Bo-Chapter-Detection) | Chapter and section titles |
| Quotation | [BDRC/Bo-Quotation-Detection](https://huggingface.co/BDRC/Bo-Quotation-Detection) | Citations from other works |
| Yigchung | [BDRC/Bo-Yigchung-Detection](https://huggingface.co/BDRC/Bo-Yigchung-Detection) | Small-print notes |

Helper modules (windowing, BIO Viterbi decoding) extracted from the layer_detection repo live in `common/`, so the only runtime dependencies are `transformers` and `torch`.

## Installation

```bash
pip install transformers torch
export HF_TOKEN=your_token_here  # needed for private models
```

## Usage

```bash
# single document, all layers
python detect_layers.py --text book.txt --all --json --html --out results/

# folder of books, specific layers
python detect_layers.py --dir books/ --layers tsawa sabche --json --out results/
```

## Notes

- GPU recommended for books over 50K characters (CPU works but is slow)
- Yigchung positive detection not yet validated on real data
