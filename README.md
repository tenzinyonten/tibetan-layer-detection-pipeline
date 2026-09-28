# tibetan-layer-detection-pipeline

Unified inference pipeline that runs all trained Tibetan layer detection
models on any Tibetan text and returns detected spans as character offsets.

## Models used

| Layer | Model | What it detects |
|---|---|---|
| Tsawa | BDRC/Bo-Tsawa-Detection | Root text (verse quoted and explained) |
| Sabche | BDRC/Bo-Sabche-Detection | Outline headings |
| Chapter | BDRC/Bo-Chapter-Detection | Chapter and section titles |
| Quotation | BDRC/Bo-Quotation-Detection | Citations from other works |
| Yigchung | BDRC/Bo-Yigchung-Detection | Small-print notes |

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
