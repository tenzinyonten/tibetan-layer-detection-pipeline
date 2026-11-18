"""Stage 3 HTML output (stacked underlines for overlapping layers)."""

from __future__ import annotations

import html as html_mod
from pathlib import Path

from ..layer_config import LAYERS


def write_html(book_id: str, text: str, layers: dict[str, list[dict]],
               errors: dict[str, str], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{book_id}.html"

    events = []  # (char_pos, delta, layer_name) delta +1 open, -1 close
    for name, spans in layers.items():
        for s in spans:
            events.append((s["start"], 1, name))
            events.append((s["end"], -1, name))
    bounds = sorted({p for p, _, _ in events} | {0, len(text)})

    active: set[str] = set()
    open_at = {p: [] for p in bounds}
    close_at = {p: [] for p in bounds}
    for p, d, name in events:
        (open_at if d == 1 else close_at)[p].append(name)

    parts = []
    for i in range(len(bounds) - 1):
        p, nxt = bounds[i], bounds[i + 1]
        for name in close_at.get(p, ()):
            active.discard(name)
        for name in open_at.get(p, ()):
            active.add(name)
        chunk = html_mod.escape(text[p:nxt])
        if not chunk:
            continue
        if active:
            shadows = ", ".join(f"inset 0 -{2 + 2 * i}px {LAYERS[n].color}"
                                for i, n in enumerate(sorted(active)))
            title = ", ".join(sorted(active))
            parts.append(f'<span class="span" style="box-shadow:{shadows}" title="{title}">{chunk}</span>')
        else:
            parts.append(chunk)
    body = "".join(parts)

    counts = {name: len(spans) for name, spans in layers.items()}
    legend = "".join(
        f'<div class="legend-row"><span class="swatch" style="background:{LAYERS[n].color}"></span>'
        f'{n} <span class="mut">({counts.get(n, 0)} spans'
        + (f", {errors[n]}" if n in errors else "")
        + ')</span></div>'
        for n in layers.keys() | errors.keys()
    )

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>{html_mod.escape(book_id)} — layer detection</title>
<style>
body {{ font: 15px/1.9 system-ui, sans-serif; max-width: 900px; margin: 24px auto; padding: 0 16px; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 16px; margin-bottom: 16px; padding: 8px 12px;
          border: 1px solid #ddd; border-radius: 6px; }}
.legend-row {{ display: flex; align-items: center; gap: 6px; }}
.swatch {{ width: 14px; height: 14px; border-radius: 3px; display: inline-block; }}
.mut {{ color: #777; font-size: 13px; }}
.text {{ white-space: pre-wrap; word-break: break-word;
        font-family: "Noto Serif Tibetan", "Jomolhari", serif; font-size: 17px; }}
.span {{ padding: 0 1px; }}
</style></head><body>
<h1>{html_mod.escape(book_id)}</h1>
<div class="legend">{legend}</div>
<div class="text">{body}</div>
</body></html>"""
    path.write_text(page, encoding="utf-8")
    return path
