"""
Glu_absolute_heatmap.py
======================
Absolute-position glutamate (E) density heatmap.

Input  : a FASTA file — the proteins in it are exactly what gets plotted.
Output : heatmap_absolute.pdf/.svg  (legend embedded below x-axis)

Each row = one protein. Each cell = one sliding window of --window aa,
stepped 1 residue at a time. Cell colour encodes local %E density.
Shorter proteins end earlier; the remainder of the row is neutral grey
("no data") so it is visually distinct from zero %E.


Colour schemes  (--colorscheme)
--------------------------------
  redmaroon   white → steel-blue → salmon → dark maroon  [default]
  classic     white → steel-blue → yellow-orange → red   [original scheme]
  magma       black → purple → orange → yellow
  viridis     purple → teal → yellow-green
  heat        white → yellow → orange → red
  ice         white → light blue → dark navy

Usage
-----
    python Glu_absolute_heatmap.py --fasta proteins.fasta

    python Glu_absolute_heatmap.py \\
        --fasta       proteins.fasta \\
        --window      30 \\
        --threshold   20 \\
        --sort        maxE \\
        --top         50 \\
        --colorscheme redmaroon \\
        --lengthbar \\
        --rowlines \\
        --fmt         pdf \\
        --outdir      results/

Arguments
---------
  --fasta         FASTA file to plot (required)
  --window        Sliding window size in aa (default 30)
  --threshold     %E value for colour transition + legend marker (default 20)
  --sort          maxE | meanE | name  (default maxE)
  --top           Keep only top N proteins by sort key (optional)
  --colorscheme   redmaroon | classic | magma | viridis | heat | ice
  --vmax          Colour scale ceiling in %E (default 50)
  --rowlines      Add faint horizontal lines between rows
  --lengthbar     Add right-side bar showing each protein's actual length
  --fmt           pdf | svg  (default pdf)
  --outdir        Output directory (default results/)
  --dpi           DPI for PDF (default 200; ignored for SVG)

Dependencies: numpy, matplotlib
"""

import argparse
import copy
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec


# ============================================================
# FASTA
# ============================================================

def parse_fasta(path: str) -> dict[str, str]:
    """Return OrderedDict {name: sequence}, preserving FASTA order."""
    from collections import OrderedDict
    seqs, name, parts = OrderedDict(), None, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if name:
                    seqs[name] = "".join(parts)
                name, parts = line[1:].split()[0], []
            else:
                parts.append(line.upper())
    if name:
        seqs[name] = "".join(parts)
    return seqs


# ============================================================
# Density
# ============================================================

def sliding_window(seq: str, window: int) -> np.ndarray:
    arr = np.array([1.0 if c == "E" else 0.0 for c in seq])
    if len(arr) < window:
        return np.array([], dtype=np.float32)
    cs = np.concatenate([[0.0], np.cumsum(arr)])
    return ((cs[window:] - cs[:-window]) / window * 100.0).astype(np.float32)


# ============================================================
# Colour schemes
# ============================================================

def make_cmap(scheme: str, threshold_frac: float) -> LinearSegmentedColormap:
    t = np.clip(threshold_frac, 0.05, 0.95)

    if scheme == "redmaroon":
        # white → steel-blue (below threshold) → salmon → dark maroon (above)
        cdict = {
            "red":   [(0.0, 1.00, 1.00), (t*0.5, 0.78, 0.78),
                      (t,   0.98, 0.98), (t+(1-t)*0.4, 0.80, 0.80), (1.0, 0.35, 0.35)],
            "green": [(0.0, 1.00, 1.00), (t*0.5, 0.82, 0.82),
                      (t,   0.60, 0.60), (t+(1-t)*0.4, 0.15, 0.15), (1.0, 0.04, 0.04)],
            "blue":  [(0.0, 1.00, 1.00), (t*0.5, 0.88, 0.88),
                      (t,   0.60, 0.60), (t+(1-t)*0.4, 0.15, 0.15), (1.0, 0.08, 0.08)],
        }
        return LinearSegmentedColormap("redmaroon", cdict, N=512)

    elif scheme == "classic":
        # Original scheme from scripts 02–05:
        # white → steel-blue (Blues) → yellow-orange → red (YlOrRd)
        blues = plt.cm.Blues(np.linspace(0.10, 0.70, int(512 * t)))
        reds  = plt.cm.YlOrRd(np.linspace(0.30, 1.00, 512 - int(512 * t)))
        return LinearSegmentedColormap.from_list(
            "classic", np.vstack([blues, reds]), N=512
        )

    elif scheme == "heat":
        colors = ["#ffffff", "#ffffb2", "#fecc5c", "#fd8d3c", "#f03b20", "#bd0026"]
        return LinearSegmentedColormap.from_list("heat", colors, N=512)

    elif scheme == "ice":
        colors = ["#ffffff", "#d0e8f5", "#6baed6", "#2171b5", "#08306b"]
        return LinearSegmentedColormap.from_list("ice", colors, N=512)

    elif scheme == "magma":
        base = plt.cm.magma(np.linspace(0.05, 1.0, 512))
        return LinearSegmentedColormap.from_list("magma_w", base, N=512)

    elif scheme == "viridis":
        base = plt.cm.viridis(np.linspace(0.0, 1.0, 512))
        return LinearSegmentedColormap.from_list("viridis_w", base, N=512)

    else:
        raise ValueError(f"Unknown colorscheme: {scheme}")


# ============================================================
# Standalone legend
# ============================================================

def save_legend(
    cmap, vmax, threshold, window, scheme, fmt, out_path, dpi
):
    fig, ax = plt.subplots(figsize=(5.5, 0.85))
    fig.subplots_adjust(left=0.05, right=0.95, top=0.52, bottom=0.0)

    cb = fig.colorbar(
        ScalarMappable(norm=Normalize(vmin=0, vmax=vmax), cmap=cmap),
        cax=ax, orientation="horizontal",
    )
    ticks = sorted({0.0, threshold, vmax / 2, vmax})
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{t:.0f}%" for t in ticks])
    cb.ax.tick_params(labelsize=8)

    # Threshold marker in data coordinates
    cb.ax.axvline(threshold, color="#333", lw=1.4, ls="--")
    cb.ax.text(
        threshold, 1.08, f"threshold\n{threshold:.0f}%",
        transform=cb.ax.get_xaxis_transform(),
        fontsize=7, ha="center", va="bottom", color="#444",
    )

    ax.set_title(
        f"% Glutamate (E) density  |  window = {window} aa  |  scheme: {scheme}",
        fontsize=8, pad=14,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fmt_kw = {"format": fmt, "bbox_inches": "tight"}
    if fmt == "pdf":
        fmt_kw["dpi"] = dpi
    fig.savefig(out_path, **fmt_kw)
    plt.close(fig)
    print(f"  Legend  -> {out_path}")


# ============================================================
# Heatmap
# ============================================================

def plot_heatmap(
    entries, window, threshold, vmax, scheme,
    rowlines, lengthbar, border, fmt, out_path, dpi,
):
    n        = len(entries)
    names    = [e["name"] for e in entries]
    max_pos  = max(e["dens"].size for e in entries)
    lengths  = [len(e["seq"]) for e in entries]
    max_len  = max(lengths)

    # Matrix — NaN where protein ends
    mat = np.full((n, max_pos), np.nan, dtype=np.float32)
    for i, e in enumerate(entries):
        d = e["dens"]
        mat[i, : d.size] = d

    cmap     = make_cmap(scheme, threshold / vmax)
    row_h    = max(0.1, min(0.22, 24.0 / n))
    label_fs = max(7.0,   min(8.5,  110.0 / n))
    heat_h   = max(5.0, n * row_h)
    legend_h = 0.3    # inches — legend strip height
    gap_h    = 0.5     # inches — gap between heatmap x-axis and legend
    fig_h    = heat_h + legend_h + gap_h + 2.0
    fig_w    = 14.0

    # ── GridSpec layout ───────────────────────────────────────
    # Convert heights to ratios
    hr = [heat_h, legend_h]

    if lengthbar:
        gs = GridSpec(
            2, 2,
            height_ratios=hr,
            width_ratios=[1, 0.035],
            hspace=gap_h / (heat_h + legend_h),
            wspace=0.012,
        )
    else:
        gs = GridSpec(
            2, 1,
            height_ratios=hr,
            hspace=gap_h / (heat_h + legend_h),
        )

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs.update(
        left=0.01, right=0.98,
        top=0.96,  bottom=0.02,
    )

    ax      = fig.add_subplot(gs[0, 0])
    ax_leg  = fig.add_subplot(gs[1, 0])
    ax_len  = fig.add_subplot(gs[0, 1]) if lengthbar else None

    # ── Main heatmap ──────────────────────────────────────────
    # Mask NaN upfront and set bad-colour to white so empty regions
    # (where protein ends) are indistinguishable from the background.
    masked_mat = np.ma.masked_invalid(mat)
    cmap_plot  = copy.copy(cmap)
    cmap_plot.set_bad(color="white")

    im = ax.imshow(
        masked_mat, aspect="auto", cmap=cmap_plot,
        vmin=0.0, vmax=vmax, interpolation="nearest",
        extent=[1, max_pos + 1, n - 0.5, -0.5],
    )

    # Thin grey border around each protein's full length (optional)
    if border:
        border_lw = max(0.1, min(0.4, 6.0 / n))
        for i, e in enumerate(entries):
            seq_len = len(e["seq"])
            rect = mpatches.Rectangle(
                (1, i - 0.5),
                seq_len, 1.0,
                linewidth=border_lw,
                edgecolor="#999999",
                facecolor="none",
                zorder=4,
            )
            ax.add_patch(rect)

    # Row separators (optional) — use same grey as borders
    if rowlines and n <= 200:
        for y in np.arange(-0.5, n, 1):
            ax.axhline(y, color="#bbbbbb", lw=0.2, zorder=3)

    # Y-axis labels
    ax.set_yticks(range(n))
    ax.set_yticklabels(names, fontsize=label_fs, ha="right")
    ax.yaxis.set_tick_params(length=0, pad=3)

    # X-axis — position numbers and label
    ax.set_xlim(1, max_pos + 1)
    ax.tick_params(axis="x", labelsize=8, length=3)
    ax.set_xlabel("Window start position (aa)", fontsize=9, labelpad=5)

    ax.set_title(
        f"Glutamate (E) density — absolute position  |  "
        f"window = {window} aa  |  threshold = {threshold:.0f}%  |  "
        f"n = {n} proteins",
        fontsize=10, pad=7,
    )

    ax.text(
        0.995, 0.005, "White = sequence ends (no data)",
        transform=ax.transAxes, fontsize=6.5,
        ha="right", va="bottom", color="#666",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#ccc", alpha=0.8),
    )

    # ── Gradient legend (below x-axis, same width as heatmap) ─
    cb = fig.colorbar(
        ScalarMappable(norm=Normalize(vmin=0, vmax=vmax), cmap=cmap),
        cax=ax_leg,
        orientation="horizontal",
    )
    # Ticks: 0, threshold, midpoint, vmax
    ticks = sorted({0.0, threshold, vmax / 2, vmax})
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{t:.0f}%" for t in ticks])
    cb.ax.tick_params(labelsize=8, length=3)
    cb.set_label("% Glutamate (E) per window", fontsize=8, labelpad=4)

    # Threshold marker
    cb.ax.axvline(threshold, color="#333", lw=1.4, ls="--")
    cb.ax.text(
        threshold, 1.55,
        f"threshold ({threshold:.0f}%)",
        transform=cb.ax.get_xaxis_transform(),
        fontsize=7, ha="center", va="bottom", color="#444",
    )

    # No-data swatch
    ax_leg.text(
        1.01, 0.5, "  □ no data",
        transform=ax_leg.transAxes,
        fontsize=7, va="center", color="#888",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#ccc"),
    )

    # ── Optional length bar ───────────────────────────────────
    if ax_len is not None:
        len_mat  = np.array(lengths, dtype=float).reshape(-1, 1) / max_len
        len_cmap = LinearSegmentedColormap.from_list(
            "len", ["#f0f0f0", "#333333"], N=256
        )
        ax_len.imshow(
            len_mat, aspect="auto", cmap=len_cmap,
            vmin=0, vmax=1,
            extent=[0, 1, n - 0.5, -0.5],
        )
        ax_len.set_xticks([])
        ax_len.set_yticks([])
        ax_len.set_xlabel("len", fontsize=6, labelpad=2)
        ax_len.spines[["top","right","bottom","left"]].set_visible(False)
        ax_len.text(
            0.5, -0.01, f"max\n{max_len:,} aa",
            transform=ax_len.transAxes, fontsize=5.5,
            ha="center", va="top", color="#555",
        )

    # ── Save ──────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fmt_kw = {"format": fmt, "bbox_inches": "tight"}
    if fmt == "pdf":
        fmt_kw["dpi"] = dpi
    fig.savefig(out_path, **fmt_kw)
    plt.close(fig)
    print(f"  Saved -> {out_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Absolute-position glutamate heatmap. "
                    "Input FASTA = proteins to plot.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--fasta",        required=True,
                        help="FASTA file — every sequence in it will be plotted")
    parser.add_argument("--window",       type=int,   default=30)
    parser.add_argument("--threshold",    type=float, default=20.0)
    parser.add_argument("--sort",
                        choices=["maxE","meanE","name"], default="maxE")
    parser.add_argument("--top",          type=int,   default=None,
                        help="Keep only top N by sort key (optional)")
    parser.add_argument("--colorscheme",
                        choices=["redmaroon","classic","magma",
                                 "viridis","heat","ice"],
                        default="redmaroon")
    parser.add_argument("--vmax",         type=float, default=50.0)
    parser.add_argument("--rowlines",     action="store_true")
    parser.add_argument("--lengthbar",    action="store_true")
    parser.add_argument("--border",       action="store_true",
                        help="Draw a thin grey border around each protein bar")
    parser.add_argument("--fmt",
                        choices=["pdf","svg"], default="pdf")
    parser.add_argument("--outdir",       default="results")
    parser.add_argument("--dpi",          type=int,   default=200)
    args = parser.parse_args()

    # Load
    all_seqs = parse_fasta(args.fasta)
    print(f"FASTA: {len(all_seqs)} sequences in {args.fasta}")

    usable = {n: s for n, s in all_seqs.items() if len(s) >= args.window}
    if len(usable) < len(all_seqs):
        print(f"Skipped {len(all_seqs)-len(usable)} sequence(s) "
              f"shorter than window={args.window} aa")
    if not usable:
        print("ERROR: no usable sequences."); return

    # Compute
    entries = []
    for nm, seq in usable.items():
        dens = sliding_window(seq, args.window)
        entries.append({
            "name":  nm,
            "seq":   seq,
            "dens":  dens,
            "maxE":  float(np.nanmax(dens))  if dens.size else 0.0,
            "meanE": float(np.nanmean(dens)) if dens.size else 0.0,
        })

    # Sort
    if args.sort == "maxE":
        entries.sort(key=lambda e: e["maxE"],  reverse=True)
    elif args.sort == "meanE":
        entries.sort(key=lambda e: e["meanE"], reverse=True)
    else:
        entries.sort(key=lambda e: e["name"])

    # Top N
    if args.top and args.top < len(entries):
        print(f"--top {args.top}: keeping {args.top} of "
              f"{len(entries)} by {args.sort}")
        entries = entries[: args.top]

    print(f"Plotting {len(entries)} proteins  "
          f"(sort={args.sort}, scheme={args.colorscheme})\n")

    outdir = Path(args.outdir)
    ext    = args.fmt

    plot_heatmap(
        entries=entries, window=args.window,
        threshold=args.threshold, vmax=args.vmax,
        scheme=args.colorscheme,
        rowlines=args.rowlines, lengthbar=args.lengthbar,
        border=args.border,
        fmt=ext, out_path=outdir / f"heatmap_absolute.{ext}",
        dpi=args.dpi,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
