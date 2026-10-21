"""Figures for the seam_qa report (docs/fig/*.png). Run from seam_qa/: python docs/figures.py"""

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from rasterio.enums import Resampling
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import synth  # noqa: E402
from config import OUT, STRIDE, load_campaigns  # noqa: E402
from crops import CropReader  # noqa: E402
from metrics import detections, found_ranks, targets  # noqa: E402
from seam_graph import read_seams  # noqa: E402

FIG = Path(__file__).resolve().parent / "fig"
FIG.mkdir(exist_ok=True)
NAME = "eure_et_loir_nord_2025"
D = OUT / NAME

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29c"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": GRID, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.facecolor": "white",
})

st = pd.read_parquet(D / "stations.parquet")
cnn = pd.read_parquet(D / "cnn_scores.parquet")
base = pd.read_parquet(D / "baseline_scores.parquet")
pts_all = gpd.read_file(D / "artefacts.gpkg")
TYPE_LABEL = {2: "Bâtiment coupé", 8: "Véhicule coupé", 13: "Décalage", 16: "Erreur de mosaïquage",
              10: "Hotspot", 14: "Ombre de nuage", 3: "type 3", 5: "type 5", 15: "type 15", 21: "type 21"}


def fig_distance():
    """Distance from every Eure-et-Loir point to the nearest seamline (full seam layer)."""
    cfg = load_campaigns()[NAME]
    d = []
    for g in pts_all.geometry:  # one small read per point: the whole layer does not fit in memory
        near = read_seams(cfg, g.buffer(200).bounds)
        seams = near.geometry if near.attrs["kind"] == "lines" else near.boundary
        d.append(seams.distance(g).min())
    df = pd.DataFrame({"type": pts_all["art_arttype_id"], "d": np.maximum(d, 0.05),
                       "seam": pts_all["is_seam_type"]})
    order = df.groupby("type")["d"].median().sort_values().index
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    rng = np.random.default_rng(0)
    for i, t in enumerate(order):
        v = df[df.type == t]
        c = BLUE if v.seam.iloc[0] else GRAY
        ax.scatter(v.d, i + rng.uniform(-0.18, 0.18, len(v)), s=22, color=c, edgecolor="white", lw=0.6, zorder=3)
        ax.text(130, i, f"n={len(v)}", va="center", fontsize=7.5, color=MUTED)
    ax.axvline(3, color=INK2, lw=0.8, ls="--", zorder=2)
    ax.text(3.2, len(order) - 0.45, "3 m: positive radius", fontsize=7.5, color=INK2)
    ax.set_yticks(range(len(order)), [TYPE_LABEL.get(t, str(t)) for t in order])
    ax.set_xscale("log")
    ax.set_xlim(0.04, 200)
    ax.set_xticks([0.1, 1, 10, 100], ["0.1 m", "1 m", "10 m", "100 m"])
    ax.set_xlabel("distance from the artefact point to the nearest seamline")
    ax.grid(axis="y", visible=False)
    ax.invert_yaxis()
    fig.savefig(FIG / "distance.png")
    plt.close(fig)


def read_tile(path, factor=8):
    with rasterio.open(path) as ds:
        img = ds.read([1, 2, 3], out_shape=(3, ds.height // factor, ds.width // factor),
                      resampling=Resampling.average)
        b = ds.bounds
    return np.moveaxis(img, 0, -1), (b.left, b.right, b.bottom, b.top)


def fig_tile(tile="3788"):
    edges = gpd.read_file(D / "edges.gpkg")
    path = st.loc[st.tile == tile, "tile_path"].iloc[0]
    img, ext = read_tile(path)
    inside = (st.x.between(ext[0], ext[1])) & (st.y.between(ext[2], ext[3]))
    s = st[inside]
    p = pts_all[pts_all.geometry.x.between(ext[0], ext[1]) & pts_all.geometry.y.between(ext[2], ext[3])]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.7))
    for ax in axes:
        ax.imshow(img, extent=ext)
        edges.plot(ax=ax, color="white", lw=0.8, alpha=0.9)
        ax.set_xlim(ext[:2]); ax.set_ylim(ext[2:])
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.scatter(p.geometry.x, p.geometry.y, marker="*", s=170, facecolor="none", edgecolor="#ffd400", lw=1.3,
                   zorder=5)
    a = axes[0]
    for lab, c, sz, txt in ((0, BLUE, 7, "negative"), (-1, GRAY, 7, "ignored"), (1, ORANGE, 16, "positive")):
        v = s[s.label == lab]
        a.scatter(v.x, v.y, s=sz, color=c, edgecolor="white", lw=0.3, zorder=4, label=txt)
    a.legend(loc="lower left", fontsize=7, framealpha=0.9, markerscale=1.4)
    a.set_title("Stations every 4 m and their labels", fontsize=9, color=INK)
    b = axes[1]
    sc = b.scatter(s.x, s.y, c=cnn.loc[s.index, "cnn"], cmap="Blues", vmin=0, vmax=1, s=10,
                   edgecolor="white", lw=0.2, zorder=4)
    b.set_title("SeamNet out-of-fold score", fontsize=9, color=INK)
    cb = fig.colorbar(sc, ax=b, fraction=0.04, pad=0.02)
    cb.ax.tick_params(labelsize=7)
    fig.savefig(FIG / "tile.png")
    plt.close(fig)


def fig_channels():
    r = CropReader(D)
    row = st[(st.label == 1) & (st.tile == "3766")].sort_values("d_seam_pt").iloc[0]
    c5 = r.crop(row.x, row.y, row.angle, 192, 0.05)
    c10 = r.crop(row.x, row.y, row.angle, 192, 0.10)
    sd = c5["side"] * np.minimum(c5["dist"], 32) / 32
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    axes[0].imshow(c5["rgb"]); axes[0].set_title("RGB, 5 cm (9.6 m)", fontsize=8.5)
    axes[1].imshow(c10["rgb"]); axes[1].set_title("RGB, 10 cm (19.2 m)", fontsize=8.5)
    im = axes[2].imshow(sd, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[2].set_title("4th channel: signed distance", fontsize=8.5)
    axes[2].text(40, 100, "left\nphoto +", color="white", fontsize=7.5, ha="center", va="center")
    axes[2].text(160, 100, "right\nphoto −", color="white", fontsize=7.5, ha="center", va="center")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for ax, s in ((axes[0], 192), (axes[1], 192)):
        ax.annotate("", xy=(s * 0.93, s * 0.1), xytext=(s * 0.93, s * 0.4),
                    arrowprops=dict(arrowstyle="-|>", color="#ffd400", lw=1.4))
    axes[0].text(192 * 0.90, 192 * 0.45, "seam\ntangent", color="#ffd400", fontsize=6.5, ha="right")
    fig.colorbar(im, ax=axes[2], fraction=0.045, pad=0.03).ax.tick_params(labelsize=7)
    fig.savefig(FIG / "channels.png")
    plt.close(fig)


def fig_synth(seed=5):
    crops = np.load(D / "crops_192.npy", mmap_mode="r")
    rng = np.random.default_rng(seed)
    full = pd.read_parquet(D / "features.parquet")["valid_frac"] > 0.999
    clean = (st.label == 0) & full
    pools = [st.index[clean & (st.bldg_dist < 1)], st.index[clean & (st.on_road == 1)],
             st.index[clean & (st.bldg_dist < 1)], st.index[clean & (st.bldg_dist < 5)]]
    fig, axes = plt.subplots(4, 3, figsize=(5.6, 7.6))
    names = ["object_cut (bâtiment coupé)", "vehicle_cut (véhicule coupé)", "shift (décalage)",
             "radiometric (raccord)"]
    for t in range(4):
        c = np.array(crops[rng.choice(pools[t])])
        o = synth.MAKERS[t](c.copy(), np.random.default_rng(seed + t))
        for j, (im, title) in enumerate(((c[0, ..., :3], "clean, 5 cm"), (o[0, ..., :3], "synthetic, 5 cm"),
                                         (o[1, ..., :3], "synthetic, 10 cm"))):
            ax = axes[t, j]
            ax.imshow(im)
            ax.axvline(95.5, color="#ffd400", lw=0.5, ls=(0, (2, 3)), alpha=0.8)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            if t == 0:
                ax.set_title(title, fontsize=8.5)
        axes[t, 0].set_ylabel(names[t], fontsize=8)
    fig.tight_layout(h_pad=0.4, w_pad=0.3)
    fig.savefig(FIG / "synth.png")
    plt.close(fig)


def box(ax, x, y, w, h, text, fc="#f4f3ef", ec=GRID, fs=7.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec=ec, lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
            fontweight="bold" if bold else "normal", linespacing=1.35)


def arrow(ax, a, b, dashed=False, color=INK2):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, color=color, lw=0.9,
                                 ls=(0, (3, 2)) if dashed else "-"))


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    ax.set_xlim(-0.1, 12.1); ax.set_ylim(0, 5); ax.axis("off")
    box(ax, 0.0, 2.0, 1.95, 1.5, "Inputs\ntiles (RGBA, 5 cm)\nseamline polygons\ncv_artefact points\nBD TOPO (WFS)", fs=6.6)
    box(ax, 2.35, 2.0, 2.05, 1.5, "seam_graph.py\nstations every 4 m\nphotos L / R, strips\nBD TOPO context\nlabels", fc="#e8f1fc", ec=BLUE, fs=6.8)
    box(ax, 4.8, 3.7, 2.15, 0.95, "features.py\n25 cross-seam\nfeatures", fc="#e8f1fc", ec=BLUE, fs=6.8)
    box(ax, 4.8, 2.15, 2.15, 1.1, "dataset.py + crops.py\nseam-aligned crops\n2 scales × 4 channels", fc="#e8f1fc", ec=BLUE, fs=6.6)
    box(ax, 4.8, 0.85, 2.15, 0.9, "synth.py\nsynthetic positives", fc="#fdeee7", ec=ORANGE, fs=6.8)
    box(ax, 7.35, 3.7, 2.05, 0.95, "baseline.py\ngradient boosting\n(GBM)", fc="#e8f1fc", ec=BLUE, fs=6.8)
    box(ax, 7.35, 1.6, 2.05, 1.4, "train.py\nSeamNet\nerror + type heads", fc="#e8f1fc", ec=BLUE, fs=6.8)
    box(ax, 9.8, 2.25, 2.2, 1.6, "infer.py\nrank average\nNMS 8 m\ntop N per km", fc="#e8f1fc", ec=BLUE, fs=6.8)
    box(ax, 9.8, 0.55, 2.2, 1.2, "detections.gpkg\ncv_artefact schema\nreview in QGIS", fc="#e7f6f0", ec=AQUA, fs=6.8)
    arrow(ax, (1.95, 2.75), (2.35, 2.75))
    arrow(ax, (4.4, 3.2), (4.8, 4.0))
    arrow(ax, (4.4, 2.7), (4.8, 2.7))
    arrow(ax, (6.95, 4.17), (7.35, 4.17))
    arrow(ax, (6.95, 2.6), (7.35, 2.45))
    arrow(ax, (6.95, 1.3), (7.35, 1.95))
    arrow(ax, (9.4, 4.0), (9.8, 3.6))
    arrow(ax, (9.4, 2.3), (9.8, 2.7))
    arrow(ax, (10.9, 2.25), (10.9, 1.75))
    ax.add_patch(FancyArrowPatch((9.8, 0.75), (3.35, 2.0), arrowstyle="-|>", mutation_scale=9, color=AQUA,
                                 lw=0.9, ls=(0, (3, 2)), connectionstyle="angle,angleA=180,angleB=90,rad=0"))
    ax.text(3.5, 0.3, "operator accept / reject → new labels", fontsize=6.8, color=INK2)
    fig.savefig(FIG / "pipeline.png")
    plt.close(fig)


def fig_seamnet():
    fig, ax = plt.subplots(figsize=(7.4, 3.0))
    ax.set_xlim(-0.15, 10.15); ax.set_ylim(0, 4); ax.axis("off")
    box(ax, 0.0, 2.45, 1.55, 1.0, "crop 5 cm\n4 × 192 × 192\nRGB + signed dist.", fs=6.8)
    box(ax, 0.0, 0.55, 1.55, 1.0, "crop 10 cm\n4 × 192 × 192\nRGB + signed dist.", fs=6.8)
    box(ax, 2.05, 1.2, 1.9, 1.6, "ResNet-18\nImageNet weights\n4-channel stem\n(shared by\nboth scales)", fc="#e8f1fc", ec=BLUE, fs=7)
    box(ax, 4.45, 2.45, 1.1, 0.75, "GAP\n512", fs=7)
    box(ax, 4.45, 0.8, 1.1, 0.75, "GAP\n512", fs=7)
    box(ax, 6.0, 1.55, 1.0, 0.9, "concat\n1024", fs=7)
    box(ax, 7.4, 1.4, 1.2, 1.2, "Linear 256\nReLU\nDropout 0.3", fs=7)
    box(ax, 9.0, 2.45, 1.0, 0.9, "error\n1 logit\nBCE", fc="#fdeee7", ec=ORANGE, fs=7)
    box(ax, 9.0, 0.65, 1.0, 0.9, "type\n4 logits\nCE", fc="#e7f6f0", ec=AQUA, fs=7)
    arrow(ax, (1.55, 2.95), (2.05, 2.35)); arrow(ax, (1.55, 1.05), (2.05, 1.65))
    arrow(ax, (3.95, 2.35), (4.45, 2.8)); arrow(ax, (3.95, 1.65), (4.45, 1.2))
    arrow(ax, (5.55, 2.8), (6.0, 2.2)); arrow(ax, (5.55, 1.2), (6.0, 1.8))
    arrow(ax, (7.0, 2.0), (7.4, 2.0))
    arrow(ax, (8.6, 2.3), (9.0, 2.85)); arrow(ax, (8.6, 1.7), (9.0, 1.1))
    fig.savefig(FIG / "seamnet.png")
    plt.close(fig)


def fig_recall():
    pts = targets(D / "artefacts.gpkg")
    pts = pts[pts["d_seam_line"] <= 5.0]
    km = len(st) * STRIDE / 1000
    budgets = np.geomspace(0.5, 20, 40)
    series = [("SeamNet", cnn["cnn"].to_numpy(), BLUE),
              ("SeamNet + GBM", rankdata(cnn["cnn"]) + rankdata(base["gbm_all"]), AQUA),
              ("GBM (image + context)", base["gbm_all"].to_numpy(), ORANGE)]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    rng = np.random.default_rng(0)
    rand = np.mean([[(found_ranks(st, rng.random(len(st)), pts)[1] < round(b * km)).mean() for b in budgets]
                    for _ in range(20)], axis=0)
    ax.plot(budgets, rand, color=GRAY, lw=1.4, ls="--")
    ax.text(20.5, rand[-1], "random", color=INK2, fontsize=7.5, va="center")
    ends = []
    for name, s, c in series:
        ranks = found_ranks(st, s, pts)[1]
        r = [(ranks < round(b * km)).mean() for b in budgets]
        ax.plot(budgets, r, color=c, lw=1.8, label=name)
        ends.append((r[-1], name))
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 5, 10, 20], ["0.5", "1", "2", "5", "10", "20"])
    ax.set_xlim(0.5, 20)
    ax.set_ylim(0, 1.02)
    ax.set_yticks(np.linspace(0, 1, 6), [f"{int(v * 100)}%" for v in np.linspace(0, 1, 6)])
    ax.set_xlabel("review budget: detections per km of seam (log scale)")
    ax.set_ylabel("artefacts found (on-seam, n=29)")
    ax.legend(loc="upper left", fontsize=7.5, frameon=False)
    fig.savefig(FIG / "recall.png")
    plt.close(fig)


def fig_top(n=12):
    """Top SeamNet detections (out-of-fold), 5 cm crops, marked TP if an artefact point is within 10 m."""
    crops = np.load(D / "crops_192.npy", mmap_mode="r")
    pts = targets(D / "artefacts.gpkg")
    det = detections(st[["x", "y"]].to_numpy(), cnn["cnn"].to_numpy())[:n]
    pxy = np.c_[pts.geometry.x, pts.geometry.y]
    fig, axes = plt.subplots(2, n // 2, figsize=(7.4, 3.2))
    for k, (ax, i) in enumerate(zip(axes.flat, det)):
        ax.imshow(crops[i, 0, ..., :3])
        d = np.hypot(*(pxy - st.loc[i, ["x", "y"]].to_numpy(float)).T).min()
        hit = d <= 10
        typ = synth.TYPES[int(cnn.loc[i, [f"p_{t}" for t in synth.TYPES]].to_numpy().argmax())]
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color(AQUA if hit else ORANGE); sp.set_linewidth(2.2)
        ax.set_title(f"#{k + 1} {'artefact' if hit else 'no point'}", fontsize=6.8, color=INK)
        ax.set_xlabel(typ.replace("_", " "), fontsize=6.2, color=INK2)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.tight_layout(w_pad=0.3, h_pad=0.6)
    fig.savefig(FIG / "top.png")
    plt.close(fig)


if __name__ == "__main__":
    # python docs/figures.py [name ...]   e.g. "pipeline synth"; no name = all figures
    figs = [fig_pipeline, fig_seamnet, fig_distance, fig_tile, fig_channels, fig_synth, fig_recall, fig_top]
    for f in figs:
        if len(sys.argv) == 1 or f.__name__[4:] in sys.argv[1:]:
            f()
            print("ok", f.__name__)
