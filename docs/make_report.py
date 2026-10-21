"""Build docs/seam_qa_report.pdf. Run from seam_qa/ after docs/figures.py: python docs/make_report.py

Every number in the results section is recomputed here from out/<campaign>/*_scores.parquet.
"""

import datetime
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from config import OUT  # noqa: E402
from metrics import evaluate, operating_points, targets  # noqa: E402

DOCS = Path(__file__).resolve().parent
FIG = DOCS / "fig"
NAME = "eure_et_loir_nord_2025"
D = OUT / NAME

ttf = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("Sans", ttf / "DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("Sans-Bold", ttf / "DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("Sans-Oblique", ttf / "DejaVuSans-Oblique.ttf"))
pdfmetrics.registerFont(TTFont("Mono", ttf / "DejaVuSansMono.ttf"))
pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="Sans-Bold", italic="Sans-Oblique",
                              boldItalic="Sans-Bold")

INK, INK2, RULE, TINT = colors.HexColor("#0b0b0b"), colors.HexColor("#52514e"), colors.HexColor("#d9d8d2"), \
    colors.HexColor("#f4f3ef")
BLUE = colors.HexColor("#2a78d6")
S = {
    "title": ParagraphStyle("title", fontName="Sans-Bold", fontSize=19, leading=24, textColor=INK, spaceAfter=4),
    "sub": ParagraphStyle("sub", fontName="Sans", fontSize=10.5, leading=14, textColor=INK2, spaceAfter=14),
    "h1": ParagraphStyle("h1", fontName="Sans-Bold", fontSize=13, leading=17, textColor=INK, spaceBefore=14,
                         spaceAfter=6, keepWithNext=1),
    "h2": ParagraphStyle("h2", fontName="Sans-Bold", fontSize=10.5, leading=14, textColor=INK, spaceBefore=8,
                         spaceAfter=4, keepWithNext=1),
    "body": ParagraphStyle("body", fontName="Sans", fontSize=9, leading=13, textColor=INK, spaceAfter=6,
                           alignment=TA_LEFT),
    "bullet": ParagraphStyle("bullet", fontName="Sans", fontSize=9, leading=13, textColor=INK, leftIndent=12,
                             bulletIndent=2, spaceAfter=2),
    "cap": ParagraphStyle("cap", fontName="Sans", fontSize=7.8, leading=10.5, textColor=INK2, spaceBefore=3,
                          spaceAfter=10),
    "cell": ParagraphStyle("cell", fontName="Sans", fontSize=7.8, leading=10, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="Sans-Bold", fontSize=7.8, leading=10, textColor=INK),
    "code": ParagraphStyle("code", fontName="Mono", fontSize=7.6, leading=10, textColor=INK, leftIndent=8,
                           backColor=TINT, borderPadding=5, spaceBefore=4, spaceAfter=8),
    "box": ParagraphStyle("box", fontName="Sans", fontSize=9, leading=13, textColor=INK),
}
W = A4[0] - 4 * cm


def P(text, style="body"):
    return Paragraph(text, S[style])


def bullets(items):
    return [Paragraph(t, S["bullet"], bulletText="•") for t in items]


def code(text):
    return Paragraph(text.replace("\n", "<br/>").replace(" ", "&nbsp;"), S["code"])


def fig(name, caption, width=W):
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(str(FIG / name)).getSize()
    img = Image(str(FIG / name), width=width, height=width * ih / iw)
    return KeepTogether([img, P(caption, "cap")])


def table(rows, widths, header=True, bold_col0=False):
    data = [[P(str(c), "cellb" if (header and i == 0) or (bold_col0 and j == 0) else "cell")
             for j, c in enumerate(r)] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
          ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), TINT), ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK2)]
    t.setStyle(TableStyle(st))
    return t


def callout(flowables):
    t = Table([[flowables]], colWidths=[W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eef4fc")),
                           ("LINEBEFORE", (0, 0), (0, -1), 2.5, BLUE),
                           ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                           ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    return t


def results():
    st = pd.read_parquet(D / "stations.parquet")
    base = pd.read_parquet(D / "baseline_scores.parquet")
    cnn = pd.read_parquet(D / "cnn_scores.parquet")
    pts = targets(D / "artefacts.gpkg")
    scorers = [
        ("Random ranking", None),
        ("Rule: BD TOPO parallax prior", base["rule"]),
        ("GBM, context features", base["gbm_context"]),
        ("GBM, image features", base["gbm_image"]),
        ("GBM, image + context", base["gbm_all"]),
        ("SeamNet", cnn["cnn"]),
        ("SeamNet + GBM (rank average)", rankdata(cnn["cnn"]) + rankdata(base["gbm_all"])),
    ]
    rng = np.random.default_rng(0)
    out = {}
    for on_seam in (True, False):
        rows = {}
        for name, s in scorers:
            if s is None:
                rr = [evaluate(st, rng.random(len(st)), pts, on_seam) for _ in range(20)]
                rows[name] = {k: np.mean([r[k] for r in rr]) for k in rr[0] if isinstance(rr[0][k], float)}
            else:
                rows[name] = evaluate(st, s, pts, on_seam)
        out[on_seam] = rows
    return st, out


def build():
    st, res = results()
    ops = operating_points(st, pd.read_parquet(D / "cnn_scores.parquet")["cnn"], targets(D / "artefacts.gpkg"))
    n_pos = int((st.label == 1).sum())
    km = len(st) * config.STRIDE / 1000
    on, al = res[True], res[False]
    cnn5, gbm5, rnd5 = on["SeamNet"]["R@5/km"], on["GBM, image + context"]["R@5/km"], on["Random ranking"]["R@5/km"]
    n_on, n_all = on["SeamNet"]["n_instances"], al["SeamNet"]["n_instances"]
    story = []

    story += [
        P("Seamline artefact detection on CRAIG orthophotos", "title"),
        P(f"seam_qa: method, SeamNet architecture, dataset and synthetic data, and first results on the "
          f"Eure-et-Loir nord 2025 sample · {datetime.date.today():%d %B %Y}", "sub"),
        callout([
            P("<b>Summary</b>", "box"), Spacer(1, 3),
            P("Mosaicking errors (bâtiment coupé, véhicule coupé, décalage, erreur de mosaïquage) happen "
              "<i>on</i> the seamline, where two source photos meet. seam_qa samples every seamline into "
              "<b>stations</b> every 4 m and scores each one by comparing the two photos at that point. The "
              "operator then reviews the highest-scoring stations in QGIS.", "box"), Spacer(1, 3),
            P(f"<b>Result.</b> In spatially grouped cross-validation, SeamNet finds <b>{cnn5:.0%}</b> of the "
              f"on-seam artefacts when the operator reviews 5 candidates per km of seam. Gradient boosting on "
              f"hand-made features finds {gbm5:.0%} and random ranking {rnd5:.0%}. At 20 candidates per km, "
              f"SeamNet finds {on['SeamNet']['R@20/km']:.0%}.", "box"), Spacer(1, 3),
            P(f"<b>Caveat.</b> These are {n_on} artefacts from one campaign, in tiles centred on artefacts. The "
              "numbers show the approach works. They are not yet a production estimate.", "box"),
        ]),
    ]

    story += [P("1. The problem and the data", "h1"),
              P("CRAIG's contrôle visuel (see <i>tuto_cv.pdf</i>) is done by hand: operators place a point on "
                "each artefact, and SG decides which ones are sent back to the vendor. The goal is to automate "
                "finding, locating and classifying the errors caused by mosaicking. The first deliverable is a "
                "ranked review list; the long-term target is fully automatic flagging."),
              table([
                  ["Item", "Content"],
                  ["Ortho tiles", "200 m × 200 m GeoTIFF, RGBA, ~5 cm, EPSG:2154; one tile per artefact, named by "
                                  "art_id. 44 tiles here; ~70 GB over 3 campaigns in total."],
                  ["Seamline graph", "GrapheMosaiquage_EureLoir gpkg: 31,902 polygons in EPSG:3948 (CC48), one "
                                     "per source photo. Image = strip_photo (170 flight strips)."],
                  ["Artefact points", "cv_artefact: 1,253 points (vallées Cher-Loir-Loire 1,027, Indre sud 168, "
                                      "Eure-et-Loir nord 58), all sent back (art_renvoi = 1)."],
                  ["BD TOPO", "Buildings (with hauteur) and road sections, fetched per tile from the IGN WFS."],
              ], [3.2 * cm, W - 3.2 * cm]),
              Spacer(1, 8),
              P("Two measurements shape the whole design. First, <b>seam errors sit on the seam</b> (Figure 1): "
                "every véhicule coupé point is within 2.2 m of a seamline, and most bâtiment coupé points within "
                "3 m. Hotspots, cloud shadows and the unnamed types 3, 5, 15 and 21 sit 5–80 m away. They are a "
                "different problem and are left out of the seam model."),
              fig("distance.png", "Figure 1. Distance from each Eure-et-Loir artefact point to the nearest "
                  "seamline. Blue: seam-caused types used as targets. Grey: other types. Log scale."),
              P(f"Second, <b>the imbalance is extreme</b>. The Eure-et-Loir mosaic holds about 29,700 km of seams "
                "for about 38 seam artefacts, roughly one per 800 km. Inside the tiles it is better: about 466 m "
                f"of seam per tile and usually one artefact. Here that gives {len(st):,} stations ({km:.0f} km) "
                f"and {n_pos} positive stations, about 1 : 110."),
              ]

    story += [P("2. Why the earlier attempts struggled", "h1"),
              P("The earlier models (~/Downloads/CRAIG) were a YOLO hotspot detector and a DeepLabV3+ segmenter "
                "on 256 px good/bad patches along the seamlines. Neither has recorded metrics; the report calls "
                "them \"pas encore optimales\". The table lists the likely causes and what seam_qa changes."),
              table([
                  ["Earlier approach", "What changes in seam_qa"],
                  ["The network does not know where the seam is, nor which side is which photo; patches have "
                   "arbitrary orientation.",
                   "Crops are rotated so the seam is vertical, plus a 4th channel holding the signed distance to "
                   "the seam."],
                  ["One 256 px scale.", "Two scales: 5 cm for detail (vehicles), 10 cm for context (buildings)."],
                  ["Pixel masks needing near-millimetre precision.",
                   "One label per station (within 3 m of a point), with an ignore band around each point."],
                  ["Every patch along every line, most of them trivially clean.",
                   "Context features (BD TOPO, parallax) feed the ranking; synthetic positives add variety."],
                  ["Few positives.", "synth.py builds artefacts on clean seams, always on one side only."],
                  ["No honest metric.",
                   "Grouped CV by area, point matching, recall at a review budget (detections per km of seam)."],
              ], [W / 2, W / 2]),
              ]

    story += [PageBreak(), P("3. Pipeline overview", "h1"),
              fig("pipeline.png", "Figure 2. The seam_qa pipeline. Blue: scripts. Orange: synthetic data. "
                  "Green: the QGIS review layer, whose accept/reject decisions can become new labels."),
              table([
                  ["Script", "Role"],
                  ["seam_graph.py", "Seam lines, stations, photos on each side, BD TOPO context, labels; mosaic.vrt"],
                  ["features.py", "25 cross-seam image features per station"],
                  ["baseline.py", "Rule and gradient-boosting baselines, grouped CV, gbm_all.joblib"],
                  ["crops.py / dataset.py", "Seam-aligned crop reader; two-scale crop cache"],
                  ["synth.py", "Synthetic seam artefacts (4 types)"],
                  ["train.py", "SeamNet: grouped CV, then the final model seamnet.pt"],
                  ["infer.py", "Rank-average scores, NMS, top N per km, export detections.gpkg"],
                  ["metrics.py", "NMS, point matching, recall at budget, AP"],
              ], [3.6 * cm, W - 3.6 * cm]),
              ]

    story += [P("4. Seam graph and stations (seam_graph.py)", "h1"),
              P("<b>Seam lines.</b> Each photo polygon's boundary is shared exactly with its neighbours: in a "
                "test tile, the summed boundary length is exactly twice the merged length. The boundaries of all "
                "polygons touching a tile are therefore noded, merged into lines (shapely <i>unary_union</i> + "
                "<i>linemerge</i>) and clipped to the tile."),
              P("<b>Stations.</b> Every 4 m along each line, a station stores its position and the seam tangent, "
                "estimated over ±3 m. Points 1.5 m to the left and right of the line are tested against the "
                "polygons to find the photo on each side (img_L, img_R) and their flight strips. A station "
                "whose two sides show the same photo, or no photo, is dropped. Stations from overlapping tiles "
                "closer than 2 m are deduplicated."),
              P("<b>Context.</b> From BD TOPO, per station:"),
              *bullets([
                  "whether the seam crosses a building within ±2 m, and that building's height;",
                  "distance to the nearest building;",
                  "margin to the nearest road edge (centreline distance minus half the carriageway width);",
                  "whether the two photos come from different strips.",
              ]),
              P("A physical prior completes the context. A building of height <i>h</i> is displaced differently "
                "in the two photos. Approximating each photo's nadir by its polygon centroid, the mismatch is "
                "about <i>h × baseline / H</i>, where the baseline is the distance between the two nadirs and "
                "<i>H</i> ≈ 1,250 m is the flying height. Expressed in pixels, this <b>parallax_px</b> predicts "
                "how badly a building is cut."),
              table([
                  ["Label", "Rule", "Stations"],
                  ["positive (1)", "within 3 m of a seam-type point (types 2, 8, 13, 16)", str(n_pos)],
                  ["ignored (−1)", "3–30 m from a seam-type point, or within 10 m of any other point",
                   str(int((st.label == -1).sum()))],
                  ["negative (0)", "everything else in the tiles (weak: the operator looked here and flagged "
                                   "nothing)", str(int((st.label == 0).sum()))],
              ], [2.6 * cm, W - 5.6 * cm, 3.0 * cm]),
              Spacer(1, 6),
              fig("tile.png", "Figure 3. Tile 3788 (véhicule coupé). Left: seam lines (white), stations and "
                  "their labels; the yellow stars are the two artefact points, merged into one instance. Right: "
                  "SeamNet's out-of-fold score for each station."),
              ]

    story += [P("5. Seam-aligned crops (crops.py, dataset.py)", "h1"),
              P("For every station, a square crop is resampled from a VRT of all the campaign's tiles. The crop "
                "is rotated so that walking up the crop follows the seam tangent, with the left photo on the "
                "left. The mapping from crop pixel to source pixel is a pure rotation, applied in a single "
                "<i>cv2.warpAffine</i> after a window read at the output resolution."),
              P("The seam lines near the station are drawn into the crop. The two regions on either side of the "
                "station (connected components) give the <b>side map</b>: +1 for the left photo, −1 for the "
                "right, 0 for any third photo. Combined with the distance transform, this gives the 4th input "
                "channel: side × min(distance, 32 px) / 32."),
              fig("channels.png", "Figure 4. One station (tile 3766, véhicule coupé). The same point at 5 cm and "
                  "10 cm, and the signed-distance channel. The seam need not be straight; the channel follows the "
                  "real line."),
              P("dataset.py caches both scales for every station (192 × 192 × 5 bytes per scale: RGB, side, "
                "distance), which takes 46 s and 1.8 GB here. For full campaigns, crops are read on the fly "
                "instead (about 33 ms per crop)."),
              ]

    story += [P("6. Hand-made features and gradient boosting (features.py, baseline.py)", "h1"),
              P("The interpretable baseline uses a 256 px, 5 cm crop. In every crop row it finds the seam column "
                "(the left-to-right side change) and compares pixels just beside it:"),
              table([
                  ["Group", "Features", "Targets"],
                  ["Profile match", "Correlation and median-corrected mean absolute difference between the grey "
                                    "profiles 3–9 px left and right of the seam (whole crop, and central half)",
                   "cuts, offsets"],
                  ["Shift search", "Correlation of the along-seam gradient profiles at lag 0 vs the best lag "
                                   "(±2 m), the gain, and |lag|", "décalage"],
                  ["Seam edge", "Gradient across the seam vs 8–12 px beside it (mean ratio, p90)", "visible cut"],
                  ["Colour", "ΔE in Lab and ΔL between 2 m bands on each side; std ratio", "raccord"],
                  ["Objects", "98th-percentile saturation and 99th-percentile brightness, per side and their "
                              "difference", "vehicles"],
                  ["Context", "bldg_cross, bldg_h, bldg_dist, road_margin, on_road, inter_strip, baseline_m, "
                              "parallax_px, ...", "prior"],
              ], [2.4 * cm, W - 4.6 * cm, 2.2 * cm]),
              Spacer(1, 6),
              P("Model: sklearn HistGradientBoostingClassifier (depth 3, 150 iterations, learning rate 0.05, "
                "min 20 samples per leaf, balanced class weights), trained on labelled stations only."),
              ]

    story += [P("7. Synthetic artefacts (synth.py)", "h1"),
              P("With about 40 real positives per campaign, the network would mostly memorise them. synth.py "
                "creates positives from <b>clean</b> training stations, applying one rule: the change touches "
                "<b>one side of the seam only</b>. That is the definition of a mosaicking error: the two photos "
                "disagree at the seam. Every artefact's geometry is drawn in metres and applied identically to "
                "both scales."),
              table([
                  ["Generator", "What it does", "Parameters", "Mimics"],
                  ["object_cut", "A blob straddling the seam is displaced on one side only",
                   "half-axes 2–8 m, shift 0.3–1.5 m, random direction", "bâtiment coupé"],
                  ["vehicle_cut", "A car exists on one side only (it moved between exposures)",
                   "1.6–2.0 × 3.8–5.0 m, 7 colours, windows, soft shadow", "véhicule coupé"],
                  ["shift", "The whole side is offset along/across the seam",
                   "along ±0.15–2 m, across ±0.3 m", "décalage, mosaïquage"],
                  ["radiometric", "Gain/offset jump between the two sides",
                   "gain 0.95–1.05, one channel ±6–10 %, offset ±6", "raccord radiométrique"],
              ], [2.2 * cm, 5.0 * cm, 5.4 * cm, W - 12.6 * cm]),
              Spacer(1, 6),
              fig("synth.png", "Figure 5. One example of each generator: the clean crop, then the synthetic "
                  "artefact at both scales. The dotted line is the crop centre. Like real ones, some synthetic "
                  "artefacts are hard to see.", width=W * 0.78),
              P("This generalises the cut-shift augmentation from ~/Downloads/CRAIG/patch_tif_mosaic.ipynb, "
                "which drew random curved lines inside a patch, to the real seam of every crop. Generator weights "
                "are 0.30 / 0.30 / 0.25 / 0.15."),
              ]

    import torch  # noqa: F401
    from train import SeamNet
    n_par = sum(p.numel() for p in SeamNet(pretrained=False).parameters()) / 1e6

    story += [PageBreak(), P("8. SeamNet", "h1"),
              fig("seamnet.png", f"Figure 6. SeamNet ({n_par:.1f} M parameters). One backbone sees both "
                  "scales; their pooled features are concatenated before the two heads."),
              P("<b>Architecture.</b> A ResNet-18 (timm, ImageNet weights) with a 4-channel stem: timm adapts "
                "the pretrained RGB filters to 4 inputs. It runs on the 5 cm and 10 cm crops <i>with shared "
                "weights</i>, so the same filters learn \"what a discontinuity looks like\" at two zoom levels "
                "without doubling the parameters. Global average pooling gives 512 features per scale; the "
                "concatenated 1,024 go through Linear(256), ReLU and Dropout(0.3) into two heads:"),
              *bullets([
                  "<b>error</b>: one logit, \"is there a seam error at this station\", trained with binary "
                  "cross-entropy. This is the ranking score.",
                  "<b>type</b>: four logits (bâtiment coupé, véhicule coupé, décalage / mosaïquage, radiométrie), "
                  "trained with cross-entropy on positives only (weight 0.5). Real labels map types 2→0, 8→1, "
                  "13 and 16→2; synthetic ones take their generator's class.",
              ]),
              P("<b>Why this design.</b> Telling the network where the seam is and which photo is on which "
                "side turns the task into a <i>comparison</i>: do the two sides agree? That is far easier to "
                "learn from a few examples than finding an artefact anywhere in an arbitrary patch. The task is "
                "framed as classification of a station, not segmentation, because the labels are points, not "
                "outlines."),
              P("<b>Training recipe.</b>", "h2"),
              table([
                  ["Setting", "Value"],
                  ["Batch (32)", "8 real positives + 8 synthetic positives (from training negatives) + 16 negatives"],
                  ["Augmentation", "Flip along the seam; flip across it (which swaps the side code); "
                                   "brightness and contrast jitter"],
                  ["Optimiser", "AdamW, weight decay 1e-4, one-cycle schedule up to 3e-4"],
                  ["Length", "250 steps per fold (CPU, about 7–8 min per fold here); the final model trains on "
                             "all stations"],
                  ["Inference", "Probabilities averaged over the identity and an along-seam flip"],
              ], [3.0 * cm, W - 3.0 * cm]),
              ]

    story += [P("9. Evaluation protocol", "h1"),
              *bullets([
                  "<b>Spatially grouped CV.</b> Tiles within 1.5 km of each other form one group (27 groups), "
                  "split into 5 folds. Overlapping tiles such as 339/340 can never land on both sides. Every score "
                  "in Section 10 is out-of-fold.",
                  "<b>Artefact instances.</b> Same-type points within 20 m are merged (339/340, 3559/3560, "
                  "3788/3789, ...). This leaves 34 seam-type instances in the tiles, 29 of them within 5 m of a "
                  "seam (\"on-seam\").",
                  "<b>Detections.</b> Station scores go through non-maximum suppression with an 8 m radius. A "
                  "detection finds an instance if it lies within 10 m of one of its points.",
                  "<b>Review budget.</b> The top <i>N</i> detections, with <i>N</i> = budget × km of seam. "
                  "Recall at 5/km means: the operator reviews 5 candidates per km of seam, and this share of the "
                  "artefacts is among them.",
                  "<b>AP.</b> Instance-level average precision over the ranked detections. A detection counts as "
                  "a true positive only the first time it finds an instance.",
              ]),
              ]

    def row(name, r):
        return [name] + [f"{r[f'R@{b}/km']:.2f}" for b in (1, 2, 5, 10, 20)] + [f"{r['AP']:.2f}"]

    hdr = ["Scorer", "R@1/km", "R@2/km", "R@5/km", "R@10/km", "R@20/km", "AP"]
    cw = [5.4 * cm] + [(W - 5.4 * cm) / 6] * 6
    story += [P("10. Results", "h1"),
              P(f"<b>On-seam artefacts ({n_on} instances, {km:.0f} km of seam, out-of-fold)</b>", "h2"),
              table([hdr] + [row(k, v) for k, v in on.items()], cw, bold_col0=False),
              Spacer(1, 6),
              fig("recall.png", "Figure 7. Share of on-seam artefacts found as a function of the review "
                  "budget. Random ranking is the mean of 20 draws."),
              P(f"<b>All seam-type artefacts in the tiles ({n_all} instances, including 5 points more than 5 m "
                "from any seam)</b>", "h2"),
              table([hdr] + [row(k, v) for k, v in al.items()], cw),
              Spacer(1, 6),
              ]
    types = [("batiment_coupe", "Bâtiment coupé"), ("vehicule_coupe", "Véhicule coupé"),
             ("decalage", "Décalage"), ("erreur_mosaiquage", "Erreur de mosaïquage")]
    story += [P("<b>Per type, at 5 detections per km (on-seam)</b>", "h2"),
              table([["Scorer"] + [t[1] for t in types]] +
                    [[k] + [on[k].get(f"R@5/km {t[0]}", "–") for t in types]
                     for k in ("Rule: BD TOPO parallax prior", "GBM, image + context", "SeamNet",
                               "SeamNet + GBM (rank average)")],
                    [5.4 * cm] + [(W - 5.4 * cm) / 4] * 4),
              Spacer(1, 6),
              P("<b>Reading the results.</b>", "h2"),
              *bullets([
                  "The BD TOPO parallax rule alone already finds nearly all on-seam bâtiment coupé: when a seam "
                  "crosses a tall building between two distant photos, a cut is likely. It cannot see vehicles at "
                  "all.",
                  "SeamNet's gain comes mainly from véhicule coupé, the most frequent label in the full data (898 "
                  "of the 1,027 vallées points). It needs to <i>see</i> the half-vehicle, which hand-made features "
                  "capture poorly.",
                  "Averaging the SeamNet and GBM ranks helps at small budgets (best at 1–2 per km and best AP). "
                  "SeamNet alone is best from about 4 per km upwards.",
                  f"Uncertainty is large: one artefact is {1 / n_on:.1%} of recall. The ranking of the methods is "
                  "consistent across budgets, but individual values will move with more data.",
              ]),
              fig("top.png", "Figure 8. SeamNet's 12 best detections (out-of-fold, 5 cm crop, seam vertical in the "
                  "centre). Green: an artefact point lies within 10 m. Orange: no point; these are worth a look, "
                  "since only artefacts that were sent back are labelled. The caption under each crop is the "
                  "predicted type."),
              P("Two of the three \"no point\" detections (#5, #6) sit on the same roof with skylights. #10 is a "
                "parked car beside the seam. Some of these may be real but unreported artefacts. With the full "
                "operator data (including art_renvoi = 0), they could be checked."),
              P("<b>How far from fully automatic?</b>", "h2"),
              P("For automatic flagging, what matters is precision: the share of flagged stations that are real "
                "errors. The table keeps SeamNet's top detections at several budgets and gives the score "
                "threshold that implies. Precision is a <i>lower bound</i>, because unreported artefacts count "
                "as false alarms."),
              table([["Top detections per km", "Detections", "Score threshold", "Precision", "Recall"]] +
                    [[f"{r.per_km:g}", str(r.detections), f"{r.threshold:.3f}", f"{r.precision:.0%}",
                      f"{r.recall:.0%}"] for r in ops.itertuples()],
                    [3.8 * cm] + [(W - 3.8 * cm) / 4] * 4),
              Spacer(1, 6),
              P("The most confident detections are mostly real (70–80 % precision at 0.25–0.5 per km), but they "
                "cover under a fifth of the artefacts. Today the model is a <b>triage tool</b>: it cuts the "
                "length of seam an operator has to inspect, while the operator still decides. Automatic flagging "
                "becomes realistic only if precision at useful recall rises with more training data. That can be "
                "measured with transfer mode once the other campaigns are added (Section 11)."),
              ]

    story += [P("11. Limits and next steps", "h1"),
              *bullets([
                  "<b>More data, the decisive step.</b> Add the other two 2025 campaigns to campaigns.yaml and "
                  "run <b>train.py --train A B --test C</b> for each choice of C. Transfer mode trains SeamNet and "
                  "the GBM on two campaigns, scores the third, and writes recall, AP and the operating points to "
                  "out/C/transfer_from_A+B.md. It has been tested end to end; crops of campaigns without a cache "
                  "are read on the fly. No other local data could stand in: the historical tiles in "
                  "~/Downloads/CRAIG have lost their seamline geometry (.shp files missing), and the GAM 2025 test "
                  "seamlines have no matching imagery.",
                  "<b>Train on GPU.</b> Use the 4×T4 VM: longer schedules, a larger backbone (ConvNeXt-T), and "
                  "crops read on the fly. The local AMD GPU fails with a HIP error.",
                  "<b>Weak negatives.</b> Only artefacts SG sent back are labelled, so some negatives are "
                  "unreported artefacts. An export of art_renvoi = 0 and cv_zone_suivi would fix the labels and "
                  "the precision estimate.",
                  "<b>Production false-positive rate.</b> Every tile is centred on an artefact. Before going "
                  "fully automatic, the false-alarm rate per km must be measured on one complete block (all tiles "
                  "plus its seamlines).",
                  "<b>Not built yet.</b> A 1-D model over consecutive stations (NMS stands in for it), "
                  "hard-negative mining, and an id for raccord radiométrique (missing from the 2025 export).",
                  "<b>Hotspots and cloud shadows</b> are not seam errors; Section 12 covers them. Their rules are "
                  "fitted on 5 and 3 examples, so they need more labelled data before their numbers mean much.",
              ]),
              ]

    story += [P("12. Using it: detect.py, with hotspots and cloud shadows", "h1"),
              P("One command takes tiles (GeoTIFF, Lambert-93, about 5 cm, for example 4000 × 4000 px) and a "
                "seamline file (.shp or .gpkg; polygons with photo names, or plain polylines) and writes "
                "<b>errors.gpkg</b>: one point per error in the cv_artefact schema. It also writes review layers "
                "and a per-tile summary. Polylines work because SeamNet's seam channel comes from the drawn line, "
                "not from photo names. On the same 33 tiles, the polygon and polyline runs give the same 73 errors."),
              code('python detect.py --tiles "/data/tiles/*.tif" --seamlines seamlines.shp --out /data/results\n'
                   '                 [--checks seams,hotspots,clouds] [--threshold 0.7] [--no-bdtopo]'),
              table([
                  ["Check", "Method", "Eure-et-Loir, 44 tiles"],
                  ["seams", "SeamNet probability ≥ threshold at stations every 4 m, after 8 m NMS",
                   "out-of-fold at 0.7: ~3.8 errors/km, precision ≥ 41 %, recall 71 %"],
                  ["hotspots", "all glints (saturated cores), then the reportable ones: core ≥ 2 m² with a glow "
                               "beyond its outline (bloom)", "5/5 reported, 1 other flag of 797 glints (in-sample)"],
                  ["clouds (experimental)", "≥ 80 m of seam where the same open land is darker on one side, same "
                                            "colour, texture scaled like brightness",
                   "2/3 found (both cut by a seam), 0 other tiles (in-sample)"],
              ], [2.6 * cm, W - 8.1 * cm, 5.5 * cm]),
              Spacer(1, 6),
              P("<b>Hotspots.</b> In CRAIG's usage (tuto_cv pp. 6, 19–22), a hotspot is a <i>specular sun glint</i>: "
                "a saturated white disc, often with star rays, on a roof or a windscreen. It is reported only when "
                "visible at 1/800 and disturbing. Every reported hotspot in the sample is a saturated blob of "
                "2.4–7.5 m² within 1.4 m of the point. But the tiles hold about 24 glints each, so detection is not "
                "the problem: <i>reportability</i> is. That explains the earlier YOLO's 3,640 boxes on the GAM test "
                "zone. Large white objects (roofs, tents, skylights, pools, a bus) are just as saturated. What "
                "separates a glint is <b>bloom</b>, light spreading beyond the outline, measured with a distance "
                "transform from the core so it works for any shape. The antisolar-geometry idea was tested and "
                "dropped: without exposure positions and times, the offsets point in every direction. With 20 or "
                "more labelled hotspots, hotspot.py trains a classifier on the 18 blob features to replace the "
                "rule."),
              P("<b>Cloud shadows.</b> With 3 examples, the method has to be unsupervised. A shadow belongs to "
                "one photo and becomes an artefact where the mosaic exposes it, typically along a seam (one "
                "operator's comment: \"déplacer la ligne de mosaiquage\"). Along each seam, 20 m segments compare "
                "the open land 3–25 m on each side, away from BD TOPO buildings and roads. A shadow segment is "
                "darker by ≥ 12 % with the same chromaticity, and its texture contrast is reduced in proportion. "
                "Only long stretches count: the two shadows cut by a seam span 111 m and 130 m, and nothing else "
                "exceeds 51 m. The error point lies on that seam stretch, where the fix applies. Small shadows away "
                "from seams (3498, about 20 m across) are missed. Large dark patches go to the review layer only, "
                "because alone they do not separate shadows from darker fields."),
              ]

    story += [P("Appendix: running it", "h1"),
              P("From CRAIG ARTIFACT/seam_qa with the image_analysis/roof venv. In that venv, osgeo.osr "
                "segfaults; the code uses pyproj and rasterio only."),
              code("python detect.py --tiles \"tiles/*.tif\" --seamlines s.shp --out results   # new data, all checks\n"
                   "python seam_graph.py                  # stations, context, labels, mosaic.vrt\n"
                   "python features.py --workers 8        # cross-seam features\n"
                   "python baseline.py                    # rule + GBM, grouped CV, gbm_all.joblib\n"
                   "python dataset.py                     # two-scale crop cache (small campaigns)\n"
                   "python train.py --steps 250 --final   # SeamNet CV + final model (--device cuda on the VM)\n"
                   "python train.py --train A B --test C --steps 2000 --device cuda   # leave-one-campaign-out\n"
                   "python infer.py --campaign C --models-from model_A+B --budget 10\n"
                   "python hotspot.py --campaign C        # glints, rule; trains the classifier with >= 20 labels\n"
                   "python cloud.py --campaign C          # cloud-shadow candidates and report\n"
                   "python docs/figures.py && python docs/make_report.py"),
              P("Main constants (config.py): STRIDE 4 m; positive radius 3 m; ignore radius 30 m; match radius "
                "10 m; instance merge radius 20 m; NMS radius 8 m; flying height 1,250 m; crops 192 px at 5 and "
                "10 cm."),
              ]

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Sans", 7.5)
        canvas.setFillColor(INK2)
        canvas.drawString(2 * cm, 1.2 * cm, "seam_qa: seamline artefact detection")
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, str(doc.page))
        canvas.restoreState()

    doc = SimpleDocTemplate(str(DOCS / "seam_qa_report.pdf"), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.8 * cm, bottomMargin=2 * cm, title="Seamline artefact detection (seam_qa)",
                            author="seam_qa")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print("wrote", DOCS / "seam_qa_report.pdf")


if __name__ == "__main__":
    build()
