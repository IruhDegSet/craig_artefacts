# seam_qa: automatic error detection on CRAIG orthophotos

You give it orthophoto tiles and their seamlines; it returns the errors as a QGIS layer in the `cv_artefact` schema. Three checks:

| Check | Errors | Method | On the 44 Eure-et-Loir tiles |
|---|---|---|---|
| `seams` | bâtiment coupé, véhicule coupé, décalage / erreur de mosaïquage | SeamNet (deep model on seam-aligned crops) at stations every 4 m along the seams | out-of-fold, threshold 0.7: recall 71 %, precision ≥ 41 % |
| `hotspots` | hotspot (specular sun glint) | every glint found, then the ones an operator would report: saturated core ≥ 2 m² with a glow beyond its outline | 5/5 reported hotspots, 1 other flag out of 797 glints (in-sample) |
| `clouds` | ombre de nuage (**experimental**) | ≥ 80 m of seam where the same open land is darker on one side | 2/3 (both shadows cut by a seam), 0 other tiles (in-sample) |

## Detect errors on new tiles

From this folder, with the `image_analysis/roof` venv:

```bash
python detect.py --tiles "/data/tiles/*.tif" --seamlines /data/seamlines.shp --out /data/results
```

**Inputs:**
- `--tiles`: GeoTIFFs, RGB or RGBA, about 5 cm; for example 4000 × 4000 px = 200 m. Globs or paths, several allowed.
  - Lambert-93 (EPSG:2154) tiles are read as they are.
  - Tiles in any other projection, such as lot 3 in CC47, are read through a warped VRT in Lambert-93, written to `--out/l93/`. This is a few KB per tile: the pixels are resampled (bilinear, same resolution) when read, and no copy is written.
- `--seamlines`: one or more `.shp` or `.gpkg`, each in any projection; a block delivered in parts (lot 3 Nord + Sud) is given as both files.
  - **Polygons, one per source photo** (the mosaic graph): best. The photo names (`--image-field`, default `Image`) give the flight strip and the parallax context.
  - **Polylines** (the seams themselves) also work. Photo context is then unknown, and a GBM trained without it is used. The automatic decision relies on SeamNet, which doesn't need photo context.
  - Not needed with `--checks hotspots`.

**Outputs, in `--out`:**
- `errors.gpkg`: **the automatic result**, one point per error: `art_arttype_id` (2/8/13 seams, 10 hotspot, 14 cloud shadow), type, score, tile and rank. It opens in the contrôle visuel QGIS project.
- `review.gpkg`, for an operator:
  - `cv_artefact`: ranked seam candidates, `--budget` per km of seam;
  - `hotspot_candidates`: every glint with its features;
  - `ombre_de_nuage_stretches` / `ombre_de_nuage_patches`: all shadow candidates.
- `scored.gpkg`: every seam station with its probabilities.
- `summary.csv`: per tile, seam length and errors by type.

**Options:**
- `--checks seams,hotspots,clouds` (default: all).
- `--threshold 0.7`: SeamNet probability needed to flag a seam error. The trade-off below is out-of-fold on Eure-et-Loir; precision is a lower bound, because unreported artefacts count as false alarms.

  | threshold | errors / km | precision | recall |
  |---|---|---|---|
  | 0.5 | 9.1 | 20 % | 76 % |
  | 0.7 | 3.8 | 41 % | 71 % |
  | 0.8 | 2.2 | 48 % | 47 % |
  | 0.9 | 1.1 | 52 % | 29 % |
  | 0.95 | 0.6 | 73 % | 21 % |

- `--models DIR`: a folder of trained models (default `out/eure_et_loir_nord_2025`).
- `--no-bdtopo`: offline; skips the BD TOPO buildings and roads.
- `--no-cnn`: GBM only.

Speed on this CPU: 44 tiles take about 5 minutes with all checks.

**Errors are refused with a clear message when:**
- a tile has no CRS, or has fewer than 3 bands;
- no tile matches `--tiles`;
- the seamlines don't cover the tiles;
- `--seamlines` is missing for the seams or clouds check.

## How each check works

**Seams.** Each seam is sampled into stations every 4 m. Each station is scored by comparing the two photos that meet there; see the rest of this README and `docs/seam_qa_report.pdf`. Seam errors sit within about 2 m of a seam.

**Hotspots** (`hotspot.py`). In CRAIG's usage (tuto_cv.pdf pp. 6, 19–22), a hotspot is a **specular sun glint**: a saturated white disc, often with star rays, on a roof or a windscreen. It is reported only when visible at 1/800 and disturbing.
- Finding glints is easy: about 24 per tile, 797 on the 44 tiles. The task is deciding which ones an operator would report. That is what the old YOLO lacked: it found every glint, 3,640 boxes on the GAM test zone.
- The cue that separates a glint from a white object (roof, tent, skylight, pool, vehicle) is **bloom**: a glint glows beyond its outline, while a white object ends at a sharp edge.
- The rule is: core ≥ 2 m² and relative bloom ≥ 0.4. Its thresholds were chosen on the same 5 examples it is scored on.
- With ≥ 20 labelled hotspots in a campaign, `python hotspot.py --campaign NAME` trains a classifier on the 18 blob features, under grouped CV. Copy its `hotspot_gbm.*` files into the `--models` folder, and `detect.py` then uses it instead of the rule.

**Cloud shadows** (`cloud.py`, **experimental**). There are only 3 examples locally (1 in the 2020–2024 statistics), so this module is unsupervised.
- A shadow belongs to one photo. It becomes an artefact where the mosaic shows it, typically cut by a seam ("déplacer la ligne de mosaiquage").
- The decision: ≥ 80 m of seam where the land is darker on one side, with:
  - the same colour on both sides (a shadow dims the light, it doesn't change the colour);
  - texture contrast reduced in the same proportion as the brightness;
  - open land only: away from BD TOPO buildings and roads, where gray surfaces and small shadows mislead.
- On Eure-et-Loir, the two shadows cut by a seam have stretches of 111 m and 130 m; every other tile stays at 51 m or less.
- The error point sits on the seam stretch, where the fix applies, 77–108 m from where the operators clicked inside the shadow.
- Not detected: small shadows away from seams, such as 3498, a patch about 20 m across. Dark-patch candidates are written to `review.gpkg` but not flagged: on their own they don't separate shadows from darker fields.

## Training pipeline (seams)

All scripts are run from this folder with the `image_analysis/roof` venv.

| Step | Script | Output in `out/<campaign>/` |
|---|---|---|
| 1. Seam graph, stations, BD TOPO context, labels | `seam_graph.py` | `stations.parquet/.gpkg`, `edges.gpkg`, `artefacts.gpkg`, `mosaic.vrt` |
| 2. Cross-seam image features | `features.py` | `features.parquet` |
| 3. Baselines (rule, GBM) with grouped CV | `baseline.py` | `baseline_scores.parquet`, `baseline_report.md`, `gbm_all.joblib`, `gbm_lines.joblib` |
| 4. Two-scale crop cache (small campaigns) | `dataset.py` | `crops_192.npy` (~0.37 MB per station) |
| 5. SeamNet with grouped CV, then the final model | `train.py --final` | `cnn_scores.parquet`, report section, `seamnet.pt`, `thresholds.json` |
| (hotspots) Glint candidates, rule or classifier | `hotspot.py` | `hotspot_candidates.parquet`, `hotspot_report.md` |
| (clouds) Shadow candidates | `cloud.py` | `cloud_candidates.gpkg`, `cloud_report.md` |
| 6. Review layer for QGIS | `infer.py --campaign X --models-from Y` | `detections.gpkg` (layer `cv_artefact`) |

Campaigns are declared in `campaigns.yaml`: tile globs, seamline file (or a list of files), image field and artefact gpkg. Paths are relative to the yaml; another file can be used with `SEAM_QA_CAMPAIGNS=/path/campaigns.yaml`. Tiles not in Lambert-93 are read through warped VRTs under `cache/l93/<campaign>/`.

The yaml expects this layout, with the repo next to `data/`:

```
<root>/craig_artefacts/                      this repo
<root>/data/jeuDeDonnees_artefacts_2025/
    artefacts_renvoyes_cvl_2025.gpkg         labels of the 3 campaigns (art_campagne)
    lot1_eure-et-loir_nord/<class>/<id>.tif  Lambert-93
    lot2_vallees_cher_loir_loire/<class>/<id>.tif   Lambert-93 (reprojected from CC48 by the producer)
    lot3_indre_sud/<class>/<id>.tif          CC47
    ligne_mosaiquage/lot1_eure-et-loir_nord/GrapheMosaiquage_EureLoir-[3948]_tous_blocs.gpkg
    ligne_mosaiquage/lot2_vallee_cher_loir_loire/Lignes_Mosaiquage_RGF93.shp
    ligne_mosaiquage/lot3_indre_sud/18385_..._Lot3A-Partie{Nord,Sud}_Lignes_de_mosaiquage_CC47.shp
<root>/data/jdd_old/                         older campaigns: seamlines + artefacts_v2.gpkg, no tiles yet
```

| Campaign (`art_campagne`) | Lot | Seam-type points (2 / 8 / 13 / 16) |
|---|---|---|
| `eure_et_loir_nord_2025` | 1 | 17 / 16 / 1 / 2 |
| `vallees_cher_loir_loire_2025` | 2 | 35 / 898 / 5 / 22 |
| `indre_sud_2025` | 3 | 14 / 5 / 8 / 2 |

## Key ideas, and why they differ from the 256 px good/bad patches

- **Seam-aligned crops** (`crops.py`): each crop is rotated so that the seam runs vertically through its centre, with the left photo on the left. A 4th channel holds the signed distance to the seam, so the network knows where the seam is and which side is which.
- **Two scales** (5 cm and 10 cm, 192 px each): fine detail for vehicles, and roughly 19 m of context for buildings.
- **Context** from BD TOPO (`bdtopo.py`):
  - whether the seam crosses a building, and that building's height;
  - distance to the nearest road;
  - inter-strip seams;
  - **parallax risk** = height × baseline ÷ flying height, the predicted size of a building cut in pixels.
- **Labels** at station level:
  - positive: within 3 m of a seam-type point;
  - ignore: 3–30 m from a seam-type point, or within 10 m of another artefact point;
  - negative: everything else inside the tile.
- **Synthetic artefacts** (`synth.py`) on clean stations, always on one side of the seam: object cut, vehicle present on one side only, side offset, and radiometric jump.
- **Honest evaluation** (`metrics.py`):
  - spatially grouped CV, with tiles within 1.5 km kept in the same fold;
  - point-level matching (10 m), after NMS (8 m);
  - **recall at a review budget** in detections per km of seam.

## Results: Eure-et-Loir nord 2025 sample

These are measured on the 44 local tiles: 4,790 stations (19 km of seam) and 34 seam-type artefacts, 29 of them within 5 m of a seam. Folds are 5-fold grouped. With this few artefacts, the numbers are indicative only.

See `out/eure_et_loir_nord_2025/baseline_report.md` for the full table, including SeamNet.

| On-seam artefacts | R@5/km | R@10/km | AP |
|---|---|---|---|
| random | 0.12 | 0.23 | 0.04 |
| rule (BD TOPO parallax only) | 0.34 | 0.34 | 0.07 |
| GBM, context only | 0.24 | 0.38 | 0.06 |
| GBM, image features only | 0.62 | 0.86 | 0.20 |
| GBM, image + context | 0.69 | 0.83 | 0.25 |
| **SeamNet** | **0.86** | **0.93** | 0.48 |
| SeamNet + GBM (rank average) | 0.83 | 0.93 | **0.53** |

- The parallax rule alone finds 10/11 on-seam bâtiment coupé.
- Véhicule coupé: GBM finds 8/15 at 5/km, SeamNet 12/15.
- Toward automatic flagging (SeamNet, out-of-fold, all 34 instances): the top 0.5 detections per km have 70 % precision but 18 % recall; at 2/km it is 53 % / 47 %. Precision is a lower bound, since unreported artefacts count as false alarms. For now this is a triage tool.
- The full write-up is `docs/seam_qa_report.pdf`, rebuilt with `python docs/figures.py && python docs/make_report.py`.

## Scaling to the 3 campaigns (~70 GB)

1. The 3 campaigns of 2025 are in `campaigns.yaml`. Lot 1 now has all its classes, 58 tiles instead of the 44 above, so its numbers change when it is rebuilt.
2. Run `seam_graph.py` and `features.py` for each campaign. Features take about 12 min per 145k stations on 12 cores.
3. Do not cache crops for big campaigns: about 53 GB for 145k stations. `infer.py` reads them on the fly. For training, either cache only positives plus a sample of negatives, or read through `CropReader` in a DataLoader.
4. Evaluate **leave-one-campaign-out**: `train.py --train A B --test C` trains SeamNet and the GBM on A+B (saved to `out/model_A+B/`). It scores C and writes `out/C/transfer_from_A+B.md`, with recall, AP and the operating points (precision and recall at fixed budgets) needed to judge automatic flagging. Then run `infer.py --campaign C --models-from model_A+B`.
5. Train on the 4×T4 VM (`train.py --device cuda`). The local AMD GPU fails with a HIP error, and the CPU is about 4 s per step.

## Not done yet

- The 1-D sequence model over consecutive stations: NMS stands in for it for now.
- Hard-negative mining.
- A raccord radiométrique id (not in the 2025 export).
- Full-campaign false-positive rate. The tiles are centred on artefacts, so moving to fully automatic flagging needs one complete block as a final test.
- Hotspot classifier and cloud-shadow thresholds on more data: both are fitted on the same few examples they are scored on.

Environment note: in the venv, `osgeo.osr` segfaults on `ImportFromEPSG`. This code uses pyproj/geopandas/rasterio only.
