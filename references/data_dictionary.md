# Data dictionary

## Source

Virtual KITTI 1.3.1 (Gaidon, Wang, Cabon and Vig, "Virtual Worlds as Proxy for Multi-Object Tracking Analysis", CVPR 2016): five of the KITTI tracking videos rebuilt in the Unity game engine, each rendered ten ways. The brief links the [Dataset Ninja copy](https://datasetninja.com/virtual-kitti#download), a single 29.5 GB tar in Supervisely format that holds the RGB, depth and optical-flow rendering of every frame, each with a copy of the same annotation file. The project uses the RGB frames and their annotations. The archive's own license file puts Virtual KITTI under [CC BY-NC-SA 3.0 US](https://creativecommons.org/licenses/by-nc-sa/3.0/).

| World | Moments | What it shows | Partition |
|---|---|---|---|
| `0001` | 447 | a town street with buildings and parked cars | test |
| `0002` | 233 | an intersection among trees | train |
| `0006` | 270 | a wide avenue with traffic lights and many signs | train |
| `0018` | 339 | a road through dense trees | validation |
| `0020` | 837 | a highway with guard rails and overhead gantries | train |

| Rendering (`variation`) | What changes |
|---|---|
| `clone` | the reconstruction of the real KITTI video |
| `15-deg-left`, `15-deg-right`, `30-deg-left`, `30-deg-right` | the camera turned by that angle, so the view changes |
| `fog`, `morning`, `overcast`, `rain`, `sunset` | weather or light only; the labels are identical to the clone's |

The 14 classes, in the order the label maps index them: building, car, guard rail, misc, pole, road, sky, terrain, traffic light, traffic sign, tree, truck, van, vegetation. Pixels without a class are 255.

## Raw files (`data/raw/virtual_kitti/`, not tracked)

The frames and annotations take about 15 GB, so they are not in the repository. `download_virtual_kitti()` in `src/driving_scenes/download.py` streams the archive once and the notebook calls it first.

| File | Contents |
|---|---|
| `img/rgb_<world>_<variation>_<step>.png` | one RGB frame, 1242 x 375 pixels |
| `ann/rgb_<world>_<variation>_<step>.png.json` | its annotation: image size, tags (`sequence` = world, `rotation` = rendering, `im_id`) and a list of objects, each one connected region of one class stored as a bitmap (a zlib-compressed PNG in base64 with the position of its top-left corner); vehicles also carry tracking rectangles, which the project ignores |
| `meta.json` | the 14 classes and their colors |
| `download_offset.txt` | the download's resume position, `done` when finished |

## Derived caches (`data/interim/` and `models/`, not tracked)

Every cached result carries a fingerprint of what produced it (the frames and their labels, the starting weights, every setting and the seed): `fingerprint.txt` inside a run folder, `<file>.fingerprint` beside a cached file. A result is reused only when the fingerprint of the current request matches, so a change to any input recomputes it instead of reloading a stale result under the same name.

| Path | Contents |
|---|---|
| `data/interim/labels/<frame_id>.png` | one label map per frame: the class index of every pixel (0 to 13), 255 where unlabeled |
| `data/interim/frames.csv` | one row per frame, columns below |
| `data/interim/signs.csv` | one row per traffic-sign region, columns below |
| `data/interim/predictions/<run>_<frames>.npz` | a run's predictions on a frame set (`test_moments`, `validation`, `test`): every detection's box, confidence and frame, and the road counts (true and false positives and negatives) of every frame |
| `data/interim/road_calls/{validation,test}.csv` | for the final network on one rendering per moment of the validation and test worlds: every class's labeled pixels and how many of them the network calls road |
| `models/runs/<run>/` | one training stage: `best.ckpt` (the weights of the kept epoch), `metrics.csv` (Lightning's log), `pulls.csv` (each task's measured pull and weight), `settings.json` (the stage's settings and its minutes), `fingerprint.txt` |

Run names: `heads_seed<N>` and `recipe_seed<N>` are the two stages of the recipe with seed N; `sum`, `signs_only`, `road_only`, `frozen`, `lr_0.0025` and `lr_0.01` are stage two with one setting changed, from the seed-42 heads; `coco_anchors_heads_seed42` and `coco_anchors_recipe_seed42` are the recipe with COCO's anchors; `test_0002_0006_*` and `test_0020_*` are the recipe trained with those worlds held out.

`frames.csv`:

| Column | Meaning |
|---|---|
| `frame_id` | `<world>_<variation>_<step>` |
| `world`, `variation`, `step` | the video, its rendering, the frame's position in the video |
| `file` | the frame's file name under `img/` |
| `width`, `height`, `labels_match_frame` | frame size, and whether the label map has the same size |
| `brightness`, `contrast` | mean and standard deviation of the luma (ITU-R BT.601 weights), 0 to 255 |
| `red`, `green`, `blue` | mean of each channel |
| `unlabeled_pixels` | pixels without a class |
| `signs` | traffic-sign regions in the frame |
| `overlapping_pixels` | pixels claimed by two regions (0 everywhere) |
| `tagged_world`, `tagged_variation` | the annotation's own tags, checked against the name |
| `pixels_<class>` | pixels of each of the 14 classes |

`signs.csv`:

| Column | Meaning |
|---|---|
| `frame_id` | the frame |
| `x_min`, `y_min`, `x_max`, `y_max` | the region's tight box in pixels, maxima exclusive |
| `pixels` | the region's own pixels |

## Results tables (`data/processed/`, tracked, written by the notebook's last cell)

Every file below is written by the notebook's last cell; scores are fractions between 0 and 1 unless a column says otherwise. AP columns follow COCO (`sign_ap50` at an IoU of 0.5, `sign_map` averaged over IoU 0.5 to 0.95, `sign_ap75`, AP by size class, `sign_ar100` the average recall with up to 100 detections per frame); road columns count labeled pixels only (`road_iou`, `miou` the mean of the road and not-road IoU, `pixel_accuracy`, `road_precision`, `road_recall`).

| File | Contents |
|---|---|
| `worlds.csv` | per world: `moments`, `share of moments`, `frames`, `renderings`, `signs per frame`, `frames with a sign` (share), `road share` (share of all pixels) |
| `partitions.csv` | the same per partition, with the `worlds` it holds |
| `sign_regions.csv` | `measure` and `value`: the sign regions' count, median box, elongation, fill, COCO size shares, shares cut off by the frame or with a close neighbor, smallest region |
| `anchor_coverage.csv` | per candidate anchor set (`anchor_set`): the median best-anchor IoU of the training sign boxes and the share of boxes whose best anchor reaches IoU 0.3, 0.5 and 0.7 |
| `rendering_agreement.csv` | per rendering (`variation`): mean road-mask IoU with the clone at the same moment (`road_iou`) and the share of the clone's sign boxes matched at IoU 0.5 (`signs_matched`) |
| `frame_repetition.csv` | per training world and lag in frames: mean road-mask IoU between frames that far apart (`road_iou`) and the share of sign boxes still matched (`signs_still_matched`) |
| `mask_ceilings.csv` | per sampled training frame: the best road IoU a semantic head (stride 4) and a 28 x 28 mask head could reach |
| `parameters.csv` | per part of the network: its weights and how many of them stage one and stage two train |
| `training_history.csv` | per stage and epoch of the recipe with seed 42: training losses per term, validation losses, validation AP50, mAP, road IoU, mean IoU, pixel accuracy and `val_score` (the mean of AP50 and road IoU) |
| `pulls.csv` | every 10th step of stage two: each task's pull (gradient norm at the pyramid outputs) and the weight its gradient carried |
| `validation_runs.csv` | per run: the kept `epoch`, the training `minutes`, and the validation scores at that epoch |
| `variant_verdicts.csv` | per candidate: its `val_score`, how far it is above the recipe's seed mean, the recipe's seed range, and whether it `replaces the recipe` |
| `test_runs.csv` | per run: every score on one rendering per moment of the test world (447 frames) |
| `final_test_scores.csv` | per `score`: the network carried forward on all 4,470 test frames, beside the fixed road mask baseline |
| `scores_by_rendering.csv` | the network carried forward per rendering of the test world |
| `threshold_sweep.csv` | per confidence `threshold` on validation: `precision`, `recall` and `f1` of the sign detections |
| `reliability.csv` | per confidence `bin` on validation: `detections`, `mean confidence`, `hit rate` |
| `sign_sizes.csv` | per COCO `size` class of the test signs: `signs`, `found` and `recall` at the chosen threshold, and COCO `ap` |
| `sign_recall_by_flag.csv` | per `flag` (`at_border`, `fragment`) and `value`: `signs`, `found`, `recall` |
| `unseen_worlds.csv` | per `world`: every score of a network that never trained on it, on one rendering per moment |
| `road_calls.csv` | per frame set (`frames`) and `true class`: labeled `pixels`, pixels `called road`, `share called road` and `share of false road` |
| `latency.csv` | per `network`: timed `frames`, `median_ms` and `p95_ms` per frame on the GPU |
| `frame_scores.csv` | per test frame: `signs`, `found`, `missed`, `false_alarms`, `errors`, `sign_f1`, `road_iou`, `combined` (their mean), `variation`, `step` |
| `worst_frame_renderings.csv` | per `rendering`: its share of the worst 5% of test frames by combined score and of all test frames |
