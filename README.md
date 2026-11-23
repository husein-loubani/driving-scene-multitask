# Traffic signs and road in one pass

**Module 4, Sprint 4: Detection and segmentation** | Turing College, Data Science Program

One network that, in a single forward pass, puts a box around every traffic sign and marks every road pixel of a driving frame: the two-task version of the shared-backbone "HydraNets" the brief points to. The network is a Faster R-CNN with a ResNet-50 feature pyramid and COCO weights, with a road head (the semantic branch of Panoptic FPN) on the same pyramid. It trains in two stages, heads first and then the backbone, with each task's gradient into the shared backbone scaled to the same size, and it is tested on a driving world it never saw.

Dataset: [Virtual KITTI 1.3.1](https://datasetninja.com/virtual-kitti#download) (Gaidon et al. 2016), the Dataset Ninja copy the brief links: 21,260 RGB frames of 1242 x 375 pixels from five KITTI driving videos rebuilt in the Unity game engine, each rendered ten ways (the original, four camera angles, fog, morning, overcast, rain and sunset), with a class for every pixel. The frames and their labels take about 15 GB and are not in the repository; the notebook downloads them on its first run, no account needed.

---

## Result

Test world: world 0001, a town street the network never trained on.

| Network | Frames | Sign AP50 | Sign mAP | Road IoU | Road precision | Road recall | ms per frame |
|---|---|---|---|---|---|---|---|
| Fixed road mask (where road usually is, no network) | all 4,470 | | | 0.579 | 0.668 | 0.814 | |
| **Two heads, learning rate 0.01 (carried forward)** | **all 4,470** | **0.656** | **0.415** | **0.827** | 0.832 | 0.993 | 25.3 |
| the same, three seeds | 447 moments | 0.671 to 0.690 | 0.427 to 0.446 | 0.824 to 0.840 | | | |
| Starting recipe (learning rate 0.005), three seeds | 447 moments | 0.639 to 0.652 | 0.404 to 0.421 | 0.817 to 0.829 | | | |
| Signs only, same recipe | 447 moments | 0.692 | 0.473 | | | | 22.9 |
| Road only, same recipe | 447 moments | | | 0.842 | | | 15.6 |
| Heads on the frozen backbone | 447 moments | 0.325 | 0.147 | 0.856 | | | |

The 447 moments are one rendering per moment of the test world, the frames every run is compared on; the network carried forward is also scored on all 4,470 frames. On the validation world the same network reaches an AP50 of 0.953 and a road IoU of 0.981. Latency is the median per frame on an NVIDIA RTX 3090, one frame at a time.

**What the project found:**

- **The world matters more than any choice of network:** every run loses 0.25 to 0.34 of AP50 and 0.07 to 0.18 of road IoU from validation to test, while the runs differ by a few hundredths. Networks tested on the other unseen worlds score 0.370 to 0.665 AP50, so the validation world (a quiet road through trees with few, large signs) is the easy exception rather than the test world being uniquely hard.
- **The road head calls the town's pavement road:** the road is almost never missed (recall 0.993), but 84.4% of the false road is terrain, the sidewalks and a paved square that the grassy training worlds never showed; the worst frames, with the camera turned toward that square, fall to a road IoU of 0.29 to 0.34.
- **Shop signs and billboards cause the false alarms:** at the confidence threshold chosen on validation (0.90), precision falls from 0.934 on validation to 0.587 on test while recall holds at 0.777. Small signs are the other weakness: 52.8% are found, against over 91% of medium and large ones.
- **Equal training is a trade between the tasks:** without balancing, the detector's gradient into the shared backbone is 3 to 12 times the road's. Balancing it lifts the road and lowers the signs at the same validation score (0.958 either way). On the test world each task trained alone beats the shared network (AP50 0.692 against 0.649, road IoU 0.842 against 0.817), while one pass saves 13.1 ms per frame against running both.
- **Fine-tuning the backbone pays on validation and costs the road in the town:** it adds 6.7 points of AP50 and 5.4 of road IoU on validation, yet the frozen backbone has one of the best road IoUs on test (0.856).
- **The learning rate was the one setting that mattered:** among the one-change variants judged by a rule fixed in advance (beat the recipe's three-seed mean by more than its range), doubling the stage-two learning rate to 0.01 won clearly and held over three seeds, and 0.02 was worse, so 0.01 sits inside the bracket. COCO's default anchors passed only just, with higher AP50 but looser boxes.
- **The confidence is not a probability:** the expected calibration error on validation is 0.171; detections above 0.9 are right 93.4% of the time, and those between 0.5 and 0.9 far less often than their confidence says.

## How the project is built

```
download      the Dataset Ninja tar streamed over HTTP, keeping only the RGB frames and their
              annotations; resumes at the last complete file after a dropped connection
index         every frame decoded; every annotation's bitmaps turned into a label map (class per
              pixel, 255 unlabeled); 52,367 sign regions boxed; audit of sizes, tags and overlaps
partitions    by world: train 0002, 0006, 0020 (63% of moments) / validation 0018 (16%) / test 0001
              (21%), so renderings and neighboring frames never cross a partition
exploration   class shares, sign shapes, label agreement across renderings, repetition along a
              video, anchor coverage of the signs, the road ceilings of a mask head and a semantic
              head, a fixed-mask road baseline
network       Faster R-CNN, ResNet-50 FPN, COCO weights, anchors from 16 px with ratios up to 8,
              frames at their native 1242 x 375; Panoptic FPN's semantic branch as the road head
training      stage one: the heads on the frozen backbone (2 epochs); stage two: the FPN and ResNet
              stages 2 to 4 as well (6 epochs); SGD with momentum, warmup then cosine; each task's
              gradient into the backbone scaled to the same norm; one rendering per moment per epoch
comparison    three seeds; one-change variants (plain sum of the losses, learning rates 0.0025 to
              0.02, COCO anchors) judged by a rule fixed in advance; single-task and frozen baselines
evaluation    COCO AP (torchmetrics with pycocotools) and road IoU on the test world, by rendering
              and by sign size; threshold chosen on validation; reliability; held-out worlds; false
              road by true class; latency; best and worst frames
```

Validation runs optimistic for every network, and the held-out runs explain why: the validation world is the easiest of the five for the signs, and it also chose each run's epoch. Validation still ranked the learning rates the way the test world does; the numbers to quote are the test numbers above.

Every cached result (label maps, trained runs, predictions) is stored with a fingerprint of the data, labels, starting weights, settings and seed that produced it, so a change to any of them recomputes the result instead of reusing a stale one.

## Repository structure

```
.
├── data/
│   ├── raw/virtual_kitti/            <- frames and annotations, downloaded by the notebook (not tracked)
│   ├── interim/                      <- label maps, frame and sign tables, predictions (not tracked)
│   └── processed/                    <- every results table the notebook reports (tracked)
├── models/runs/                      <- every training run: checkpoint, log, pulls, settings (not tracked)
├── notebooks/
│   └── driving_scene_multitask.ipynb
├── src/driving_scenes/
│   ├── config.py                     <- every constant, path, threshold and hyperparameter
│   ├── cache.py                      <- fingerprints that key every cached result
│   ├── download.py                   <- resumable streaming download of the RGB frames and labels
│   ├── dataset.py                    <- annotation decoding, label maps, audit, partitions, the frame dataset
│   ├── exploration.py                <- summaries, sign shapes, repetition, rendering agreement,
│   │                                    anchor coverage, mask ceilings
│   ├── plots.py                      <- every figure, returns a Figure
│   └── modeling/
│       ├── model.py                  <- the two-headed network, the road loss, what each stage trains
│       ├── metrics.py                <- road counts and IoU, COCO average precision
│       ├── train.py                  <- Lightning training, task balancing, fingerprinted runs
│       ├── experiments.py            <- every training run defined once, the variant rule
│       └── evaluate.py               <- stored predictions, scores by group, threshold, calibration,
│                                        false road by class, frame scores, latency
├── tests/                            <- 125 tests, one file per module, on synthetic data
├── references/data_dictionary.md
├── reports/figures/
├── pyproject.toml
└── uv.lock
```

## How to run

```bash
uv sync --extra dev
uv run jupyter lab notebooks/driving_scene_multitask.ipynb    # kernel: "Python 3 (ipykernel)"
```

Any other Python 3.10 or newer environment works too, such as a conda environment used across the course: run `pip install -e .` inside it, then pick its kernel. The device is detected with `torch.accelerator`, so the same code uses a CUDA GPU, an Apple GPU or the CPU.

Or end to end from the command line:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/driving_scene_multitask.ipynb
```

A first run downloads about 15 GB (an hour or more, depending on the connection), decodes and indexes every frame (about a minute and a half with 15 processes), then trains 21 runs: about 2 minutes for each stage one and 5 to 10 minutes for each stage two on an RTX 3090, 136 minutes in all, plus about 15 minutes of predictions. Every result is cached, so later runs repeat the whole analysis in about two minutes. The machine must stay awake while it trains.

Checks:

```bash
uv run pytest -q              # 125 tests
uv run ruff check .           # package, tests and notebook, line length included
uv run ruff format --check .
```

## What to improve next

- Choose settings and thresholds on a world that looks like deployment, for example by rotating which world validates, since the easy validation world overstated every score.
- Show the network town scenes with sidewalks, squares, shop signs and billboards (real KITTI or Cityscapes frames, or a driving simulator), so pavement and advertising are no longer unseen.
- Fit the backbone less to the training worlds, with a lower learning rate for the backbone than for the heads or fewer fine-tuned stages, and judge it on held-out worlds, since the frozen backbone's road transferred best.
- Give small signs more pixels, by feeding the frames at twice their size or adding a finer pyramid level with smaller anchors.
- Calibrate the detection confidence with temperature scaling before any decision reads it as a probability.
- Test the balancing, the anchors and the single-task baselines again at the carried-forward learning rate, and try uncertainty weighting of the two losses (Kendall et al. 2018).
