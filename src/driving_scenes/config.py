"""
Every constant the project uses.

A reviewer asking "where does this number come from" should find the answer
here. Nothing downstream hardcodes a path, a threshold, a sample size, a
hyperparameter or a color.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "virtual_kitti"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"

RANDOM_SEED = 42
N_JOBS = -2

# --- download ------------------------------------------------------------------

# The brief links the Dataset Ninja copy of Virtual KITTI 1.3.1, one tar in
# Supervisely format. It holds three renderings of every frame (RGB, depth and
# optical flow), each with its own copy of the same annotation file.
ARCHIVE_URL = (
    "https://assets.supervisely.com/remote/"
    "eyJsaW5rIjogInMzOi8vc3VwZXJ2aXNlbHktZGF0YXNldHMvMzM5MF9WaXJ0dWFsIEtJVFRJL3ZpcnR1YWwta2l0"
    "dGktRGF0YXNldE5pbmphLnRhciIsICJzaWciOiAiNGs4bklHZzhhdm5VVWxCMXJKL3BnOCsxbjFQVHR4UEJCY282"
    "ekwwdjdwaz0ifQ==?response-content-disposition=attachment%3B%20filename%3D%22virtual-kitti-"
    "DatasetNinja.tar%22"
)
# Only the RGB rendering is used, as the brief asks.
IMAGE_KIND = "rgb"
ANNOTATION_FOLDER = "ann"
IMAGE_FOLDER = "img"
META_NAME = "meta.json"
# 2,126 frames in each of ten renderings of five worlds.
EXPECTED_FRAMES = 21_260
# The tar is read in 4 MiB blocks, a multiple of the 512-byte tar block.
STREAM_BLOCK = 4 * 1024 * 1024
CONNECT_TIMEOUT = 15
READ_TIMEOUT = 120
# A dropped connection resumes at the last member boundary. The download gives
# up only after this many attempts in a row that extract nothing new.
STALLED_ATTEMPTS = 8
RETRY_WAIT = 5.0

# --- labels --------------------------------------------------------------------

LABEL_DIR = INTERIM_DIR / "labels"
FRAME_TABLE = INTERIM_DIR / "frames.csv"
SIGN_TABLE = INTERIM_DIR / "signs.csv"
# Every pixel of a frame belongs to one of these classes, in the order of the
# archive's meta file. A label map stores the position in this tuple.
CLASSES = (
    "building",
    "car",
    "guard rail",
    "misc",
    "pole",
    "road",
    "sky",
    "terrain",
    "traffic light",
    "traffic sign",
    "tree",
    "truck",
    "van",
    "vegetation",
)
ROAD_CLASS = "road"
SIGN_CLASS = "traffic sign"
# Pixels the dataset copy leaves without a label are marked with this value and
# count neither for nor against a prediction.
IGNORE_LABEL = 255
# The ten renderings of every world: the clone of the real KITTI video, four
# camera rotations and five changes of weather or time of day.
VARIATIONS = (
    "clone",
    "15-deg-left",
    "15-deg-right",
    "30-deg-left",
    "30-deg-right",
    "fog",
    "morning",
    "overcast",
    "rain",
    "sunset",
)
FRAME_WIDTH = 1242
FRAME_HEIGHT = 375
# The two things the brief asks the network to find.
TASKS = ("signs", "road")
SIGN_CLASSES = ("background", SIGN_CLASS)

# --- model ---------------------------------------------------------------------

# Every frame is 1242 x 375. Setting the detector's resize to exactly that size
# feeds the frames at their own resolution.
MIN_SIZE = FRAME_HEIGHT
MAX_SIZE = FRAME_WIDTH
# ResNet stages from the output backwards, the order in which they are unfrozen.
BACKBONE_STAGES = ("layer4", "layer3", "layer2", "layer1", "conv1")
# The pyramid levels the road head reads, at strides 4, 8, 16 and 32.
PYRAMID_LEVELS = ("0", "1", "2", "3")
ROAD_HEAD_CHANNELS = 128
ROAD_HEAD_GROUPS = 32
# The anchor boxes the proposal network starts from: sizes per pyramid level
# (the fifth level is a pooled copy of the fourth) and the height-to-width
# ratios tried at every position. The COCO defaults suit everyday objects; the
# signs, drawn with their posts, are smaller and far taller, so the sizes start
# one octave lower and move in half-octave steps, and two taller ratios are
# added. Section 6 of the notebook compares the candidates.
COCO_ANCHOR_SIZES = ((32,), (64,), (128,), (256,), (512,))
COCO_ANCHOR_RATIOS = (0.5, 1.0, 2.0)
ANCHOR_SIZES = ((16, 23), (32, 45), (64, 91), (128, 181), (256, 362))
ANCHOR_RATIOS = (0.5, 1.0, 2.0, 4.0, 8.0)
# The two sets a network can be built with.
ANCHOR_SETS = {
    "adapted": (ANCHOR_SIZES, ANCHOR_RATIOS),
    "coco": (COCO_ANCHOR_SIZES, COCO_ANCHOR_RATIOS),
}
ANCHOR_CANDIDATES = {
    "COCO defaults": (COCO_ANCHOR_SIZES, COCO_ANCHOR_RATIOS),
    "one octave smaller": (((16,), (32,), (64,), (128,), (256,)), COCO_ANCHOR_RATIOS),
    "one octave smaller, taller ratios": (((16,), (32,), (64,), (128,), (256,)), ANCHOR_RATIOS),
    "half-octave steps, taller ratios": (ANCHOR_SIZES, ANCHOR_RATIOS),
}
# Best-anchor IoU levels reported: below 0.3 an anchor counts as background for
# the proposal network, from 0.7 as a positive, and the box head's own
# threshold for a positive proposal is 0.5.
ANCHOR_IOU_LEVELS = (0.3, 0.5, 0.7)
# Detections below this score are dropped before evaluation, the usual floor
# for average precision.
SCORE_FLOOR = 0.05
DETECTIONS_PER_IMAGE = 100

# --- partitions ----------------------------------------------------------------

# Five worlds cannot be cut into exact shares. This is the assignment closest
# to the usual 60/20/20 by frames that keeps three worlds in training, with the
# larger of the two held-out worlds as the test set.
PARTITIONS = {"train": ("0002", "0006", "0020"), "validation": ("0018",), "test": ("0001",)}

# --- exploration ---------------------------------------------------------------

# COCO's size classes by box area: small below 32 x 32 pixels, large from 96 x 96.
COCO_SIZE_EDGES = (32**2, 96**2)
# A box this close to the frame's edge is counted as cut off by it.
BORDER_MARGIN = 1
# A sign region this many times taller than wide is counted as very elongated.
TALL_ASPECT = 10
# Sign regions separated by at most this many pixels are counted as neighbors,
# the trace of one sign split by something thin in front of it.
NEIGHBOR_GAP = 3
# The stride of each pyramid level the proposal network places anchors on.
ANCHOR_STRIDES = (4, 8, 16, 32, 64)
# Mask R-CNN's mask head draws a 28 x 28 mask per object; the road head predicts
# on a grid four times coarser than the frame.
MASK_HEAD_SIZE = 28
ROAD_HEAD_STRIDE = 4

# --- training ------------------------------------------------------------------

RUN_DIR = MODEL_DIR / "runs"
BATCH_SIZE = 4
# Two loader processes decode frames far faster than the GPU trains on them;
# each extra one would hold another copy of PyTorch in memory.
LOADER_WORKERS = 2
# Training frames are mirrored left to right at this rate, the one augmentation
# of the Faster R-CNN and Mask R-CNN papers.
FLIP_RATE = 0.5
# Stochastic gradient descent with momentum, as in those papers.
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
# The learning rate climbs linearly from this fraction of its peak over the
# first steps, then follows a cosine down to zero.
WARMUP_START = 0.001
WARMUP_STEPS = 100
# Stage one trains the two heads on the frozen backbone.
HEAD_EPOCHS = 2
HEAD_LR = 0.01
# Stage two also trains the pyramid and the last three ResNet stages.
FINETUNE_EPOCHS = 6
FINETUNE_LR = 0.005
FINETUNE_STAGES = 3
# How the two tasks share the backbone: "sum" adds their losses as they are,
# "equal" rescales the gradient each sends into the shared features so that
# both arrive with the same norm.
BALANCE_MODES = ("sum", "equal")
BALANCE = "equal"
# Each task's pull on the shared features is measured every this many steps and
# smoothed with this momentum.
PULL_EVERY = 10
PULL_MOMENTUM = 0.9

# The recipe is repeated with these seeds, which change the starting heads, the
# order of the frames and the proposals sampled for the box head.
SEEDS = (42, 43, 44)
# Stage-two runs that differ from the recipe in exactly one setting.
VARIANTS = {
    "sum": {"balance": "sum"},
    "signs_only": {"tasks": ("signs",)},
    "road_only": {"tasks": ("road",)},
    "frozen": {"stages": 0},
    "lr_0.0025": {"lr": 0.0025},
    "lr_0.01": {"lr": 0.01},
    # Added once 0.01 won at the top of the first bracket, so the winner is bracketed.
    "lr_0.02": {"lr": 0.02},
}
# The recipe retrained with other worlds held out, so every world except the
# validation world is scored by a network that never saw it. Validation stays
# the same world, so the epoch kept is chosen the same way.
HELD_OUT = {
    "test_0002_0006": {"train": ("0001", "0020"), "test": ("0002", "0006")},
    "test_0020": {"train": ("0001", "0002", "0006"), "test": ("0020",)},
}

# --- evaluation ----------------------------------------------------------------

PREDICTION_DIR = INTERIM_DIR / "predictions"
# A pixel is called road when its probability reaches this level.
ROAD_THRESHOLD = 0.5
# Confidence thresholds tried when choosing, on validation, the one at which a
# detection is shown as a sign.
SCORE_THRESHOLDS = tuple(round(0.05 * step, 2) for step in range(1, 20))
# Latency is the median over this many frames, after this many untimed ones.
TIMING_FRAMES = 110
TIMING_WARMUP = 10
# A detection counts as a hit when it overlaps a sign by at least this IoU.
MATCH_IOU = 0.5
# Confidence bins of equal width for the reliability of the detection scores.
CALIBRATION_BINS = 10

# --- exploration samples -------------------------------------------------------

# Frame steps compared with later steps of the same video, and the rendering used.
REPETITION_LAGS = (1, 2, 5, 10, 25, 50)
REPETITION_VARIATION = "clone"
# Moments per world at which the ten renderings are compared.
RENDERING_STEPS = 40
# Training frames squeezed through the two mask bottlenecks.
CEILING_FRAMES = 600
# A sign box overlapping a box of another frame by this IoU counts as the same sign.
SAME_SIGN_IOU = 0.5

# --- notebook ------------------------------------------------------------------

# Stage-two runs that could replace the recipe, and the baselines that cannot.
CANDIDATE_VARIANTS = ("sum", "lr_0.0025", "lr_0.01", "lr_0.02")
BASELINE_VARIANTS = ("signs_only", "road_only", "frozen")
COCO_ANCHOR_PREFIX = "coco_anchors_"
ROAD_CALL_DIR = INTERIM_DIR / "road_calls"
# Frames shown at each end of a ranking, and the digits tables are shown with.
EXTREME_COUNT = 3
# Frames shown at one end of a ranking lie this many steps apart, so they show different moments.
EXTREME_STEP_GAP = 25
# The share of test frames, lowest combined score first, read as the worst frames.
WORST_SHARE = 0.05
DISPLAY_PRECISION = 3

# --- figures -------------------------------------------------------------------

FIGURE_DPI = 110
PALETTE = {
    "road": "#6a3d9a",
    "signs": "#1b9e77",
    "multi": "#1f78b4",
    "single": "#ff7f00",
    "frozen": "#999999",
    "hit": "#1b9e77",
    "miss": "#ff7f00",
    "false": "#e7298a",
    "truth": "#ffd92f",
}
# Overlay colors for road pixels: called road and road, called road but not,
# road that was missed.
ROAD_COLORS = {"hit": (106, 61, 154), "false": (231, 41, 138), "miss": (255, 127, 0)}
ROAD_ALPHA = 0.45
