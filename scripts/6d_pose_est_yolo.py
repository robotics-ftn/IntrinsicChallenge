#!/usr/bin/env python3
#HTTPS_PROXY="http://ftn.proxy:8080"
#HTTP_PROXY="http://ftn.proxy:8080"
#HAPPYPOSE_DATA_DIR=$HOME/happypose_data
"""Object pose estimation (YOLO + MegaPose) and compliant spiral insertion.

Changes vs the previous version are marked  # CHANGED:  and  # NEW: .
Summary:
  1. Mesh origin        - use the centred mesh; POINT_OFFSET_LOCAL updated to match.
  2. Bbox-robust search - seed grid over depth x lateral x yaw, scored by MegaPose,
                          with NO dependence on the YOLO box geometry.
  3. Self-calibrating   - the box correction is learned from bbox-free poses and
     box correction       applied to later anchors. No manual calibration step.
  4. Yaw averaging      - yaw is now averaged over the stability window (it used to
                          be taken from a single frame) and gated on its own spread.
  5. Diagnostics        - per-axis tilt is printed so the correct `up` axis is visible.

v3 (marked  # NEW (v3):  and  # CHANGED (v3): ):
  6. Known height       - the card stands in a fixture, so the base-frame height of its
                          mesh origin is fixed. Every pose is slid along its own camera
                          ray onto that height (image position kept). This removes the
                          RGB-only depth error, ~73 % of which lands in x/y from the
                          45 deg fine view.
  7. Projected refine   - MegaPose refiner steps alternate with that projection, so the
                          refiner can still move x/y/yaw but cannot trade depth for tilt.
  8. Real stability     - for the card every frame is a fresh anchor and only anchors
                          count, so "stable" now means independent estimates agree. (It
                          used to mean the tracker had copied one anchor 10 times.)
  9. --measure-z        - one-off triangulation of the card height from two viewpoints.
"""

import os
import threading
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("MEGAPOSE_DATA_DIR", os.path.expanduser("~/happypose_data"))

# Paths are relative to the project folder (one level above scripts/), so the
# folder can be moved anywhere and the script run from any working directory.
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MESH_DIR = os.path.join(PROJECT_DIR, "meshes")      # every sub-folder here is loaded as one object
CARD_PATH = os.path.join(MESH_DIR, "card")
PLUG_PATH = os.path.join(MESH_DIR, "plug")
SC_PLUG_PATH = os.path.join(MESH_DIR, "sc_plug")
SC_PORT_PATH = os.path.join(MESH_DIR, "sc_port")
CAMERA_DATA_PATH = os.path.join(PROJECT_DIR, "config", "camera_data.json")   # K used when USE_RS_INTRINSICS = False
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")   # force plots / survey images
os.makedirs(OUTPUT_DIR, exist_ok=True)

YOLO_MODEL_PATH = os.path.join(PROJECT_DIR, "models", "yolo", "best.pt")
YOLO_IMGSZ = 1280
CARD_CLASS = "nic_card"
PLUG_CLASS = "sfp_plug"
SC_PLUG_CLASS = "sc_plug"
SC_PORT_CLASS = "sc_port"

PLUG_CONFIDENCE = 0.5
SC_PLUG_CONFIDENCE = 0.3
CARD_CONFIDENCE = 0.5
BBOX_GROW_H = 0.0  # put this next to the other config constants

ARM_IP = "10.1.108.144"
# TARGET_POSITION = (136.1, -235.6, 241.1, 180.0, -49.4, 34.8)
TARGET_POSITION = (205.7, 7.1, 454.7, 180.0, 0.0, 0.0)
MOVE_SPEED, SETTLE_TIME = 100, 0.5
FORCE_PLOT_SECONDS, FORCE_RATE_HZ = 10.0, 100
DETECT_EVERY, TRACK_REFINER_ITERATIONS, LOST_AFTER = 15, 2, 5

# CHANGED: the old gate (18 mm) was wider than the error being chased, so a pose
# wandering by 17 mm counted as "stable". Start tight and relax with time so the
# loop still always terminates.
POSE_SCORE_MIN, STABLE_WINDOW = 0.95, 5 # 10
STABLE_TOL_MM = 8.0            # CHANGED: was 18.0
STABLE_TOL_DEG = 4.0           # NEW: yaw must settle too, not just x/y
STABLE_RELAX_AFTER_S = 20.0    # NEW: after this long, tolerance ramps back up ...
STABLE_TOL_MM_MAX = 18.0       # NEW: ... to the old value, so nothing hangs
STABLE_TOL_DEG_MAX = 10.0
TRACK_SCORE_MIN = 0.5   # below this a pose is dropped and the next frame re-anchors

# card search: if no stable pose within SCAN_TIMEOUT s, pitch down by PITCH_STEP and retry
SCAN_TIMEOUT, PITCH_STEP, PITCH_MIN = 10.0, 5.0, -60.0

CAMERA_SERIAL = "950122061749"
FRAME_WIDTH, FRAME_HEIGHT, FRAME_FPS = 1920, 1080, 8
USE_DEPTH = False   # True: aligned depth + ICP (needs USB 3 alongside 1080p colour)
DEPTH_WIDTH, DEPTH_HEIGHT = 1280, 720
FRAME_TIMEOUT_MS = 5000
PROBE_TIMEOUT_MS = 3000   # per-frame timeout while testing a stream mode
USE_RS_INTRINSICS = False  # True: K from RealSense factory calib; False: K from camera_data.json
WARMUP_FRAMES, FLUSH_GRABS = 5, 2
PREVIEW_SCALE = 0.5

MESH_UNITS, DEVICE = "mm", "cuda"
MODEL_NAME_RGBD, MODEL_NAME_RGB = "megapose-1.0-RGB-multi-hypothesis-icp", "megapose-1.0-RGB"


def model_name():
    """RGB-D+ICP model if depth actually started, else RGB-only."""
    return MODEL_NAME_RGBD if _state.get("use_depth") else MODEL_NAME_RGB
BATCH_SIZE, MIN_BATCH_SIZE = 256, 16

# --- pose prior / candidate rejection ---------------------------------------
# megapose-1.0-RGB defaults to 1 coarse hypothesis -> no alternative when it is wrong.
N_POSE_HYPOTHESES = 10
SEED_YAW_STEP_DEG = 30.0          # yaw sweep step for upright seeds (fallback)
SEED_REFINER_ITERATIONS = 15

# NEW: bbox-robust search ----------------------------------------------------
# The YOLO box encloses more than the mesh and its centre offset varies with
# viewpoint, so neither its size nor its centre can be trusted. MegaPose seeds
# translation from that box (TCO_init_from_boxes_autodepth_with_R) and scores
# crops of it, which is where the systematic ~20 mm and the yaw bias come from.
# Passing `coarse_estimates=` instead skips detection AND the coarse stage, so
# these seeds are completely independent of the box. The box is then used only
# to say roughly where in the image to look.
SEARCH_ENABLED = True
SEARCH_DEPTH_SCALES = (0.80, 0.90, 1.00, 1.10, 1.25)  # multiples of the raw hypothesis depth
SEARCH_LATERAL_MM = (0.0, 12.0)        # ring radii perpendicular to the viewing ray
SEARCH_LATERAL_DIRS = 4                # points per ring
SEARCH_YAW_STEP_DEG = 30.0
SEARCH_COARSE_ITERS = 2                # cheap pass over the whole grid
SEARCH_FINE_ITERS = 15                 # full refinement of the survivors
SEARCH_FINE_TOPK = 6
SEARCH_SCORE_MIN = 0.90                # re-run the full search when below this

# NEW: self-calibrating box correction ---------------------------------------
# Learned ONLY from poses produced by the bbox-free search, then applied to the
# detection box on later anchors so even the initial hypothesis starts closer.
BBOX_AUTOFIT = False                   # CHANGED (v3): it learned from search poses and so
                                       # baked their depth bias into the box. Off.
BBOX_FIT_MIN_SCORE = 0.95
BBOX_FIT_ALPHA = 0.25                  # EMA weight on each new observation
BBOX_FIT_MIN_SAMPLES = 3               # applied only once this many are collected

# NEW (v3): known-height constraint --------------------------------------------
# MegaPose RGB-only is precise ACROSS the image (about 1 px) but weak ALONG the viewing ray
# (a few % of the range, 15-30 mm here). From the 45 deg fine view 73 % of that ray error
# lands in x/y - that is the miss. The card stands in a fixture, so the base-frame height of
# its (centred) mesh origin is fixed: every pose is slid along its own camera ray until the
# origin sits at that height. Image position is kept; the depth error is gone.
# Measure it once with  `python <this file> --measure-z`  (or by touch-off) and put it here.
CARD_ORIGIN_Z_MM = None           # base z of the centred mesh origin [mm]; None = off (v2 behaviour)
PLANE_MAX_CORRECTION = 0.25       # reject a pose the plane would move by > 25 % of its range
PROJECTED_ROUNDS, PROJECTED_ITERS = 6, 1   # refine/project alternations, refiner iters per round
STATIC_WINDOW = 5                 # independent anchors that must agree (robot is standing still)
# Viewpoints for --measure-z: >= 2, viewing directions >= 30 deg apart, card fully in view.
# yaw=None -> card yaw from the first view - 90, exactly as the main sequence does.
MEASURE_Z_VIEWS = [TARGET_POSITION, (435.0, -191.0, 237.0, 180.0, -45.0, None)]
MEASURE_Z_FRAMES = 5

# Per YOLO class. Classes not listed are not filtered.
#   up:              mesh axis pointing along robot-base +Z when the object stands
#                    (OBJ is Y-up -> (0,1,0); verify with the printed "tilt" on a good pose)
#   tilt_max_deg:    max angle between that axis and base +Z
#   yaw_nominal_deg: expected yaw (same definition as the printed "yaw", i.e. heading of
#                    the mesh X axis in base XY). None = any yaw. Set it to kill 180° flips.
#   yaw_tol_deg:     allowed |yaw - yaw_nominal|
#   z_range_mm:      (zmin, zmax) of the object origin in base frame, or None
#   snap:            output the exactly-upright projection of the pose (keeps x, y, z, yaw).
#                    Applied to the OUTPUT only; tracking always continues from the raw pose.
#   seed_always:     on every anchor frame also refine upright yaw-sweep seeds and keep the
#                    best-scoring candidate (object is known to stand upright)
#   plane_z_mm:      NEW (v3) base z of the mesh origin standing in its fixture, or None.
#                    Turns on the known-height constraint and anchor-only stability.
POSE_PRIORS = {
    # CHANGED (v3): seed_always off. The 300-seed grid took the max of a score that cannot
    # rank 20 mm differences, so it mainly produced confident wrong poses. It is now only a
    # fallback when the normal path scores below SEARCH_SCORE_MIN. plane_z_mm added.
    CARD_CLASS: dict(up=(0.0, 1.0, 0.0), tilt_max_deg=15.0,
                     yaw_nominal_deg=None, yaw_tol_deg=40.0,
                     z_range_mm=None, snap=True, seed_always=False,
                     plane_z_mm=CARD_ORIGIN_Z_MM),
    # NEW (v4): SC plug. Seen from above, upright and upside-down give the same outline
    # (holder is mirror-symmetric, tabs on both sides), so MegaPose often returns it flipped.
    #   flip_fix:      turn an upside-down candidate right side up (180 deg about mesh X,
    #                  through the mesh bbox centre) instead of rejecting it.
    #   anchor_always: every frame is a fresh MegaPose estimate and only those count towards
    #                  stability (STATIC_WINDOW), so one bad anchor cannot be copied 10 times.
    #   plane_z_mm:    set SC_PLUG_ORIGIN_Z_MM once measured to turn on the known-height mode.
    SC_PLUG_CLASS: dict(up=(0.0, 1.0, 0.0), tilt_max_deg=15.0,
                        yaw_nominal_deg=None, yaw_tol_deg=40.0,
                        z_range_mm=None, snap=True, seed_always=False,
                        flip_fix=True, anchor_always=True,
                        plane_z_mm=None,    # NEW (v4): SC_PLUG_ORIGIN_Z_MM when known
                        # NEW (v5) speed - SC plug only, every other class keeps the globals:
                        #   search_score_min: run the expensive 300-seed search only when the
                        #                     best pose scores below this (global: 0.90)
                        #   n_hypotheses:     MegaPose hypotheses per anchor (global: 10)
                        #   keep_on_low:      a frame below POSE_SCORE_MIN is skipped instead
                        #                     of wiping the frames already collected
                        #   timing:           print per-stage times in the status line
                        search_score_min=0.80, n_hypotheses=5,
                        keep_on_low=True, timing=True,
                        # NEW (v6) speed:
                        #   fast_yaw_step: after the first good frame, skip MegaPose's coarse
                        #                  stage (~576 rendered orientations). Instead refine
                        #                  upright seeds every fast_yaw_step deg at the last good
                        #                  position; yaw is re-found from scratch every frame.
                        #                  None = off (full pipeline every frame, as in v5).
                        #   fast_iters:    refiner iterations for those seeds
                        #   window:        independent frames that must agree (global: 5)
                        fast_yaw_step=30.0, fast_iters=4, window=3),
}

Z_SPIRAL_PRESS, Z_DROP_THRESHOLD, Z_RETREAT = 325.0, 337.0, 450.0
SPIRAL_R_START, SPIRAL_R_END, SPIRAL_STEPS = 1.0, 8.0, 200
SC_Z_SPIRAL_PRESS, SC_Z_DROP_THRESHOLD = 105.0, 134.0   # touch at 160, threshold 2 mm below, command 11 mm below that
SC_Z_RETREAT, SC_GRIPPER_OPEN = 220.0, 300
import atexit
import inspect
import sys
import json
import math
import time
import warnings
from collections import deque
from pathlib import Path

warnings.filterwarnings("ignore")

import cv2
import numpy as np
import pandas as pd
import pyrealsense2 as rs
import torch
from ultralytics import YOLO
from xarm.wrapper import XArmAPI

from happypose.pose_estimators.megapose.inference.types import ObservationTensor
from happypose.toolbox.datasets.object_dataset import RigidObject, RigidObjectDataset
from happypose.toolbox.datasets.scene_dataset import CameraData, ObjectData
from happypose.toolbox.lib3d.transform import Transform
from happypose.toolbox.utils.tensor_collection import PandasTensorCollection

try:
    from happypose.toolbox.inference.utils import make_detections_from_object_data
except ImportError:
    from happypose.pose_estimators.megapose.inference.utils import make_detections_from_object_data
try:
    from happypose.toolbox.utils.load_model import NAMED_MODELS, load_named_model
except ImportError:
    from happypose.pose_estimators.megapose.inference.utils import NAMED_MODELS, load_named_model
try:
    from happypose.toolbox.renderer import Panda3dLightData
except ImportError:
    from happypose.toolbox.renderer.types import Panda3dLightData
from happypose.toolbox.renderer.panda3d_scene_renderer import Panda3dSceneRenderer
from happypose.toolbox.utils.conversion import convert_scene_observation_to_panda3d

OOM_ERRORS = tuple({getattr(torch, "OutOfMemoryError", RuntimeError), torch.cuda.OutOfMemoryError})
MESH_SUFFIXES = {".obj", ".ply", ".gltf", ".glb"}

T_cam2flange = np.array([
    [-0.02744011, -0.99921564,  0.02855071,  0.06477879],
    [ 0.99961012, -0.02757605, -0.00437855, -0.03413778],
    [ 0.00516243,  0.02841943,  0.99958276,  0.084606  ],
    [ 0.0,         0.0,         0.0,         1.0       ],
])

T_cam_10_2flange = np.array([
    [-0.7004919,   0.71356182, -0.01185847,  0.06680221],
    [-0.71365322, -0.70031387,  0.01611101, -0.01175298],
    [ 0.00319155,  0.01974847,  0.99979989,  0.10523659],
    [ 0.0,         0.0,         0.0,         1.0       ],
])


# CHANGED: for the CENTRED card mesh (my_object_centered.obj).
# The mesh origin moved by its bbox centre (-12.212, 66.839, 16.3525) mm, so the
# target point in the object frame moves by the same amount in the opposite sense:
#   new = old - c = (-0.0081, 0.0918, 0.0034) - (-0.012212, 0.066839, 0.016352)
# POINT_OFFSET_LOCAL = np.array([
#     [1, 0, 0,  0.004112],
#     [0, 1, 0,  0.024961],
#     [0, 0, 1, -0.012952],
#     [0, 0, 0,  1.0],
# ])

# Original values, for the UN-centred mesh. Keep these if you put the old mesh back.
POINT_OFFSET_LOCAL = np.array([
    [1, 0, 0, -0.0081 + 0.0004],
    [0, 1, 0, 0.0918],
    [0, 0, 1, 0.0034 + 0.001],
    [0, 0, 0, 1.0],
])

arm = XArmAPI(ARM_IP)
_bsz = BATCH_SIZE
_state = {}


def _supported_fps(dev):
    """{(stream, w, h): set(fps)} for bgr8 colour and z16 depth profiles of the device."""
    out = {}
    for s in dev.query_sensors():
        for p in s.get_stream_profiles():
            st, fmt = p.stream_type(), p.format()
            if (st, fmt) not in ((rs.stream.color, rs.format.bgr8), (rs.stream.depth, rs.format.z16)):
                continue
            v = p.as_video_stream_profile()
            out.setdefault((st, v.width(), v.height()), set()).add(p.fps())
    return out


def start_camera():
    """Start colour (+ aligned depth if USE_DEPTH) with the first mode combination the
    device actually accepts. Returns (pipeline, profile, depth_enabled)."""
    if not USE_DEPTH:                     # original colour-only start
        pipe, cfg = rs.pipeline(), rs.config()
        cfg.enable_device(CAMERA_SERIAL)
        cfg.enable_stream(rs.stream.color, FRAME_WIDTH, FRAME_HEIGHT, rs.format.bgr8, FRAME_FPS)
        try:
            profile = pipe.start(cfg)
        except RuntimeError as e:
            raise RuntimeError(f"Could not start RealSense {CAMERA_SERIAL}: {e}")
        for _ in range(WARMUP_FRAMES):
            pipe.wait_for_frames(FRAME_TIMEOUT_MS)
        return pipe, profile, False

    devs = {d.get_info(rs.camera_info.serial_number): d for d in rs.context().query_devices()}
    if CAMERA_SERIAL not in devs:
        raise RuntimeError(f"RealSense {CAMERA_SERIAL} not found (found: {list(devs)})")
    dev = devs[CAMERA_SERIAL]
    usb = dev.get_info(rs.camera_info.usb_type_descriptor) \
        if dev.supports(rs.camera_info.usb_type_descriptor) else "?"
    fw = dev.get_info(rs.camera_info.firmware_version)
    sup = _supported_fps(dev)
    color_fps = sup.get((rs.stream.color, FRAME_WIDTH, FRAME_HEIGHT), set())
    if not color_fps:
        raise RuntimeError(f"{FRAME_WIDTH}x{FRAME_HEIGHT} bgr8 colour not offered "
                           f"(USB {usb}, FW {fw}). Colour modes: "
                           f"{sorted(k[1:] for k in sup if k[0] == rs.stream.color)}")
    usb2 = usb.startswith("2")
    # USB 2: lowest frame rates / smallest depth first (bandwidth); USB 3: closest to FRAME_FPS
    by_pref = (lambda fps_set: sorted(fps_set)) if usb2 else \
        (lambda fps_set: sorted(fps_set, key=lambda f: (abs(f - FRAME_FPS), f)))
    depth_res = ((640, 480), (480, 270), (424, 240)) if usb2 else \
        ((DEPTH_WIDTH, DEPTH_HEIGHT), (848, 480), (640, 480))
    if usb2:
        print(f"WARNING: RealSense on USB {usb} - use a USB 3 port/cable for 1080p + depth")

    candidates = []                                   # (color_fps, depth (w, h, fps) | None)
    if USE_DEPTH:
        for cf in by_pref(color_fps):
            for dw, dh in depth_res:
                dfps = sup.get((rs.stream.depth, dw, dh), set())
                for df in ([cf] if cf in dfps else []) + by_pref(dfps - {cf}):
                    candidates.append((cf, (dw, dh, df)))
    candidates += [(cf, None) for cf in sorted(color_fps, key=lambda f: (abs(f - FRAME_FPS), f))]

    errors = []
    for cf, dmode in candidates:
        pipe, cfg = rs.pipeline(), rs.config()
        cfg.enable_device(CAMERA_SERIAL)
        cfg.enable_stream(rs.stream.color, FRAME_WIDTH, FRAME_HEIGHT, rs.format.bgr8, cf)
        if dmode:
            cfg.enable_stream(rs.stream.depth, dmode[0], dmode[1], rs.format.z16, dmode[2])
        try:
            profile = pipe.start(cfg)
        except RuntimeError as e:
            errors.append(f"colour@{cf} depth={dmode}: {e}")
            continue
        # starting is not enough: on USB 2 the streams can start but never deliver
        try:
            for _ in range(WARMUP_FRAMES):
                fs = pipe.wait_for_frames(PROBE_TIMEOUT_MS)
            if not fs.get_color_frame() or (dmode and not fs.get_depth_frame()):
                raise RuntimeError("incomplete frameset")
        except RuntimeError as e:
            pipe.stop()
            errors.append(f"colour@{cf} depth={dmode}: {e}")
            print(f"  colour@{cf} depth={dmode}: no frames, trying next mode")
            continue
        print(f"RealSense {CAMERA_SERIAL} USB {usb} FW {fw}: colour "
              f"{FRAME_WIDTH}x{FRAME_HEIGHT}@{cf}"
              + (f", depth {dmode[0]}x{dmode[1]}@{dmode[2]}" if dmode else ", NO depth"))
        if USE_DEPTH and not dmode:
            print("WARNING: depth could not be started -> RGB-only (check USB 3 cable/port)")
        return pipe, profile, dmode is not None
    raise RuntimeError(f"Could not start RealSense {CAMERA_SERIAL} (USB {usb}):\n  "
                       + "\n  ".join(errors))


def init():
    arm.motion_enable(True)
    arm.clean_error()
    arm.set_mode(0)
    arm.set_state(0)
    arm.set_gripper_enable(True)

    pipe, profile, use_depth = start_camera()
    atexit.register(pipe.stop)
    _state["use_depth"] = use_depth
    if use_depth:
        _state.update(align=rs.align(rs.stream.color),
                      depth_scale=profile.get_device().first_depth_sensor().get_depth_scale())
    if USE_RS_INTRINSICS:
        i = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        K = [[i.fx, 0, i.ppx], [0, i.fy, i.ppy], [0, 0, 1]]
    else:
        K = json.loads(Path(CAMERA_DATA_PATH).read_text())["K"]
    _state.update(
        pipe=pipe, K=np.array(K, dtype=np.float32).reshape(3, 3), models={},
        yolo=YOLO(YOLO_MODEL_PATH), bbox_fit={},
    )


def read_frame():
    """Latest (BGR frame, depth in metres aligned to colour | None), or (None, None)."""
    pipe = _state["pipe"]
    for _ in range(FLUSH_GRABS):
        pipe.poll_for_frames()
    try:
        frames = pipe.wait_for_frames(FRAME_TIMEOUT_MS)
    except RuntimeError:
        return None, None
    if _state["use_depth"]:
        frames = _state["align"].process(frames)
    color = frames.get_color_frame()
    if not color:
        return None, None
    bgr = np.asanyarray(color.get_data()).copy()
    if not _state["use_depth"]:
        return bgr, None
    d = frames.get_depth_frame()
    if not d:
        return None, None
    depth = np.asanyarray(d.get_data()).astype(np.float32) * _state["depth_scale"]
    return bgr, depth


def get_T_flange2base():
    _, (x, y, z, roll, pitch, yaw) = arm.get_position()
    r, p, yw = np.radians([roll, pitch, yaw])
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(yw), -np.sin(yw), 0], [np.sin(yw), np.cos(yw), 0], [0, 0, 1]])
    T = np.eye(4)
    T[:3, :3] = Rz @ Ry @ Rx
    T[:3, 3] = np.array([x, y, z]) / 1000.0
    return T


def to_device(x):
    try:
        return x.to(DEVICE)
    except (TypeError, AttributeError, NotImplementedError):
        return x.cuda() if DEVICE.startswith("cuda") else x


def detect(frame, cls_name, confidence):
    """Largest YOLO box of class cls_name in pixel xyxy, or (None, None)."""
    res = _state["yolo"].predict(frame, imgsz=YOLO_IMGSZ, conf=confidence,
                                 device=DEVICE, verbose=False)[0]
    names = res.names
    keep = [i for i, c in enumerate(res.boxes.cls.tolist()) if names[int(c)] == cls_name]
    if not keep:
        return None, None
    boxes = res.boxes.xyxy[keep]
    best = int(torch.argmax((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])))
    box = boxes[best].cpu().numpy().astype(np.float32)
    x0, y0, x1, y1 = box
    dh = (y1 - y0) * BBOX_GROW_H / 2          # 5% on top, 5% on bottom
    h_img, w_img = frame.shape[:2]
    box = np.array([x0, max(0.0, y0 - dh), x1, min(h_img - 1.0, y1 + dh)], dtype=np.float32)
    return box, float(res.boxes.conf[keep][best])


# NEW ------------------------------------------------------------- box autofit
def silhouette_bbox(render):
    """Pixel xyxy of the rendered object silhouette, or None if nothing rendered."""
    mask = render.any(axis=2)
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def _box_cwh(b):
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0, max(b[2] - b[0], 1.0), max(b[3] - b[1], 1.0)


def bbox_fit_update(label, yolo_box, sil_box):
    """Learn how the YOLO box relates to the true silhouette (EMA)."""
    if not BBOX_AUTOFIT or yolo_box is None or sil_box is None:
        return
    cx, cy, w, h = _box_cwh(yolo_box)
    sx, sy, sw, sh = _box_cwh(sil_box)
    obs = np.array([sw / w, sh / h, (sx - cx) / w, (sy - cy) / h], dtype=np.float64)
    if not np.all(np.isfinite(obs)) or not (0.2 < obs[0] < 3.0 and 0.2 < obs[1] < 3.0):
        return
    fit = _state["bbox_fit"].get(label)
    if fit is None:
        _state["bbox_fit"][label] = [obs, 1]
    else:
        fit[0] = (1 - BBOX_FIT_ALPHA) * fit[0] + BBOX_FIT_ALPHA * obs
        fit[1] += 1


def bbox_fit_apply(label, yolo_box):
    """Map a YOLO box onto the expected silhouette box using what has been learned."""
    fit = _state["bbox_fit"].get(label)
    if not BBOX_AUTOFIT or yolo_box is None or fit is None or fit[1] < BBOX_FIT_MIN_SAMPLES:
        return yolo_box, False
    fx, fy, dx, dy = fit[0]
    cx, cy, w, h = _box_cwh(yolo_box)
    cx, cy, w, h = cx + dx * w, cy + dy * h, w * fx, h * fy
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32), True


def make_coarse(label, poses):
    """Coarse-estimate collection from 4x4 Tco poses (N,4,4)."""
    poses = np.asarray(poses, dtype=np.float32).reshape(-1, 4, 4)
    infos = pd.DataFrame(dict(label=[label] * len(poses), batch_im_id=0,
                              instance_id=np.arange(len(poses))))
    return to_device(PandasTensorCollection(infos=infos, poses=torch.from_numpy(poses)))


def run_estimator(est, obs, detections=None, prev=None, coarse=None, n_iter=None, n_hyp=None):
    """Returns (best_preds, extra_data)."""
    global _bsz
    if prev is not None and coarse is None:
        try:
            infos = prev.infos[["label"]].copy()
            infos["batch_im_id"] = 0
            infos["instance_id"] = np.arange(len(infos))
            coarse = to_device(PandasTensorCollection(infos=infos, poses=prev.poses))
        except Exception:
            coarse = prev
        n_iter = n_iter or TRACK_REFINER_ITERATIONS
    while True:
        try:
            params = dict(NAMED_MODELS[model_name()].get("inference_parameters", {}))
            params.update(bsz_images=_bsz, bsz_objects=1)
            if coarse is None:
                params["n_pose_hypotheses"] = n_hyp or N_POSE_HYPOTHESES   # CHANGED (v5)
            if n_iter is not None:
                params["n_refiner_iterations"] = n_iter
            sig = inspect.signature(est.run_inference_pipeline).parameters
            if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.values()):
                params = {k: v for k, v in params.items() if k in sig}
            with torch.no_grad():
                if coarse is not None:
                    out = est.run_inference_pipeline(obs, coarse_estimates=coarse, **params)
                else:
                    out = est.run_inference_pipeline(obs, detections=detections, **params)
            return out[0], (out[1] if len(out) > 1 else {})
        except OOM_ERRORS:
            if _bsz <= MIN_BATCH_SIZE:
                raise
            _bsz = max(MIN_BATCH_SIZE, _bsz // 2)
            torch.cuda.empty_cache()


# ---------------------------------------------------------------- pose prior
def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def _fwd_axis(up):
    """Mesh axis used for yaw: X, unless X is the up axis (then Y)."""
    return np.array([0.0, 1.0, 0.0]) if abs(_unit(up)[0]) > 0.9 else np.array([1.0, 0.0, 0.0])


def _align_up(up):
    """Rotation mapping mesh `up` axis onto +Z (Rodrigues)."""
    a, b = _unit(up), np.array([0.0, 0.0, 1.0])
    v, c = np.cross(a, b), float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


def _wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def axis_tilts(Tco, T_bc):
    """NEW: angle of each mesh axis to base +Z. The axis near 0 is the real `up`."""
    R = (T_bc @ Tco)[:3, :3]
    return {n: math.degrees(math.acos(np.clip(R[2, i], -1.0, 1.0)))
            for i, n in enumerate("XYZ")}


def prior_check(Tco, T_bc, prior):
    """(ok, tilt_deg, yaw_deg, reason) for a camera-frame pose under the prior."""
    T_bo = T_bc @ Tco
    R = T_bo[:3, :3]
    tilt = math.degrees(math.acos(np.clip((R @ _unit(prior["up"]))[2], -1.0, 1.0)))
    f = R @ _fwd_axis(prior["up"])
    yaw = math.degrees(math.atan2(f[1], f[0]))
    if Tco[2, 3] <= 0:
        return False, tilt, yaw, "behind cam"
    if tilt > prior["tilt_max_deg"]:
        return False, tilt, yaw, f"tilt {tilt:.0f}"
    if prior.get("yaw_nominal_deg") is not None and \
            abs(_wrap180(yaw - prior["yaw_nominal_deg"])) > prior["yaw_tol_deg"]:
        return False, tilt, yaw, f"yaw {yaw:.0f}"
    zr = prior.get("z_range_mm")
    if zr is not None and not (zr[0] <= T_bo[2, 3] * 1000.0 <= zr[1]):
        return False, tilt, yaw, f"z {T_bo[2, 3] * 1000:.0f}mm"
    return True, tilt, yaw, ""


def snap_upright(Tco, T_bc, prior):
    """Closest exactly-upright pose: keeps position and yaw, removes tilt."""
    T_bo = T_bc @ Tco
    up, fwd = _unit(prior["up"]), _fwd_axis(prior["up"])
    f = T_bo[:3, :3] @ fwd
    f[2] = 0.0
    if np.linalg.norm(f) < 1e-6:
        return Tco
    yaw = math.atan2(f[1], f[0])
    Ra = _align_up(up)
    f0 = Ra @ fwd
    psi = yaw - math.atan2(f0[1], f0[0])
    Rz = np.array([[math.cos(psi), -math.sin(psi), 0], [math.sin(psi), math.cos(psi), 0], [0, 0, 1]])
    T_new = T_bo.copy()
    T_new[:3, :3] = Rz @ Ra
    return np.linalg.inv(T_bc) @ T_new


def mesh_bbox_center(mesh_path):
    """Bounding-box centre of an OBJ mesh in metres (zeros for other formats)."""
    if Path(mesh_path).suffix.lower() != ".obj":
        return np.zeros(3)
    scale = {"mm": 1e-3, "cm": 1e-2, "m": 1.0}[MESH_UNITS]
    v = np.array([l.split()[1:4] for l in open(mesh_path) if l.startswith("v ")], dtype=np.float64)
    return (v.min(0) + v.max(0)) / 2.0 * scale


def upright_seeds(c_base, T_bc, prior, c_obj=np.zeros(3), yaw_step=None):
    """Upright Tco hypotheses swept over yaw, with the mesh point c_obj (object frame)
    placed at base position c_base for every yaw."""
    up, fwd = _unit(prior["up"]), _fwd_axis(prior["up"])
    Ra = _align_up(up)
    f0 = Ra @ fwd
    yaw0 = math.atan2(f0[1], f0[0])
    nom = prior.get("yaw_nominal_deg")
    step = yaw_step or SEED_YAW_STEP_DEG
    if nom is None:
        yaws = np.arange(0.0, 360.0, step)
    else:
        tol = prior["yaw_tol_deg"]
        yaws = nom + np.arange(-tol, tol + 1e-6, min(10.0, max(tol, 1.0)))
    T_cb = np.linalg.inv(T_bc)
    seeds = []
    for y in np.radians(yaws):
        psi = y - yaw0
        Rz = np.array([[math.cos(psi), -math.sin(psi), 0], [math.sin(psi), math.cos(psi), 0], [0, 0, 1]])
        T_bo = np.eye(4)
        R = Rz @ Ra
        T_bo[:3, :3], T_bo[:3, 3] = R, c_base - R @ c_obj
        seeds.append(T_cb @ T_bo)
    return np.stack(seeds)


def search_seeds(T_raw, T_bc, prior, c_obj):
    """NEW: upright seeds over a depth x lateral x yaw grid around the raw hypothesis.

    The raw hypothesis inherits the box error, so instead of trusting its position
    we keep only its viewing direction and sweep distance along that ray plus a
    lateral ring, letting MegaPose's own scorer decide. Nothing here reads the box.
    """
    p = np.asarray(T_raw[:3, 3], dtype=np.float64)
    d = float(np.linalg.norm(p))
    if d <= 1e-6 or p[2] <= 0:
        return None
    ray = p / d
    a = np.array([0.0, 0.0, 1.0])
    u = np.cross(ray, a)
    u = u / np.linalg.norm(u) if np.linalg.norm(u) > 1e-6 else np.array([1.0, 0.0, 0.0])
    v = np.cross(ray, u)
    out = []
    for s in SEARCH_DEPTH_SCALES:
        base = p * s
        for r in SEARCH_LATERAL_MM:
            if r <= 0.0:
                offsets = [np.zeros(3)]
            else:
                offsets = [(math.cos(t) * u + math.sin(t) * v) * (r / 1000.0)
                           for t in np.linspace(0, 2 * math.pi, SEARCH_LATERAL_DIRS, endpoint=False)]
            for off in offsets:
                c_base = (T_bc @ np.r_[base + off, 1.0])[:3]
                out.append(upright_seeds(c_base, T_bc, prior, c_obj, SEARCH_YAW_STEP_DEG))
    return np.concatenate(out) if out else None


def all_candidates(preds, extra):
    """(poses Nx4x4, scores N) of every scored hypothesis, falling back to top-1."""
    c = preds
    try:
        c = extra["scoring"]["preds"]
    except (KeyError, TypeError):
        pass
    poses = c.poses.detach().cpu().numpy()
    for col in ("pose_score", "pose_logit", "score"):
        if col in c.infos.columns:
            return poses, c.infos[col].to_numpy(dtype=np.float64)
    return poses, np.zeros(len(poses))


def unflip(Tco, T_bc, prior, c_obj=None):
    """NEW (v4): if the mesh `up` axis points down in the base frame, rotate the pose 180 deg
    about the mesh X axis through the mesh bbox centre c_obj. The outline seen from above is
    unchanged, only up/down is corrected. No-op unless the prior has flip_fix=True."""
    if prior is None or not prior.get("flip_fix"):
        return Tco
    if ((T_bc @ Tco)[:3, :3] @ _unit(prior["up"]))[2] >= 0.0:
        return Tco
    c = np.zeros(3) if c_obj is None else np.asarray(c_obj, dtype=np.float64)
    R = np.diag([1.0, -1.0, -1.0])
    F = np.eye(4)
    F[:3, :3], F[:3, 3] = R, c - R @ c
    return (np.asarray(Tco, dtype=np.float64) @ F).astype(np.asarray(Tco).dtype)


def select_pose(preds, extra, T_bc, prior, c_obj=None):
    """Best-scoring hypothesis satisfying the prior (raw, not snapped).
    Returns (Tco | None, score, reason_or_info, best_raw_Tco)."""
    poses, scores = all_candidates(preds, extra)
    if prior is not None and prior.get("flip_fix"):                  # NEW (v4)
        poses = np.stack([unflip(t, T_bc, prior, c_obj) for t in poses])
    order = np.argsort(-scores)
    if prior is None:
        i = order[0]
        return poses[i], float(scores[i]), "", poses[i]
    reasons = []
    for i in order:
        ok, tilt, yaw, why = prior_check(poses[i], T_bc, prior)
        if ok:
            return poses[i], float(scores[i]), f"tilt {tilt:.1f}  {len(reasons)} rej", poses[order[0]]
        reasons.append(why)
    return None, float(scores[order[0]]), "rejected: " + ",".join(reasons[:3]), poses[order[0]]


# NEW (v3) ------------------------------------------------------- known height
def ray_plane(Tco, T_bc, z0, max_corr=PLANE_MAX_CORRECTION):
    """Slide Tco along the camera ray through its origin until the origin is at base z0 [m].

    Rotation and the origin's image position are unchanged; only the distance changes.
    Returns (Tco_new, s) with s = new / old distance, or (None, s) if the ray is within
    ~15 deg of horizontal or |s - 1| > max_corr.
    """
    t = np.asarray(Tco[:3, 3], dtype=np.float64)
    n = float(np.linalg.norm(t))
    if n < 1e-6 or t[2] <= 0:
        return None, float("nan")
    vz = float(T_bc[2, :3] @ (t / n))             # vertical component of the ray (base)
    if abs(vz) < 0.25:
        return None, float("nan")
    s = (z0 - T_bc[2, 3]) / (vz * n)
    if not (1.0 - max_corr <= s <= 1.0 + max_corr):
        return None, s
    T = np.array(Tco, dtype=np.float64, copy=True)
    T[:3, 3] = t * s
    return T, s


def constrain_pose(Tco, T_bc, prior, max_corr=PLANE_MAX_CORRECTION):
    """Exactly upright (about its origin), then origin slid onto the known height."""
    return ray_plane(snap_upright(Tco, T_bc, prior), T_bc, prior["plane_z_mm"] / 1000.0, max_corr)


def plane_seeds(T_raw, T_bc, prior):
    """Upright yaw-sweep seeds with the origin ON the known height, where the raw hypothesis'
    viewing ray meets it, plus a small ring in base XY. Replaces the depth-scale grid:
    depth is no longer a free variable."""
    T, _ = ray_plane(T_raw, T_bc, prior["plane_z_mm"] / 1000.0, max_corr=0.5)
    if T is None:
        return None
    c = (T_bc @ T)[:3, 3]
    out = []
    for r in SEARCH_LATERAL_MM:
        if r <= 0.0:
            offsets = [np.zeros(3)]
        else:
            offsets = [np.array([math.cos(a), math.sin(a), 0.0]) * (r / 1000.0)
                       for a in np.linspace(0, 2 * math.pi, SEARCH_LATERAL_DIRS, endpoint=False)]
        for off in offsets:
            out.append(upright_seeds(c + off, T_bc, prior, np.zeros(3), SEARCH_YAW_STEP_DEG))
    return np.concatenate(out)


def projected_refine(est, obs, label, T0s, T_bc, prior):
    """Refine on the constraint set {upright, origin at the known height}.

    Each round runs PROJECTED_ITERS MegaPose refiner iterations, then projects every result
    back onto the set. The refiner can still slide the card across the image and turn it,
    but its depth (and the tilt it uses to compensate a wrong depth) is thrown away.
    Returns (poses Nx4x4, scores N) of the survivors - possibly empty.
    """
    T = [constrain_pose(t, T_bc, prior)[0] for t in np.asarray(T0s).reshape(-1, 4, 4)]
    T = [t for t in T if t is not None]
    scores = np.zeros(len(T))
    for _ in range(PROJECTED_ROUNDS):
        if not T:
            break
        preds, extra = run_estimator(est, obs, coarse=make_coarse(label, np.stack(T)),
                                     n_iter=PROJECTED_ITERS)
        poses, sc = all_candidates(preds, extra)
        T, keep = [], []
        for t, v in zip(poses, sc):
            tc, _ = constrain_pose(t, T_bc, prior)
            if tc is not None:
                T.append(tc)
                keep.append(v)
        scores = np.asarray(keep, dtype=np.float64)
    if not T:
        return np.zeros((0, 4, 4)), np.zeros(0)
    return np.stack(T), scores


def plane_estimate(est, obs, label, Tco, p, T_raw, T_bc, prior):
    """Best pose on the constraint set. Starts from the normal MegaPose result; the plane
    yaw sweep only runs if that scores below SEARCH_SCORE_MIN (or seed_always is set).
    Returns (Tco | None, score, info, from_search). `info` reports how far MegaPose's own
    depth was from the known height ("depth fix"); a steady value of 20-30 mm confirms
    the diagnosis."""
    cand, depth_fix = [], ""
    if Tco is not None:
        _, s0 = constrain_pose(Tco, T_bc, prior, max_corr=1.0)
        if np.isfinite(s0):
            depth_fix = f"  depth fix {(s0 - 1.0) * np.linalg.norm(Tco[:3, 3]) * 1000.0:+.0f}mm"
        T, sc = projected_refine(est, obs, label, Tco[None], T_bc, prior)
        cand += [(v, t, "plane") for t, v in zip(T, sc)]
    if SEARCH_ENABLED and T_raw[2, 3] > 0 and (
            not cand or max(c[0] for c in cand) < SEARCH_SCORE_MIN or prior.get("seed_always")):
        seeds = plane_seeds(T_raw, T_bc, prior)
        if seeds is not None:
            s_preds, s_extra = run_estimator(est, obs, coarse=make_coarse(label, seeds),
                                             n_iter=SEARCH_COARSE_ITERS)
            s_poses, s_scores = all_candidates(s_preds, s_extra)
            top = np.argsort(-s_scores)[:SEARCH_FINE_TOPK]
            T, sc = projected_refine(est, obs, label, s_poses[top], T_bc, prior)
            cand += [(v, t, "plane-search") for t, v in zip(T, sc)]
    cand = [c for c in cand if prior_check(c[1], T_bc, prior)[0]]
    if not cand:
        return None, p, "plane: nothing fits the known height (check CARD_ORIGIN_Z_MM)" + depth_fix, False
    v, T, tag = max(cand, key=lambda c: c[0])
    return T, float(v), tag + depth_fix, tag == "plane-search"


def closest_point_to_rays(rays):
    """Least-squares point closest to rays [(centre, unit_dir), ...] (all in base frame)."""
    A, b = np.zeros((3, 3)), np.zeros(3)
    for c, d in rays:
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ c
    return np.linalg.solve(A, b)


def render_poses(renderer, K, resolution, labels, T):
    cam_data = CameraData(K=np.asarray(K, dtype=np.float64), resolution=resolution)
    cam_data.TWC = Transform(np.eye(4))
    objs_data = [ObjectData(label=l, TWO=Transform(np.asarray(t, dtype=np.float64)))
                 for l, t in zip(labels, T)]
    cam, objs = convert_scene_observation_to_panda3d(cam_data, objs_data)
    lights = [Panda3dLightData(light_type="ambient", color=(1.0, 1.0, 1.0, 1.0))]
    try:
        r = renderer.render_scene(objs, [cam], lights, render_depth=False,
                                  render_binary_mask=False, render_normals=False,
                                  copy_arrays=True)[0]
    except TypeError:
        r = renderer.render_scene(objs, [cam], lights)[0]
    img = np.asarray(r.rgb)
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return img[..., :3]


def draw_view(frame, status, render=None, K=None, Tco=None, bbox=None, bbox_fit=None):
    view = frame.copy()
    if render is not None:
        mask = render.any(axis=2)
        eroded = mask.copy()
        for _ in range(2):
            p = np.pad(eroded, 1, constant_values=False)
            eroded = p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
        view[mask & ~eroded] = (0, 255, 0)
    if bbox is not None:
        x0, y0, x1, y1 = (int(round(v)) for v in bbox)
        cv2.rectangle(view, (x0, y0), (x1, y1), (255, 0, 0), 2)
    if bbox_fit is not None:                       # NEW: corrected box, for eyeballing
        x0, y0, x1, y1 = (int(round(v)) for v in bbox_fit)
        cv2.rectangle(view, (x0, y0), (x1, y1), (0, 165, 255), 2)
    if Tco is not None:
        proj = lambda q: None if q[2] <= 1e-6 else (int(round(K[0, 0] * q[0] / q[2] + K[0, 2])),
                                                     int(round(K[1, 1] * q[1] / q[2] + K[1, 2])))
        c, t = proj(Tco[:3, 3]), proj((Tco @ POINT_OFFSET_LOCAL)[:3, 3])
        if c:
            cv2.drawMarker(view, c, (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
        if t:
            cv2.drawMarker(view, t, (0, 255, 255), cv2.MARKER_TILTED_CROSS, 16, 2)
            cv2.putText(view, "target", (t[0] + 10, t[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
    view = cv2.resize(view, None, fx=PREVIEW_SCALE, fy=PREVIEW_SCALE,
                      interpolation=cv2.INTER_AREA)
    cv2.putText(view, f"{status}  Q=cancel", (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return view


def load_models(model_path):
    """(estimator, renderer, mesh centres, label) for the mesh folder of model_path, cached.
    CHANGED (v3): factored out of estimate_object_pose so --measure-z can use it."""
    model_path = Path(model_path).expanduser()
    label, root = model_path.name, model_path.parent

    if str(root) not in _state["models"]:
        objs, centers = [], {}
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            meshes = [m for m in sorted(d.iterdir()) if m.suffix.lower() in MESH_SUFFIXES]
            if meshes:
                objs.append(RigidObject(label=d.name, mesh_path=meshes[0], mesh_units=MESH_UNITS))
                centers[d.name] = mesh_bbox_center(meshes[0])
        obj_ds = RigidObjectDataset(objs)
        _state["models"][str(root)] = (load_named_model(model_name(), obj_ds).to(DEVICE),
                                       Panda3dSceneRenderer(obj_ds), centers)
    est, renderer, centers = _state["models"][str(root)]
    return est, renderer, centers, label


def estimate_object_pose(model_path, cls_name, confidence, timeout=None):
    """Returns (x_mm, y_mm, yaw_deg) of the target point in robot base frame,
    or None if no stable pose is found within `timeout` seconds."""
    if not _state:
        init()
    est, renderer, centers, label = load_models(model_path)
    K = _state["K"]
    prior = POSE_PRIORS.get(cls_name)

    # NEW: the coarse stage was trained on meshes whose origin is near the object
    # centre. An origin far outside the body biases both depth and yaw.
    c_obj = centers.get(label, np.zeros(3))
    if np.linalg.norm(c_obj) > 0.005:
        print(f"WARNING: '{label}' mesh origin is {np.linalg.norm(c_obj) * 1000:.0f} mm from its "
              f"bbox centre {np.round(c_obj * 1000, 1)} -> use the centred mesh and update "
              f"POINT_OFFSET_LOCAL by subtracting that offset.")

    # NEW (v3): known-height mode. The robot is still while it looks, so tracking adds
    # nothing: every frame is a fresh anchor and only those count towards stability.
    plane = prior is not None and prior.get("plane_z_mm") is not None
    # NEW (v4): anchor_always -> every frame is an independent anchor, as in known-height mode
    anchor_always = prior is not None and bool(prior.get("anchor_always"))
    window = (((prior.get("window") if prior else None) or STATIC_WINDOW)   # CHANGED (v6)
              if (plane or anchor_always) else STABLE_WINDOW)
    fast_c_base = None   # NEW (v6): base position of the mesh centre from the last good frame
    if plane:
        print(f"'{cls_name}': known-height mode, origin z = {prior['plane_z_mm']:.1f} mm")
    prev, frame_id, misses, det_score, bbox, bbox_c = None, 0, 0, None, None, None
    history = deque(maxlen=window)
    t0 = time.time()
    try:
        while True:
            if timeout is not None and time.time() - t0 > timeout:
                print(f"No stable pose for '{cls_name}' within {timeout:.1f}s")
                return None

            frame, depth = read_frame()
            if frame is None:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_id += 1

            anchor = (plane or anchor_always or prev is None
                      or frame_id % DETECT_EVERY == 0)                  # CHANGED (v4)
            detections = None
            if anchor:
                bbox, det_score = detect(frame, cls_name, confidence)
                if bbox is None:
                    misses += 1
                    if misses >= LOST_AFTER:
                        prev = None
                    history.clear()
                    cv2.imshow("capture", draw_view(frame, "searching..."))
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        raise RuntimeError("Capture cancelled by user")
                    continue
                misses = 0
                # NEW: feed MegaPose the corrected box once the fit is known
                bbox_c, _used_fit = bbox_fit_apply(label, bbox)
                obj = ObjectData(label=label, bbox_modal=bbox_c)
                detections = to_device(make_detections_from_object_data([obj]))

            obs = to_device(ObservationTensor.from_numpy(rgb, depth, K))
            T_bc = get_T_flange2base() @ T_cam2flange
            t_est = time.time()                                          # NEW (v5)
            fast = (anchor and fast_c_base is not None                   # NEW (v6)
                    and prior is not None and prior.get("fast_yaw_step"))
            if fast:
                seeds = upright_seeds(fast_c_base, T_bc, prior, c_obj, prior["fast_yaw_step"])
                preds, extra = run_estimator(est, obs, coarse=make_coarse(label, seeds),
                                             n_iter=prior.get("fast_iters", 4))
            elif anchor:
                preds, extra = run_estimator(est, obs, detections=detections,
                                             n_hyp=prior.get("n_hypotheses") if prior else None)
            else:
                preds, extra = run_estimator(est, obs, prev=prev)
            Tco, p, info, T_raw = select_pose(preds, extra, T_bc, prior, c_obj)

            # CHANGED: the old seeding re-used the raw hypothesis position, so it
            # inherited the box-induced depth error. The grid below sweeps depth and
            # lateral position as well as yaw, and never touches the box.
            from_search = False
            t_search = time.time()                                       # NEW (v5)
            if plane:                                   # NEW (v3)
                Tco, p, info, from_search = plane_estimate(est, obs, label, Tco, p, T_raw,
                                                           T_bc, prior)
            want_search = (not plane and SEARCH_ENABLED and prior is not None and anchor
                           and T_raw[2, 3] > 0
                           and (Tco is None
                                or p < prior.get("search_score_min", SEARCH_SCORE_MIN)  # v5
                                or prior.get("seed_always", False)))
            if want_search:
                seeds = search_seeds(T_raw, T_bc, prior, c_obj)
                if seeds is not None:
                    s_preds, s_extra = run_estimator(est, obs, coarse=make_coarse(label, seeds),
                                                     n_iter=SEARCH_COARSE_ITERS)
                    s_poses, s_scores = all_candidates(s_preds, s_extra)
                    top = np.argsort(-s_scores)[:SEARCH_FINE_TOPK]
                    f_preds, f_extra = run_estimator(
                        est, obs, coarse=make_coarse(label, s_poses[top]),
                        n_iter=SEARCH_FINE_ITERS)
                    f_T, f_p, f_info, _ = select_pose(f_preds, f_extra, T_bc, prior, c_obj)
                    if f_T is not None and (Tco is None or f_p > p):
                        Tco, p, info, from_search = f_T, f_p, f"search  {f_info}", True
            t_done = time.time()                                         # NEW (v5)
            if prior and prior.get("timing"):
                info += (f"  [{'fast' if fast else 'full'} {(t_search - t_est) * 1000:.0f}ms"
                         + (f" search {(t_done - t_search) * 1000:.0f}ms" if want_search else "")
                         + "]")
            keep_on_low = bool(prior and prior.get("keep_on_low"))       # NEW (v5)

            # low score -> do not track from it; re-anchor on the next frame
            if Tco is not None and p < TRACK_SCORE_MIN:
                info = f"low score  {info}"
                Tco = None

            if Tco is None:
                prev = None          # force re-anchor on next frame
                fast_c_base = None   # NEW (v6): next frame runs the full pipeline
                if not keep_on_low:                                      # CHANGED (v5)
                    history.clear()
                status = f"{info}  pose {p:.2f}"
                print(status)
                cv2.imshow("capture", draw_view(frame, status, None, K, None, bbox, bbox_c))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    raise RuntimeError("Capture cancelled by user")
                continue

            prev = make_coarse(label, Tco[None])   # track from the RAW accepted pose
            render = render_poses(renderer, K, rgb.shape[:2], [label], [Tco])
            # snapped (exactly upright) pose is used only for the output / target marker
            Tco_out = snap_upright(Tco, T_bc, prior) if prior and prior.get("snap") else Tco

            # NEW: learn the box correction, but ONLY from bbox-free poses, so the
            # correction cannot feed back on itself.
            if from_search and p >= BBOX_FIT_MIN_SCORE:
                bbox_fit_update(label, bbox, silhouette_bbox(render))

            # NEW (v6): remember where the mesh centre is, so the next frame can skip the
            # coarse stage; a weak frame sends the next one back through the full pipeline.
            if prior is not None and prior.get("fast_yaw_step"):
                fast_c_base = ((T_bc @ Tco)[:3, :3] @ c_obj + (T_bc @ Tco)[:3, 3]
                               if p >= POSE_SCORE_MIN else None)

            if p >= POSE_SCORE_MIN:
                T_bp = T_bc @ Tco_out
                if cls_name == "nic_card":
                    T_bp = T_bp @ POINT_OFFSET_LOCAL
                # CHANGED: yaw now goes into the history and is averaged like x/y.
                history.append((T_bp[0, 3] * 1000.0, T_bp[1, 3] * 1000.0,
                                math.degrees(math.atan2(T_bp[1, 0], T_bp[0, 0]))))
            elif not keep_on_low:                                        # CHANGED (v5)
                history.clear()

            spread = yaw_spread = yaw = None
            if len(history) == window:                  # CHANGED (v3)
                arr = np.array(history)
                pts = arr[:, :2]
                centre = np.median(pts, axis=0)         # CHANGED (v3): median, not mean
                spread = float(np.max(np.linalg.norm(pts - centre, axis=1)))
                a = np.radians(arr[:, 2])
                yaw = math.degrees(math.atan2(np.sin(a).mean(), np.cos(a).mean()))
                yaw_spread = float(np.max(np.abs(_wrap180(arr[:, 2] - yaw))))
            elif history:
                yaw = history[-1][2]

            # CHANGED: tolerance starts tight and relaxes, so precision is preferred
            # but the loop still always terminates.
            k = np.clip((time.time() - t0 - STABLE_RELAX_AFTER_S) / STABLE_RELAX_AFTER_S, 0.0, 1.0)
            tol_mm = STABLE_TOL_MM + k * (STABLE_TOL_MM_MAX - STABLE_TOL_MM)
            tol_deg = STABLE_TOL_DEG + k * (STABLE_TOL_DEG_MAX - STABLE_TOL_DEG)

            status = "ANCHOR" if anchor else "track"
            if det_score is not None:
                status += f"  yolo {det_score:.2f}"
            status += f"  pose {p:.2f}  {info}"
            if anchor:                                  # NEW: which mesh axis is really up
                t_ax = axis_tilts(Tco, T_bc)
                status += "  tilt " + "/".join(f"{n}{t_ax[n]:.0f}" for n in "XYZ")
                fit = _state["bbox_fit"].get(label)
                if fit is not None:
                    status += ("  boxfit " + "x".join(f"{v:.2f}" for v in fit[0][:2])
                               + f" n{fit[1]}")
            status += f"  [{len(history)}/{window}]"
            if spread is not None:
                status += f"  spread {spread:.2f}mm/{yaw_spread:.1f}deg (tol {tol_mm:.0f}/{tol_deg:.0f})"
            if yaw is not None:
                status += f"  yaw {yaw:.1f}"
            print(status)

            cv2.imshow("capture", draw_view(frame, status, render, K, Tco_out, bbox, bbox_c))

            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise RuntimeError("Capture cancelled by user")
            if spread is not None and spread <= tol_mm and yaw_spread <= tol_deg:
                mx, my = centre
                return float(mx), float(my), float(yaw)
    finally:
        cv2.destroyAllWindows()


# NEW (v3) ----------------------------------------------------------- --measure-z
def measure_origin_z(views=MEASURE_Z_VIEWS, model_path=CARD_PATH, cls_name=CARD_CLASS,
                     confidence=CARD_CONFIDENCE, n_frames=MEASURE_Z_FRAMES):
    """One-off: triangulate the card's mesh origin from >= 2 viewpoints.

    Only the DIRECTION from the camera to the origin is used (MegaPose gets that to about
    1 px); its distance, the weak part, is ignored. Rays from views >= 30 deg apart cross
    at the origin. Prints the value for CARD_ORIGIN_Z_MM. The result is only as good as
    the hand-eye calibration, so check that first (ChArUco test).
    """
    if not _state:
        init()
    est, _, centers, label = load_models(model_path)
    prior = POSE_PRIORS.get(cls_name)
    if prior is not None:
        prior = dict(prior, plane_z_mm=None)
    rays, yaw_first = [], None
    for vi, (x, y, z, roll, pitch, yaw) in enumerate(views):
        if yaw is None:
            if yaw_first is None:
                raise RuntimeError("a view with yaw=None needs a successful view before it")
            yaw = yaw_first - 90.0
        arm.set_position(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw,
                         speed=MOVE_SPEED, wait=True, timeout=30.0)
        time.sleep(SETTLE_TIME)
        dirs, yaws, centre, tries = [], [], None, 0
        while len(dirs) < n_frames and tries < 6 * n_frames:
            tries += 1
            frame, depth = read_frame()
            if frame is None:
                continue
            bbox, _ = detect(frame, cls_name, confidence)
            if bbox is None:
                continue
            obs = to_device(ObservationTensor.from_numpy(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), depth, _state["K"]))
            det = to_device(make_detections_from_object_data(
                [ObjectData(label=label, bbox_modal=bbox)]))
            T_bc = get_T_flange2base() @ T_cam2flange
            preds, extra = run_estimator(est, obs, detections=det)
            Tco, p, _, _ = select_pose(preds, extra, T_bc, prior,
                                       centers.get(label, np.zeros(3)))
            if Tco is None or p < POSE_SCORE_MIN:
                continue
            t = np.asarray(Tco[:3, 3], dtype=np.float64)
            dirs.append(T_bc[:3, :3] @ (t / np.linalg.norm(t)))
            centre = T_bc[:3, 3].copy()
            R = (T_bc @ Tco)[:3, :3]
            yaws.append(math.degrees(math.atan2(R[1, 0], R[0, 0])))
        if not dirs:
            print(f"view {vi}: no pose with score >= {POSE_SCORE_MIN}, skipped")
            continue
        d = np.mean(dirs, axis=0)
        d /= np.linalg.norm(d)
        rays.append((centre, d))
        if yaw_first is None:
            a = np.radians(yaws)
            yaw_first = math.degrees(math.atan2(np.sin(a).mean(), np.cos(a).mean()))
        print(f"view {vi}: {len(dirs)} poses, camera {np.round(centre * 1000, 1)} mm, "
              f"ray {np.round(d, 3)}")
    if len(rays) < 2:
        raise RuntimeError("need rays from at least 2 views")
    ang = max(math.degrees(math.acos(np.clip(abs(a[1] @ b[1]), 0.0, 1.0)))
              for i, a in enumerate(rays) for b in rays[i + 1:])
    if ang < 20.0:
        print(f"WARNING: rays only {ang:.0f} deg apart - z will be poorly conditioned")
    pt = closest_point_to_rays(rays)
    resid = [float(np.linalg.norm(np.cross(pt - c, d))) * 1000.0 for c, d in rays]
    print(f"origin in base = {np.round(pt * 1000, 1)} mm   ray miss {np.round(resid, 1)} mm   "
          f"ray angle {ang:.0f} deg")
    print(f"-> set  CARD_ORIGIN_Z_MM = {pt[2] * 1000:.1f}")
    return pt * 1000.0


def generate_spiral(cx, cy, r_start, r_end, num_steps, radial_pitch=1.0, dense=4000):
    if radial_pitch <= 0:
        raise ValueError("radial_pitch must be > 0")
    if r_start == r_end:
        raise ValueError("r_start and r_end must differ")

    b = math.copysign(radial_pitch / (2.0 * math.pi), r_end - r_start)
    thetas = [(r_end - r_start) / b * i / dense for i in range(dense + 1)]

    arc = [0.0]
    for i in range(1, len(thetas)):
        r0, r1 = r_start + b * thetas[i - 1], r_start + b * thetas[i]
        arc.append(arc[-1] + 0.5 * (math.hypot(r0, b) + math.hypot(r1, b))
                   * (thetas[i] - thetas[i - 1]))

    points, j = [], 0
    for k in range(num_steps):
        target = 0.0 if num_steps == 1 else arc[-1] * k / (num_steps - 1)
        while j < len(arc) - 1 and arc[j + 1] < target:
            j += 1
        if j >= len(arc) - 1:
            theta = thetas[-1]
        else:
            span = arc[j + 1] - arc[j]
            f = 0.0 if span == 0.0 else (target - arc[j]) / span
            theta = thetas[j] + f * (thetas[j + 1] - thetas[j])
        r = r_start + b * theta
        points.append((cx + r * math.cos(theta), cy + r * math.sin(theta)))
    return points


def generate_figure8(cx, cy, a, num_steps, loops=4, angle_deg=0.0, dense=4000):

    t_max = 2.0 * math.pi * loops
    ca, sa = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))

    def pt(t):
        return a * math.sin(t), 0.5 * a * math.sin(2.0 * t)

    ts = [t_max * i / dense for i in range(dense + 1)]
    xy = [pt(t) for t in ts]

    arc = [0.0]
    for i in range(1, len(xy)):
        arc.append(arc[-1] + math.hypot(xy[i][0] - xy[i - 1][0], xy[i][1] - xy[i - 1][1]))

    points, j = [], 0
    for k in range(num_steps):
        target = 0.0 if num_steps == 1 else arc[-1] * k / (num_steps - 1)
        while j < len(arc) - 1 and arc[j + 1] < target:
            j += 1
        if j >= len(arc) - 1:
            t = ts[-1]
        else:
            span = arc[j + 1] - arc[j]
            f = 0.0 if span == 0.0 else (target - arc[j]) / span
            t = ts[j] + f * (ts[j + 1] - ts[j])
        x, y = pt(t)
        points.append((cx + ca * x - sa * y, cy + sa * x + ca * y))
    return points


def set_admittance(K_adm, axes, frame = 1):
    arm.set_ft_sensor_enable(1)
    M = [0.05, 0.05, 0.05, 0.01, 0.01, 0.01]
    D = [5.0, 5.0, 5.0, 0.2, 0.2, 0.1]
    arm.set_ft_sensor_admittance_parameters(frame, axes, M, K_adm, D)
    arm.set_ft_sensor_mode(1)
    arm.set_state(0)


# CHANGED (force plot): arm.ft_ext_force is only a cached copy of the last report packet,
# and the SDK's default report socket updates a few times per second. Sampling that copy
# at 100 Hz gave runs of identical values -> the staircase in the plot. The force is now
# requested from the controller on every sample instead.
_ft_warned = False


def read_ext_force():
    """Current (Fx, Fy, Fz) [N] from the controller; falls back to the cached report value."""
    global _ft_warned
    code = None
    try:
        code, f = arm.get_ft_sensor_data()
        if code == 0 and len(f) >= 3:
            return tuple(float(v) for v in f[:3])
    except Exception as e:
        code = e
    if not _ft_warned:
        _ft_warned = True
        print(f"WARNING: get_ft_sensor_data failed ({code}) - force plot uses the slow "
              f"report value for such samples and will look stepped")
    return tuple(float(v) for v in arm.ft_ext_force[:3])


def sample_forces(samples, lock, stop):
    """Background thread: append (t, Fx, Fy, Fz) at FORCE_RATE_HZ until `stop` is set."""
    period = 1.0 / FORCE_RATE_HZ
    nxt = time.time()
    while not stop.is_set():
        f = read_ext_force()               # outside the lock: it is a network round trip
        with lock:
            samples.append((time.time(), *f))
        nxt = max(nxt + period, time.time())   # fixed rate, no drift from the read time
        time.sleep(nxt - time.time() if nxt > time.time() else 0.0)


def draw_force_plot(fig, ax, lines, samples, lock):
    with lock:
        data = np.array(samples)
    if len(data) > 1:
        t = data[:, 0] - data[-1, 0]
        for i, line in enumerate(lines):
            line.set_data(t, data[:, i + 1])
        ax.set_xlim(-FORCE_PLOT_SECONDS, 0)
        ax.relim()
        ax.autoscale_view(scalex=False)
        ax.set_title("  ".join(f"F{a} {v:+.2f} N" for a, v in zip("xyz", data[-1, 1:])))
    fig.canvas.draw_idle()
    fig.canvas.flush_events()


def spiral_insert(cx, cy, yaw):
    arm.ft_sensor_set_zero()

    samples = deque(maxlen=int(FORCE_PLOT_SECONDS * FORCE_RATE_HZ))
    lock, stop = threading.Lock(), threading.Event()

    def sample():                              # CHANGED (force plot)
        sample_forces(samples, lock, stop)

    plt.ion()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    lines = [ax.plot([], [], label=n)[0] for n in ("Fx", "Fy", "Fz")]
    ax.set_xlabel("t [s]")
    ax.set_ylabel("F [N]")
    ax.grid(True)
    ax.legend(loc="upper left")
    plt.show(block=False)

    threading.Thread(target=sample, daemon=True).start()
    try:
        set_admittance([200.0, 200.0, 10.0, 1.0, 1.0, 1.0], [0, 0, 1, 0, 0, 0], 0)

        inserted = False
        for x, y in generate_spiral(cx, cy, SPIRAL_R_START, SPIRAL_R_END, SPIRAL_STEPS):
        # for x, y in generate_figure8(cx, cy, SPIRAL_R_END, SPIRAL_STEPS, loops=3, angle_deg=yaw):
            arm.set_position(x=x, y=y, z=Z_SPIRAL_PRESS, roll=180, pitch=0, yaw=yaw,
                             speed=100, wait=True, timeout=0.1)
            draw_force_plot(fig, ax, lines, samples, lock)
            _, pos = arm.get_position()
            if pos[2] < Z_DROP_THRESHOLD:
                print(f"Dropped in: z={pos[2]:.1f}, stopping spiral.")
                inserted = True
                break

        _, pos = arm.get_position()
        set_admittance([40.0, 40.0, 500.0, 10.0, 10.0, 0.1], [1, 1, 1, 0, 0, 1])

        arm.set_state(0)
        arm.set_position(x=pos[0], y=pos[1], z=pos[2] - 40, roll=180, pitch=0, yaw=yaw,
                         speed=100, wait=True, timeout=10.0)
        t_end = time.time() + 3
        while time.time() < t_end:
            draw_force_plot(fig, ax, lines, samples, lock)
            time.sleep(0.03)
    finally:
        stop.set()
        fig.savefig(os.path.join(OUTPUT_DIR, "force.png"))
        plt.close(fig)
        plt.ioff()

    arm.set_ft_sensor_mode(0)
    arm.set_state(0)

    arm.set_gripper_position(250, wait=True)

    arm.set_position(x=pos[0], y=pos[1]+20, z=Z_RETREAT, roll=180, pitch=0, yaw=yaw,
                     speed=200, wait=True, timeout=10.0)

    arm.set_ft_sensor_enable(0)
    arm.set_state(0)
    print("done")
    return inserted

def sc_spiral_insert(cx, cy, yaw):
    """Spiral search into the SC port at the card's yaw. No push after drop-in;
    releases the plug and retreats whether or not it dropped in."""
    arm.ft_sensor_set_zero()

    samples = deque(maxlen=int(FORCE_PLOT_SECONDS * FORCE_RATE_HZ))
    lock, stop = threading.Lock(), threading.Event()

    def sample():                              # CHANGED (force plot)
        sample_forces(samples, lock, stop)

    plt.ion()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    lines = [ax.plot([], [], label=n)[0] for n in ("Fx", "Fy", "Fz")]
    ax.set_xlabel("t [s]")
    ax.set_ylabel("F [N]")
    ax.grid(True)
    ax.legend(loc="upper left")
    plt.show(block=False)

    threading.Thread(target=sample, daemon=True).start()
    inserted = False
    try:
        set_admittance([200.0, 200.0, 100.0, 1.0, 1.0, 1.0], [0, 0, 1, 0, 0, 0], 0)
        for x, y in generate_spiral(cx, cy, SPIRAL_R_START, SPIRAL_R_END, SPIRAL_STEPS):
            arm.set_position(x=x, y=y, z=SC_Z_SPIRAL_PRESS, roll=180, pitch=0, yaw=yaw,
                             speed=100, wait=True, timeout=1.0)
            draw_force_plot(fig, ax, lines, samples, lock)
            _, pos = arm.get_position()
            if pos[2] < SC_Z_DROP_THRESHOLD:
                print(f"SC dropped in: z={pos[2]:.1f}, stopping spiral.")
                inserted = True
                break

        _, pos = arm.get_position()
        set_admittance([40.0, 40.0, 700.0, 10.0, 10.0, 0.1], [1, 1, 1, 0, 0, 1])

        arm.set_state(0)
        arm.set_position(x=pos[0], y=pos[1]-40, z=pos[2]-20, roll=180, pitch=0, yaw=yaw,
                            speed=100, wait=True, timeout=10.0)
    
    finally:
        stop.set()
        plt.close(fig)
        plt.ioff()

    arm.set_ft_sensor_mode(0)
    arm.set_state(0)
    _, pos = arm.get_position()

    arm.set_gripper_position(SC_GRIPPER_OPEN, wait=True)
    arm.set_position(x=pos[0], y=pos[1]+20, z=SC_Z_RETREAT, roll=180, pitch=0, yaw=yaw,
                     speed=100, wait=True, timeout=10.0)

    arm.set_ft_sensor_enable(0)
    arm.set_state(0)
    print("SC done")
    return inserted

if __name__ == "__main__":
    init()
    if "--measure-z" in sys.argv:                   # NEW (v3)
        measure_origin_z()
        sys.exit(0)
    arm.set_gripper_position(500, wait=True)
    tx, ty, tz, troll, tpitch, tyaw = TARGET_POSITION
    arm.set_position(x=tx+100, y=ty, z=tz, roll=troll, pitch=tpitch, yaw=tyaw,
                     speed=MOVE_SPEED, wait=True, timeout=30.0)
    time.sleep(SETTLE_TIME)

    #-----scan nic card; pitch down and retry if not found within SCAN_TIMEOUT
    # pitch = tpitch
    # while True:
    res = estimate_object_pose(CARD_PATH, CARD_CLASS, CARD_CONFIDENCE)
        # if res is not None:
        #     break
        # pitch -= 0#PITCH_STEP
        # if pitch < PITCH_MIN:
        #     raise RuntimeError("NIC card not found within pitch range")
        # print(f"Card not found in {SCAN_TIMEOUT:.0f}s, pitch -> {pitch:.1f}")
        # arm.set_position(x=tx, y=ty, z=tz, roll=troll, pitch=pitch, yaw=tyaw,
        #                  speed=MOVE_SPEED, wait=True, timeout=30.0)
        # time.sleep(SETTLE_TIME)
    x_card_rough, y_card_rough, yaw_card_rough = res
    yaw_card_rough -= 90

    arm.set_position(x=tx, y=ty, z=410.0, roll=troll, pitch=tpitch, yaw=tyaw,
                     speed=MOVE_SPEED, wait=True, timeout=30.0)

    res = estimate_object_pose(PLUG_PATH, PLUG_CLASS, PLUG_CONFIDENCE, timeout = 30)
    # arm.set_position(x=tx, y=ty, z=210, roll=troll, pitch=tpitch, yaw=tyaw,
    #                              speed=MOVE_SPEED, wait=True, timeout=30.0)
    # pitch = tpitch
    # while True:
    #         res = estimate_object_pose(PLUG_PATH, PLUG_CLASS, PLUG_CONFIDENCE, timeout = 30)
    #         if res is not None:
    #             break
    #         pitch += PITCH_STEP
    #         if pitch < PITCH_MIN:
    #             raise RuntimeError("NIC card not found within pitch range")
    #         print(f"Card not found in {SCAN_TIMEOUT:.0f}s, pitch -> {pitch:.1f}")
    #         arm.set_position(x=tx, y=ty, z=210, roll=troll, pitch=pitch, yaw=tyaw,
    #                          speed=MOVE_SPEED, wait=True, timeout=30.0)
    #         time.sleep(SETTLE_TIME)
    print(res)
    x_plug_rough, y_plug_rough, yaw_plug_rough = res
    yaw_plug_rough -= 180



    #-----getting close to the plug
    arm.set_position(x=x_plug_rough - 70*np.cos(np.deg2rad(yaw_plug_rough)), y=y_plug_rough + 70*np.sin(np.deg2rad(yaw_plug_rough)), z=220, roll=180, pitch=0, yaw=yaw_plug_rough, speed=250, wait=True)

    #-----geting the plug out of its holder
    x_plug_fine, y_plug_fine, yaw_plug_fine = estimate_object_pose(PLUG_PATH, PLUG_CLASS, PLUG_CONFIDENCE)
    # x_plug_fine, y_plug_fine, yaw_plug_fine = x_plug_rough, y_plug_rough, yaw_plug_rough
    yaw_plug_fine -= 180
    arm.set_position(x=x_plug_fine, y=y_plug_fine, z=220, roll=180, pitch=0, yaw=yaw_plug_fine, speed=100, wait=True)
    arm.set_position(x=x_plug_fine - 128*np.cos(np.deg2rad(yaw_plug_fine)), y=y_plug_fine - 128*np.sin(np.deg2rad(yaw_plug_fine)), z=93, roll=180, pitch=-48, yaw=yaw_plug_fine, speed=100, wait=True)
    #z = 73, offs 172
    set_admittance([40, 200, 40, 2, 2, 2], [0, 1, 0, 0, 0, 0])
    arm.set_gripper_position(-10, wait=True)
    arm.set_tool_position(x=0, y=0, z=-70, roll=0, pitch=0, yaw=0, speed=100, wait=True)
    arm.set_ft_sensor_mode(0)
    arm.set_ft_sensor_enable(0)
    arm.set_state(0)

    code, pose = arm.get_position(is_radian=False)
    x_curr, y_curr = pose[:2]
    arm.set_position(x=x_curr, y=y_curr, z=350, roll=180, pitch=0, yaw=yaw_card_rough, speed=100, wait=True)

    #-----get above nic card
    print("GOING ABOVE NIC CARD")
    # arm.set_position(x=x_card_rough - 270*np.cos(np.deg2rad(yaw_card_rough)), y=y_card_rough - 270*np.sin(np.deg2rad(yaw_card_rough)), z=165, roll=180, pitch=-47, yaw=yaw_card_rough, speed=200, wait=True)
    # arm.set_position(x=510, y=-191, z=237, roll=180, pitch=-45, yaw=yaw_card_rough, speed=200, wait=True)
    # x_card_fine, y_card_fine, yaw_card_fine = estimate_object_pose(CARD_PATH, CARD_CLASS, CARD_CONFIDENCE)
    # yaw_card_fine -= 90
    # time.sleep(10)
    OFFSET = -8
    x_card_fine, y_card_fine, yaw_card_fine = x_card_rough + OFFSET*np.cos(np.deg2rad(yaw_plug_rough)), y_card_rough + OFFSET*np.sin(np.deg2rad(yaw_plug_rough)), yaw_card_rough
    arm.set_position(x=x_card_fine, y=y_card_fine, z=375, roll=180, pitch=0, yaw=yaw_card_fine, speed=300, wait=True)
    arm.set_position(x=x_card_fine, y=y_card_fine, z=340, roll=180, pitch=0, yaw=yaw_card_fine, speed=100, mvacc=500, wait=True)
    print(f"Inserted: {spiral_insert(x_card_fine, y_card_fine, yaw_card_fine)}")

    # SC PLUG PICKUP
    arm.set_position(x=x_plug_fine-70, y=y_plug_fine-70, z=250.0, roll=troll, pitch=tpitch, yaw=yaw_plug_fine,
                         speed=MOVE_SPEED, wait=True, timeout=30.0)
    pitch = tpitch
    # while True:
    #         res = estimate_object_pose(SC_PLUG_PATH, SC_PLUG_CLASS, PLUG_CONFIDENCE, timeout = 30)
    #         if res is not None:
    #             break
    #         pitch += PITCH_STEP
    #         if pitch < PITCH_MIN:
    #             raise RuntimeError("NIC card not found within pitch range")
    #         print(f"Card not found in {SCAN_TIMEOUT:.0f}s, pitch -> {pitch:.1f}")
    #         arm.set_position(x=tx, y=ty, z=210, roll=troll, pitch=pitch, yaw=tyaw,
    #                          speed=MOVE_SPEED, wait=True, timeout=30.0)
    #         time.sleep(SETTLE_TIME)
    res = estimate_object_pose(SC_PLUG_PATH, SC_PLUG_CLASS, PLUG_CONFIDENCE, timeout = 30)
    x_plug_rough, y_plug_rough, yaw_plug_rough = res
    yaw_plug_rough -= 180

    #-----getting close to the plug
    arm.set_position(x=x_plug_rough - 70*np.cos(np.deg2rad(yaw_plug_rough)), y=y_plug_rough + 70*np.sin(np.deg2rad(yaw_plug_rough)), z=220, roll=180, pitch=0, yaw=yaw_plug_rough, speed=250, mvacc=2000, wait=True)

    #-----geting the plug out of its holder
    # x_plug_fine, y_plug_fine, yaw_plug_fine = estimate_object_pose(PLUG_PATH, PLUG_CLASS, PLUG_CONFIDENCE)
    x_plug_fine, y_plug_fine, yaw_plug_fine = x_plug_rough, y_plug_rough, yaw_plug_rough
    # yaw_plug_fine -= 180
    arm.set_position(x=x_plug_fine, y=y_plug_fine, z=212, roll=180, pitch=0, yaw=yaw_plug_fine, speed=100, wait=True)
    arm.set_position(x=x_plug_fine - 128*np.cos(np.deg2rad(yaw_plug_fine)), y=y_plug_fine - 128*np.sin(np.deg2rad(yaw_plug_fine)), z=68, roll=180, pitch=-48, yaw=yaw_plug_fine, speed=100, wait=True)
    set_admittance([40, 200, 40, 2, 2, 2], [0, 1, 0, 0, 0, 0])
    arm.set_gripper_position(-10, wait=True)
    arm.set_tool_position(x=0, y=0, z=-70, roll=0, pitch=0, yaw=0, speed=100, wait=True)
    arm.set_ft_sensor_mode(0)
    arm.set_ft_sensor_enable(0)
    arm.set_state(0)

    code, pose = arm.get_position(is_radian=False)
    x_curr, y_curr = pose[:2]
    arm.set_position(x=x_card_fine  - 300*np.cos(np.deg2rad(yaw_card_fine)), y=y_card_fine - 300*np.sin(np.deg2rad(yaw_card_fine)), z=180, roll=180, pitch=0, yaw=yaw_card_fine, speed=100, wait=True)
    res = estimate_object_pose(SC_PORT_PATH, SC_PORT_CLASS, PLUG_CONFIDENCE, timeout = 60)

    x_port_rough, y_port_rough, yaw_port_rough = res
    print(f"SC port YAW : {yaw_port_rough}")
    print(f"SC port xoffs : {8*np.cos(np.deg2rad(yaw_port_rough))}")
    #inserting SC port
    if 125 > yaw_port_rough > 95:
        yaw_port_rough -= 180 
    arm.set_position(x=x_port_rough+8*np.cos(np.deg2rad(yaw_port_rough)), y=y_port_rough-8*np.sin(np.deg2rad(yaw_port_rough)), z=180, roll=180, pitch=0, yaw=yaw_port_rough-180, speed=100, wait=True)
    print(f"SC inserted: {sc_spiral_insert(x_port_rough+8*np.cos(np.deg2rad(yaw_port_rough)), y_port_rough-1*np.sin(np.deg2rad(yaw_port_rough)), yaw_port_rough-180)}")
    arm.move_gohome(speed=30, wait=True)