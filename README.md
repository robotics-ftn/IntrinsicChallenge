# YOLO + MegaPose pose estimation and SFP / SC insertion

These scripts find connector parts with a wrist-mounted RealSense camera on an xArm and plug them in.
YOLO finds each part in the image. MegaPose (from [happypose](https://github.com/agimus-project/happypose))
estimates its 6D pose against a CAD mesh. The arm then picks the part and inserts it with a
compliant spiral search under force/admittance control.

## Context

This code was developed for the **Intrinsic AI for Industry Challenge**. The task is robotic assembly
of network hardware: SFP modules go into NIC cards, and SC fibre connectors go into SC ports. The same
pipeline is the experimental basis for an accompanying research paper (in preparation).

- **Challenge:** Intrinsic AI for Industry Challenge
- **Paper:** *TBD (title)*, *TBD (authors)*, *TBD (venue / year)*

If you use this code, please cite the paper. A BibTeX entry will be added here once it is available:

```bibtex
@misc{TBD,
  title  = {TBD},
  author = {TBD},
  year   = {TBD},
  note   = {Developed for the Intrinsic AI for Industry Challenge}
}
```

---

## Scripts

| Script | What it does |
|---|---|
| `scripts/6d_pose_est_yolo.py` | **Single pair.** Finds the NIC card, then the SFP plug. Picks the SFP and spiral-inserts it into the card. Then finds the SC plug and the SC port, picks the SC plug and spiral-inserts it into the port. |
| `scripts/6d_pose_est_yolo_multi.py` | **Two pairs.** Surveys both NIC cards and both SFPs from an overhead view, and YOLO-detects both SC ports. It assigns SFP → card and SC plug → port, then runs the single-pair sequence twice. It also includes twin masking and twin search, close-up re-looks, and survey images. The script docstring has the changelog (v7–v20). |

Both scripts support `--measure-z`. This one-off option triangulates the card's mesh-origin height from
two viewpoints (`MEASURE_Z_VIEWS`). Put the result in `CARD_ORIGIN_Z_MM` to turn on the known-height
constraint.

---

## Folder layout

```
yolo_megapose_insertion/
├── README.md
├── scripts/
│   ├── 6d_pose_est_yolo.py          # single-pair pipeline
│   └── 6d_pose_est_yolo_multi.py    # two-pair pipeline
├── tools/
│   └── camera_intrinsics.py         # ChArUco intrinsics -> config/camera_data.json
├── config/
│   └── camera_data.json             # camera matrix K + resolution (read by both scripts)
├── models/
│   └── yolo/
│       ├── best.pt                  # YOLOv8m detector (classes below)
│       ├── args.yaml                # training arguments (for reference)
│       └── results.csv              # training curves (for reference)
├── meshes/                          # one sub-folder per object = one MegaPose label
│   ├── card/     my_object.obj/.mtl + texture.png
│   ├── plug/     my_object.obj/.mtl + texture.png   (SFP plug)
│   ├── sc_plug/  my_object.obj/.mtl + texture.png
│   └── sc_port/  my_object.obj/.mtl + texture.png
└── outputs/                         # written at runtime
```

### Path conventions

All paths come from the project folder (`PROJECT_DIR`, the parent of `scripts/`). Absolute paths are
not hard-coded. You can move the folder, and you can start the scripts from any working directory.
The path constants are at the top of each script:

| Constant | Points to |
|---|---|
| `MESH_DIR`, `CARD_PATH`, `PLUG_PATH`, `SC_PLUG_PATH`, `SC_PORT_PATH` | `meshes/…` |
| `YOLO_MODEL_PATH` | `models/yolo/best.pt` |
| `CAMERA_DATA_PATH` | `config/camera_data.json` |
| `OUTPUT_DIR` | `outputs/` (created automatically) |

> **Mesh folder note.** MegaPose loads **every sub-folder of `meshes/`** as one object. It uses the
> first mesh file in each folder, in alphabetical order. Don't put extra folders or alternative
> `.obj` files in `meshes/`. Keep old mesh versions somewhere else.

---

## Requirements

- **Hardware:** UFactory xArm with a force/torque sensor and gripper (IP `ARM_IP`), and an Intel
  RealSense D415 (`CAMERA_SERIAL`). The camera is mounted on the flange with the hand-eye transform
  `T_cam2flange` hard-coded in the scripts.
- **GPU:** CUDA (`DEVICE = "cuda"`).
- **Python packages:** `happypose` (MegaPose), `torch`, `ultralytics`, `pyrealsense2`,
  `xarm-python-sdk` (`xarm.wrapper`), `opencv-python` (with `aruco` for the calibration tool), `numpy`,
  `pandas`, `matplotlib` (TkAgg backend, so `python3-tk` is needed).
- **MegaPose weights:** these are *not* in this folder. The scripts set
  `MEGAPOSE_DATA_DIR=~/happypose_data` (unless it is already set). They expect the downloaded models
  in `~/happypose_data/megapose-models/`. The networks used are `megapose-1.0-RGB` (RGB only, the default)
  and `megapose-1.0-RGB-multi-hypothesis-icp` (when `USE_DEPTH = True`).

---

## Running

Activate the environment where happypose is installed, then run:

```bash
cd yolo_megapose_insertion

# single pair
python scripts/6d_pose_est_yolo.py

# two pairs
python scripts/6d_pose_est_yolo_multi.py

# one-off: measure the card origin height (either script)
python scripts/6d_pose_est_yolo.py --measure-z
```

A preview window shows the YOLO boxes and MegaPose overlays while the scripts run, and a live force
plot is shown during insertion. Press Ctrl+C to stop. Both scripts shut down cleanly and release the
camera, the render workers and the GPU.

### Outputs (`outputs/`)

| File | Written by | Content |
|---|---|---|
| `force.png` | both | force plot of the last spiral insertion (overwritten each time) |
| `survey_view<N>_<class>.png` | multi | survey frame with numbered YOLO boxes and where each pose landed (`SURVEY_IMAGES = True`) |

---

## Detector and objects

`models/yolo/best.pt` is a YOLOv8m model, trained at `imgsz=1280` (see `args.yaml`). The scripts use
these classes:

| YOLO class | Mesh folder | Confidence constant |
|---|---|---|
| `nic_card` | `meshes/card` | `CARD_CONFIDENCE` (0.5) |
| `sfp_plug` | `meshes/plug` | `PLUG_CONFIDENCE` (0.5) |
| `sc_plug` | `meshes/sc_plug` | `PLUG_CONFIDENCE` (the single script) |
| `sc_port` | `meshes/sc_port` | `PLUG_CONFIDENCE` |

Meshes are in millimetres (`MESH_UNITS = "mm"`). The card mesh is centred, and `POINT_OFFSET_LOCAL` in
the scripts matches that origin.

---

## Calibration

### Camera intrinsics: `config/camera_data.json`

When `USE_RS_INTRINSICS = False` (the default), K comes from this file. Otherwise the scripts use the
RealSense factory calibration. To redo the calibration, use a 7×10 ChArUco board (DICT_5X5_100,
24.5 / 18.5 mm):

```bash
python tools/camera_intrinsics.py      # SPACE = capture, Q = calibrate, ESC = abort
```

The tool writes the pinhole K to `config/camera_data.json` and backs up the old file next to it as
`camera_data_<timestamp>.json`. It writes the full model (K + distortion) to
`config/calibration_<serial>.npz`.

### Hand-eye: `T_cam2flange`

This transform is hard-coded near the top of each script. It came from the original project's
`hand_eye_calibration.py`, which is not copied here. If the camera mount changes, rerun that
calibration and paste the new 4×4 matrix into both scripts.

---

## Tuning entry points

All settings are constants at the top of each script and are commented in place. The ones you will
change most often:

- **Robot / camera:** `ARM_IP`, `CAMERA_SERIAL`, `TARGET_POSITION` (the first look pose), `MOVE_SPEED`.
- **Stability gate:** `POSE_SCORE_MIN`, `STABLE_WINDOW`, `STABLE_TOL_MM` / `STABLE_TOL_DEG` (and
  their `_MAX` relax values).
- **Per-class priors:** `POSE_PRIORS` sets the up axis, tilt and yaw limits, upright snap, flip fix
  for the SC plug, known height (`plane_z_mm`) and fast-yaw mode.
- **Insertion:** `Z_SPIRAL_PRESS`, `Z_DROP_THRESHOLD`, `Z_RETREAT`, `SPIRAL_*`, `SC_Z_*`. The multi
  script also has `CARD_Z` and `PORT_Z`, with one entry per pair.
- **Multi script only:** `OVERHEAD_VIEWS`, `SURVEY_CLASSES`, `PORT_TABLE_Z_MM`, `MASK_TWINS`,
  `TWIN_*`, `CLOSEUP_*`.

---

## Provenance

This folder is a self-contained copy of
`~/happypose_data/examples/my_object/` (the original files are unchanged there):

| Here | Original |
|---|---|
| `scripts/6d_pose_est_yolo.py` | `scripts/6d_pose_est_yolo.py` |
| `scripts/6d_pose_est_yolo_multi.py` | `scripts/6d_pose_est_yolo multi.py` (space removed from the name) |
| `tools/camera_intrinsics.py` | `scripts/camera_intrinsics.py` |
| `config/camera_data.json` | `camera_data.json` |
| `models/yolo/*` | `run3/exp/weights/best.pt`, `run3/exp/args.yaml`, `run3/exp/results.csv` |
| `meshes/*/` | `meshes/*/` (active `my_object.*` + `texture.png` only, without the `_old` / `_not_centered` variants) |

The code is unchanged apart from the path constants, which now use `PROJECT_DIR`, and the output files,
which now go to `outputs/`. The YOLO training scripts and dataset are still in the original
`yolo_scripts/`.
