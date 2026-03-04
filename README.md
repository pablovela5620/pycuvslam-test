# pycuvslam-test

Multicamera visual odometry and SLAM using [NVIDIA cuVSLAM](https://github.com/NVlabs/PyCuVSLAM), with [Rerun](https://rerun.io) visualization. Refactored from standalone scripts into a typed Python package with a dataset abstraction, tyro-based CLI, and optimized video logging.

## Project structure

```
pycuvslam-test/
├── pyproject.toml                          # Package metadata, pixi deps and tasks
├── pycuvslam/                              # Main package (editable install)
│   ├── __init__.py                         # beartype activation (dev env only)
│   ├── apis/
│   │   ├── track_odometry.py               # Odometry pipeline + tyro entrypoint
│   │   └── track_slam.py                   # SLAM pipeline + tyro entrypoint
│   ├── configs/
│   │   ├── base_config.py                  # Re-exports InstantiateConfig from simplecv
│   │   └── track_dataset_configs.py        # Dataset registry (tyro subcommands)
│   ├── data/
│   │   ├── base.py                         # BaseTrackDataset ABC + BaseTrackDatasetConfig
│   │   └── robocap.py                      # RobocapTrackDataset implementation
│   └── visualization.py                    # Rerun logging helpers
├── tools/                                  # CLI entry-point scripts
│   ├── track_odometry.py                   # python tools/track_odometry.py <dataset>
│   └── track_slam.py                       # python tools/track_slam.py <dataset>
└── examples/                               # Legacy standalone scripts (for comparison)
    ├── track_robocap.py                    # Original per-frame Image approach
    └── robocap_utils.py                    # Shared utils for legacy scripts
```

## Setup

Requires [pixi](https://prefix.dev/docs/pixi/overview) for dependency management.

```bash
pixi install          # Install default environment
pixi install -e dev   # Install dev environment (adds beartype, ruff)
```

Two environments are available:

| Environment | Purpose | Extras |
|---|---|---|
| `default` | Running pipelines | cuVSLAM, Rerun, simplecv, tyro |
| `dev` | Development | + beartype (runtime type checking), ruff (linting) |

beartype activates automatically when `PIXI_ENVIRONMENT_NAME=dev` (set by pixi). No decorators needed -- `beartype_this_package()` in `pycuvslam/__init__.py` instruments the entire package via AST import hooks.

## Usage

### Pixi tasks

```bash
pixi run track-robocap           # Multicamera odometry on Robocap data
pixi run track-robocap-slam      # Multicamera SLAM on Robocap data
pixi run track-robocap-legacy    # Legacy script (per-frame Image, for comparison)

pixi run -e dev track-robocap    # With runtime type checking
```

### CLI flags (tyro)

The tools use [tyro](https://brentyi.github.io/tyro/) to generate CLI arguments from dataclass configs. The dataset name is a positional subcommand:

```bash
# Default robocap run (spawns Rerun viewer)
python tools/track_odometry.py robocap

# Custom data path and session
python tools/track_odometry.py robocap \
    --root-directory /data/robocap \
    --device-id abc123 \
    --session-id 5 \
    --segment-id 2 \
    --pairs 0 2

# Save to RRD file
python tools/track_odometry.py robocap --rr-config.save output.rrd

# Connect to existing Rerun instance
python tools/track_odometry.py robocap --rr-config.connect

# SLAM with synchronous mode
python tools/track_slam.py robocap --slam-sync-mode
```

Run `python tools/track_odometry.py --help` or `python tools/track_odometry.py robocap --help` to see all flags.

## Architecture

### Dataset abstraction

`BaseTrackDataset` (in `pycuvslam/data/base.py`) defines the interface every dataset must implement:

```python
class BaseTrackDataset(ABC):
    cameras: list[cuvslam.Camera]               # cuVSLAM camera objects
    cam_names: list[str]                        # Human-readable names
    cam_params: dict[str, CameraParam]          # Intrinsics + extrinsics
    video_paths: dict[str, Path]                # Per-camera video files
    n_frames: int                               # Total frame count
    video_timestamps_ns: Int[ndarray, "n_frames"]  # Nanosecond timestamps
    image_plane_distance: float                 # For Rerun frustum sizing

    def get_frame(self, frame_idx: int) -> list[UInt8[np.ndarray, "h w 3"]]: ...
```

Each dataset has a paired config dataclass (subclass of `BaseTrackDatasetConfig`) with a `_target` field pointing to the dataset class. Calling `config.setup()` instantiates the dataset.

### Dataset registry (tyro subcommands)

Datasets are registered in `pycuvslam/configs/track_dataset_configs.py`:

```python
track_dataset_defaults: dict[str, BaseTrackDatasetConfig] = {
    "robocap": RobocapTrackConfig(),
}
```

`tyro.extras.subcommand_type_from_defaults` generates CLI subcommands from this dict. The key becomes the subcommand name, the value provides defaults.

### Visualization: AssetVideo vs per-frame Image

The package uses `rr.AssetVideo` instead of per-frame `rr.Image` for a significant speedup:

| Approach | What happens | Cost |
|---|---|---|
| **Per-frame Image** (legacy) | Each frame: decode -> JPEG re-encode -> transmit pixels | CPU + bandwidth per frame |
| **AssetVideo** (package) | Upload MP4 blob once, send `VideoFrameReference` timestamps | One-time upload, near-zero per-frame |

The MP4 is logged as `rr.AssetVideo(path=..., static=True)` once. Frame timestamps are read from the video container via `read_frame_timestamps_nanos()` and batched with `rr.send_columns`. Rerun handles GPU-accelerated seeking on the viewer side.

## Rerun entity structure

### Odometry

```
/                                   ViewCoordinates (static)
rig/                                Transform3D (rig pose, per-frame)
rig/cam{i}/                         Transform3D (extrinsics, static)
rig/cam{i}/pinhole/                 Pinhole (intrinsics, static)
rig/cam{i}/pinhole/video            AssetVideo (static) + VideoFrameReference
rig/cam{i}/pinhole/observations     Points2D (2D feature tracks, per-frame)
rig/landmarks                       Points3D (visible landmarks, per-frame)
trajectory                          LineStrips3D (logged every 10 frames)
final_landmarks                     Points3D (all landmarks at end)
```

### SLAM (additional entities)

```
odom_trajectory                     LineStrips3D (cyan, every 10 frames)
slam_trajectory                     LineStrips3D (green, every 10 frames)
loop_closures                       Points3D (red)
pose_graph/nodes                    Points3D (yellow)
pose_graph/edges                    LineStrips3D (gray)
slam_metrics                        TextLog (every 100 frames)
```

Timelines: `"video_time"` (duration in seconds, from actual PTS) and `"frame"` (sequence index).

## Adding a new dataset

1. **Create dataset and config classes** in `pycuvslam/data/your_dataset.py`:

```python
from dataclasses import dataclass, field
from pycuvslam.data.base import BaseTrackDataset, BaseTrackDatasetConfig

@dataclass
class YourDatasetConfig(BaseTrackDatasetConfig):
    """Configuration for YourDataset."""

    _target: type = field(default_factory=lambda: YourDataset)
    """Target class to instantiate."""
    root_directory: Path = Path("/path/to/data")
    """Root directory of the dataset."""
    # ... dataset-specific fields

class YourDataset(BaseTrackDataset):
    def __init__(self, cfg: YourDatasetConfig) -> None:
        # Load calibration, set up video readers, etc.
        ...

    # Implement all abstract properties and methods from BaseTrackDataset
```

2. **Register the dataset** in `pycuvslam/configs/track_dataset_configs.py`:

```python
from pycuvslam.data.your_dataset import YourDatasetConfig

track_dataset_defaults: dict[str, BaseTrackDatasetConfig] = {
    "robocap": RobocapTrackConfig(),
    "your-dataset": YourDatasetConfig(),  # Add this line
}
```

3. **Run it**:

```bash
python tools/track_odometry.py your-dataset
python tools/track_odometry.py your-dataset --root-directory /other/path
```

The key requirements for the dataset implementation:
- `cameras` must return `cuvslam.Camera` objects with correct intrinsics, distortion, and `rig_from_camera` transforms
- `video_paths` must point to MP4 files for `AssetVideo` logging
- `video_timestamps_ns` must return nanosecond timestamps matching the video container PTS (use `rr.AssetVideo(path).read_frame_timestamps_nanos()`)
- `get_frame()` must return BGR images in the same order as `cameras`
