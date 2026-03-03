"""Shared utilities for Robocap multicamera tracking examples.

Provides calibration loading, video reading, Rerun visualization helpers,
and CLI argument setup used by both odometry and SLAM scripts.
"""

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import cuvslam
import cv2
import numpy as np
import rerun as rr
import yaml
from jaxtyping import Float64, UInt8
from scipy.spatial.transform import Rotation

# Runtime type checking only in dev environment (zero overhead in production)
if os.environ.get("PIXI_ENVIRONMENT_NAME") == "dev":
    from beartype import beartype
    from jaxtyping import jaxtyped
else:
    from typing import Any

    def beartype(fn: Any) -> Any:  # type: ignore[no-redef]
        return fn

    def jaxtyped(fn: Any = None, *, typechecker: Any = None) -> Any:  # type: ignore[no-redef]
        return fn if fn is not None else lambda f: f


# --- Kalibr YAML → camera name mapping ---
# Each calibration YAML may contain 1 or 2 cameras (cam0, cam1).
# Maps: (yaml_directory_name, cam_index_in_yaml) → logical camera name
CALIB_MAP: dict[str, list[str]] = {
    "imus_cam_lr_front_extrinsic": ["left_front", "right_front"],
    "imus_cam_l_extrinsic": ["left"],
    "imus_cam_r_extrinsic": ["right"],
}

# Video filename suffix → logical camera name
VIDEO_SUFFIX_TO_NAME: dict[str, str] = {
    "left-front": "left_front",
    "right-front": "right_front",
    "left": "left",
    "right": "right",
    "left-eye": "left_eye",
    "right-eye": "right_eye",
}

# Stereo pair definitions: (left_cam, right_cam)
# 3 pairs from 4 physical cameras (left_front and right_front are shared)
STEREO_PAIRS: list[tuple[str, str]] = [
    ("left", "left_front"),
    ("left_front", "right_front"),
    ("right_front", "right"),
]


@beartype
def color_from_id(identifier: int) -> list[int]:
    """Generate pseudo-random colour from integer identifier for visualization."""
    return [
        (identifier * 17) % 256,
        (identifier * 31) % 256,
        (identifier * 47) % 256,
    ]


# ---------------------------------------------------------------------------
# Kalibr calibration loading
# ---------------------------------------------------------------------------


@beartype
def load_calibrations(calib_root: Path) -> dict[str, dict]:
    """Load Kalibr calibration YAMLs and return per-camera dicts.

    Returns:
        Mapping of camera_name to dict with keys:
            "T_cam_imu": np.ndarray (4x4),
            "intrinsics": [fx, fy, cx, cy],
            "distortion_coeffs": [k1, k2, k3, k4],
            "resolution": [w, h],
    """
    cameras: dict[str, dict] = {}
    for calib_dir, cam_names in CALIB_MAP.items():
        yaml_path = calib_root / calib_dir / f"{calib_dir}-camchain-imucam.yaml"
        try:
            with open(yaml_path) as f:
                data = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Calibration not found: {yaml_path}") from None
        for idx, name in enumerate(cam_names):
            cam_key: str = f"cam{idx}"
            cam_data = data[cam_key]
            T_cam_imu: Float64[np.ndarray, "4 4"] = np.array(
                cam_data["T_cam_imu"], dtype=np.float64
            )
            cameras[name] = {
                "T_cam_imu": T_cam_imu,
                "intrinsics": cam_data["intrinsics"],
                "distortion_coeffs": cam_data["distortion_coeffs"],
                "resolution": cam_data["resolution"],
            }
    return cameras


@beartype
def calibration_to_cuvslam_camera(calib: dict) -> cuvslam.Camera:
    """Convert a Kalibr calibration dict to a cuvslam.Camera."""
    cam = cuvslam.Camera()
    w, h = calib["resolution"]
    fx, fy, cx, cy = calib["intrinsics"]
    cam.size = (w, h)
    cam.focal = (fx, fy)
    cam.principal = (cx, cy)

    k1, k2, k3, k4 = calib["distortion_coeffs"][:4]
    cam.distortion = cuvslam.Distortion(
        cuvslam.Distortion.Model.Fisheye, [k1, k2, k3, k4]
    )

    # rig_from_camera = imu_T_cam = inv(T_cam_imu)
    T_cam_imu: Float64[np.ndarray, "4 4"] = calib["T_cam_imu"]
    imu_T_cam: Float64[np.ndarray, "4 4"] = np.linalg.inv(T_cam_imu)
    quat_xyzw: Float64[np.ndarray, "4"] = Rotation.from_matrix(
        imu_T_cam[:3, :3]
    ).as_quat()
    cam.rig_from_camera = cuvslam.Pose(rotation=quat_xyzw, translation=imu_T_cam[:3, 3])
    return cam


# ---------------------------------------------------------------------------
# Video loading
# ---------------------------------------------------------------------------


@beartype
def find_videos(session_dir: Path) -> dict[str, Path]:
    """Find video files and map logical camera name → path."""
    videos: dict[str, Path] = {}
    for mp4 in sorted(session_dir.glob("*.mp4")):
        # filename pattern: video_devN_sessionN_segmentN_<suffix>.mp4
        suffix: str = mp4.stem.rsplit("_", 1)[-1]
        if suffix in VIDEO_SUFFIX_TO_NAME:
            videos[VIDEO_SUFFIX_TO_NAME[suffix]] = mp4
    return videos


class MultiVideoReader:
    """Synchronised frame reader for multiple video files."""

    @beartype
    def __init__(self, video_paths: dict[str, Path]) -> None:
        self.names: list[str] = list(video_paths.keys())
        self.caps: dict[str, cv2.VideoCapture] = {
            name: cv2.VideoCapture(str(path)) for name, path in video_paths.items()
        }
        # Frame count = minimum across all cameras
        counts: list[int] = [
            int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in self.caps.values()
        ]
        self.n_frames: int = min(counts)
        # FPS from first camera
        first_cap: cv2.VideoCapture = next(iter(self.caps.values()))
        self.fps: float = first_cap.get(cv2.CAP_PROP_FPS) or 30.0

    @jaxtyped(typechecker=beartype)
    def read(self) -> tuple[bool, dict[str, UInt8[np.ndarray, "h w 3"]]]:
        """Read one synchronised frame from all cameras.

        Returns:
            Tuple of (success, {camera_name: bgr_image}).
        """
        frames: dict[str, UInt8[np.ndarray, "h w 3"]] = {}
        for name, cap in self.caps.items():
            ret, frame = cap.read()
            if not ret:
                return False, {}
            frames[name] = frame
        return True, frames

    @beartype
    def release(self) -> None:
        for cap in self.caps.values():
            cap.release()


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


@beartype
def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common CLI arguments shared by odometry and SLAM scripts."""
    parser.add_argument(
        "--root",
        type=str,
        default="/mnt/8tb/data/robocap",
        help="Robocap dataset root directory",
    )
    parser.add_argument("--device-id", type=str, default="f408193e6447b3b0")
    parser.add_argument("--session-id", type=int, default=14)
    parser.add_argument("--segment-id", type=int, default=1)
    parser.add_argument(
        "--pairs",
        type=int,
        nargs="+",
        default=[0, 2],
        help="Stereo pair indices: 0=left|left_front, "
        "1=left_front|right_front, 2=right_front|right. "
        "Default [0,2] uses all 4 unique cameras. "
        "Note: pairs sharing cameras (e.g. 0+1) are not supported.",
    )


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RobocapSetup:
    """Result of setting up cameras and video readers from CLI args."""

    cameras: list[cuvslam.Camera]
    cam_order: list[str]
    all_videos: dict[str, Path]
    reader: MultiVideoReader
    frame_dt_ns: int


@beartype
def setup_cameras_and_reader(args: argparse.Namespace) -> RobocapSetup:
    """Load calibrations, find videos, build camera list, and open readers.

    Args:
        args: Parsed CLI namespace with root, device_id, session_id, pairs.

    Returns:
        RobocapSetup with cameras, cam_order, video reader, and frame timing.
    """
    root = Path(args.root)

    # Load calibrations
    calib_root: Path = root / f"0factory-calibration-{args.device_id}"
    calibrations: dict[str, dict] = load_calibrations(calib_root)
    print(f"Loaded calibrations for: {list(calibrations.keys())}")

    # Find videos
    session_dir: Path = root / f"{args.device_id}_session_{args.session_id}"
    all_videos: dict[str, Path] = find_videos(session_dir)
    print(f"Found videos: {list(all_videos.keys())}")

    # Build camera list from selected stereo pairs
    selected_pairs: list[tuple[str, str]] = [STEREO_PAIRS[i] for i in args.pairs]
    cam_order: list[str] = []
    for left, right in selected_pairs:
        cam_order.extend([left, right])

    cameras: list[cuvslam.Camera] = [
        calibration_to_cuvslam_camera(calibrations[name]) for name in cam_order
    ]
    print(f"Stereo pairs: {selected_pairs}")
    print(f"Camera order ({len(cameras)} slots): {cam_order}")

    # Open video readers (only the cameras we need)
    physical_cams: list[str] = list(dict.fromkeys(cam_order))
    video_paths: dict[str, Path] = {name: all_videos[name] for name in physical_cams}
    reader = MultiVideoReader(video_paths)
    frame_dt_ns: int = int(1e9 / reader.fps)
    print(
        f"Video FPS: {reader.fps:.1f} (dt={frame_dt_ns}ns), frames: {reader.n_frames}"
    )

    return RobocapSetup(
        cameras=cameras,
        cam_order=cam_order,
        all_videos=all_videos,
        reader=reader,
        frame_dt_ns=frame_dt_ns,
    )


# ---------------------------------------------------------------------------
# Rerun visualization helpers
# ---------------------------------------------------------------------------


@beartype
def log_static_camera_frustums(cameras: list[cuvslam.Camera]) -> None:
    """Log static camera transforms and pinhole models to Rerun."""
    for i, cam in enumerate(cameras):
        rr.log(
            f"rig/cam{i}",
            rr.Transform3D(
                translation=cam.rig_from_camera.translation,
                rotation=rr.Quaternion(xyzw=cam.rig_from_camera.rotation),
            ),
            static=True,
        )
        rr.log(
            f"rig/cam{i}",
            rr.Pinhole(
                image_plane_distance=0.025,
                image_from_camera=np.array(
                    [
                        [cam.focal[0], 0, cam.principal[0]],
                        [0, cam.focal[1], cam.principal[1]],
                        [0, 0, 1],
                    ],
                    dtype=np.float64,
                ),
                width=cam.size[0],
                height=cam.size[1],
                camera_xyz=rr.ViewCoordinates.RDF,
            ),
            static=True,
        )


@beartype
def log_frame_visuals(
    cameras: list[cuvslam.Camera],
    images: list[UInt8[np.ndarray, "h w 3"]],
    observations: list,
    landmarks: list,
    odom_pose: cuvslam.Pose,
) -> None:
    """Log per-frame rig pose, landmarks, camera images, and observations."""
    rr.log(
        "rig",
        rr.Transform3D(
            translation=odom_pose.translation, quaternion=odom_pose.rotation
        ),
    )

    if landmarks:
        lm_xyz = [lm.coords for lm in landmarks]
        lm_colors = [color_from_id(lm.id) for lm in landmarks]
        rr.log("rig/landmarks", rr.Points3D(lm_xyz, radii=0.02, colors=lm_colors))

    for i in range(len(cameras)):
        obs_uv = [[o.u, o.v] for o in observations[i]]
        obs_colors = [color_from_id(o.id) for o in observations[i]]
        rr.log(
            f"rig/cam{i}/image",
            rr.Image(images[i], color_model="BGR").compress(jpeg_quality=80),
        )
        rr.log(
            f"rig/cam{i}/observations",
            rr.Points2D(obs_uv, radii=5, colors=obs_colors),
        )


@beartype
def log_final_landmarks(tracker: cuvslam.Tracker) -> None:
    """Log final accumulated landmarks after tracking completes."""
    final_landmarks = tracker.get_final_landmarks()
    if final_landmarks:
        rr.log(
            "final_landmarks",
            rr.Points3D(list(final_landmarks.values()), radii=0.01),
        )
