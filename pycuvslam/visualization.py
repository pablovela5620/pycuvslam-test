"""Shared Rerun visualization helpers for cuVSLAM tracking."""

from pathlib import Path

import cuvslam
import rerun as rr
from simplecv.rerun_log_utils import log_pinhole, log_video

from pycuvslam.data.base import BaseTrackDataset


def color_from_id(identifier: int) -> list[int]:
    """Generate pseudo-random colour from integer identifier for visualization."""
    return [
        (identifier * 17) % 256,
        (identifier * 31) % 256,
        (identifier * 47) % 256,
    ]


def log_static_cameras_and_videos(
    dataset: BaseTrackDataset,
    timeline: str = "video_time",
) -> None:
    """Log pinholes, extrinsics, and video assets (all static, called once).

    Args:
        dataset: Track dataset providing camera params and video paths.
        timeline: Timeline name for video frame timestamps.
    """
    for i, cam_name in enumerate(dataset.cam_names):
        cam_log_path: Path = Path(f"rig/cam{i}")

        # Log pinhole intrinsics + extrinsics (static)
        log_pinhole(
            dataset.cam_params[cam_name],
            cam_log_path,
            image_plane_distance=dataset.image_plane_distance,
            static=True,
        )

        # Log video asset + frame references (static)
        if cam_name in dataset.video_paths:
            log_video(
                dataset.video_paths[cam_name],
                cam_log_path / "pinhole" / "video",
                timeline=timeline,
            )


def log_frame_visuals(
    n_cameras: int,
    observations: list,
    landmarks: list,
    odom_pose: cuvslam.Pose,
) -> None:
    """Log per-frame rig pose, landmarks, and 2D observations.

    Args:
        n_cameras: Number of cameras in the rig.
        observations: Per-camera observation lists from tracker.
        landmarks: Landmark list from tracker.
        odom_pose: Current odometry pose estimate.
    """
    rr.log(
        "rig",
        rr.Transform3D(translation=odom_pose.translation, quaternion=odom_pose.rotation),
    )

    if landmarks:
        lm_xyz = [lm.coords for lm in landmarks]
        lm_colors = [color_from_id(lm.id) for lm in landmarks]
        rr.log("rig/landmarks", rr.Points3D(lm_xyz, radii=0.02, colors=lm_colors))

    for i in range(n_cameras):
        obs_uv = [[o.u, o.v] for o in observations[i]]
        obs_colors = [color_from_id(o.id) for o in observations[i]]
        rr.log(
            f"rig/cam{i}/pinhole/observations",
            rr.Points2D(obs_uv, radii=5, colors=obs_colors),
        )


def log_final_landmarks(tracker: cuvslam.Tracker) -> None:
    """Log final accumulated landmarks after tracking completes."""
    final_landmarks = tracker.get_final_landmarks()
    if final_landmarks:
        rr.log(
            "final_landmarks",
            rr.Points3D(list(final_landmarks.values()), radii=0.01),
        )
