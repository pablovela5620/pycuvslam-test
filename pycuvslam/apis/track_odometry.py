"""Multicamera visual odometry API using cuVSLAM."""

from dataclasses import dataclass

import cuvslam
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Float32, UInt8
from simplecv.rerun_log_utils import RerunTyroConfig
from tqdm.auto import tqdm

from pycuvslam.configs.track_dataset_configs import AnnotatedTrackDatasetUnion
from pycuvslam.data.base import BaseTrackDataset
from pycuvslam.visualization import log_final_landmarks, log_frame_visuals, log_static_cameras_and_videos


@dataclass
class TrackOdometryConfig:
    """Configuration for multicamera visual odometry."""

    rr_config: RerunTyroConfig
    """Rerun viewer options (spawn, save, serve, headless)."""
    dataset: AnnotatedTrackDatasetUnion
    """Dataset to track (e.g., robocap)."""


def main(config: TrackOdometryConfig) -> None:
    """Run multicamera visual odometry.

    Args:
        config: Odometry configuration with dataset and Rerun settings.
    """
    dataset: BaseTrackDataset = config.dataset.setup()

    # Set up cuVSLAM tracker (odometry only)
    rig = cuvslam.Rig()
    rig.cameras = dataset.cameras

    tracker_cfg = cuvslam.Tracker.OdometryConfig(
        enable_observations_export=True,
        enable_final_landmarks_export=True,
        horizontal_stereo_camera=False,
        odometry_mode=cuvslam.Tracker.OdometryMode.Multicamera,
    )

    tracker: cuvslam.Tracker = cuvslam.Tracker(rig, tracker_cfg)
    print(f"Tracker initialized (mode={tracker_cfg.odometry_mode}, slam=disabled)")

    # Set up Rerun blueprint
    cam_views = [
        rrb.Spatial2DView(origin=f"rig/cam{i}/pinhole", name=name)
        for i, name in enumerate(dataset.cam_names)
    ]
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.TimePanel(state="collapsed"),
            rrb.Vertical(
                contents=[
                    rrb.Horizontal(contents=cam_views),
                    rrb.Spatial3DView(name="3D"),
                ]
            ),
        )
    )
    rr.log("/", rr.ViewCoordinates.LFD, static=True)
    log_static_cameras_and_videos(dataset, timeline="video_time")

    # Tracking loop
    trajectory: list[Float32[np.ndarray, "3"]] = []
    n_cameras: int = len(dataset.cameras)

    for frame_idx in tqdm(range(dataset.n_frames), desc="Tracking"):
        images: list[UInt8[np.ndarray, "h w 3"]] = dataset.get_frame(frame_idx)
        timestamp_ns: int = int(dataset.video_timestamps_ns[frame_idx])

        rr.set_time("video_time", duration=timestamp_ns / 1e9)
        rr.set_time("frame", sequence=frame_idx)

        track_result: tuple[cuvslam.PoseEstimate, cuvslam.Pose | None] = tracker.track(timestamp_ns, images)
        odom_pose_estimate: cuvslam.PoseEstimate = track_result[0]

        if odom_pose_estimate.world_from_rig is None:
            tqdm.write(f"Warning: Failed to track frame {frame_idx}")
            continue

        odom_pose: cuvslam.Pose = odom_pose_estimate.world_from_rig.pose
        observations = [tracker.get_last_observations(i) for i in range(n_cameras)]
        landmarks = tracker.get_last_landmarks()

        trajectory.append(odom_pose.translation)

        # Re-log trajectory every 10 frames to avoid O(n^2) data transmission
        is_batch_frame: bool = frame_idx % 10 == 0 or frame_idx == dataset.n_frames - 1
        if is_batch_frame:
            rr.log("trajectory", rr.LineStrips3D(trajectory, colors=[[0, 200, 255]]))

        log_frame_visuals(n_cameras, observations, landmarks, odom_pose)

    log_final_landmarks(tracker)
    print(f"\nDone. Tracked {len(trajectory)}/{dataset.n_frames} frames.")


def entrypoint() -> None:
    """CLI entrypoint for multicamera visual odometry."""
    import tyro

    tyro.extras.set_accent_color("bright_cyan")
    main(tyro.cli(TrackOdometryConfig, description="Run multicamera visual odometry."))
