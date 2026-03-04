"""Multicamera visual SLAM API using cuVSLAM."""

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
from pycuvslam.visualization import log_final_landmarks, log_frame_visuals, log_rig_mesh, log_static_cameras_and_videos


@dataclass
class TrackSlamConfig:
    """Configuration for multicamera visual SLAM."""

    rr_config: RerunTyroConfig
    """Rerun viewer options (spawn, save, serve, headless)."""
    dataset: AnnotatedTrackDatasetUnion
    """Dataset to track (e.g., robocap)."""
    slam_sync_mode: bool = False
    """Run SLAM in synchronous mode (slower but deterministic)."""


def main(config: TrackSlamConfig) -> None:
    """Run multicamera visual SLAM.

    Args:
        config: SLAM configuration with dataset, Rerun settings, and SLAM options.
    """
    dataset: BaseTrackDataset = config.dataset.setup()

    # Set up cuVSLAM tracker (with SLAM)
    rig = cuvslam.Rig()
    rig.cameras = dataset.cameras

    tracker_cfg = cuvslam.Tracker.OdometryConfig(
        enable_observations_export=True,
        enable_final_landmarks_export=True,
        horizontal_stereo_camera=False,
        odometry_mode=cuvslam.Tracker.OdometryMode.Multicamera,
    )

    slam_cfg = cuvslam.Tracker.SlamConfig(sync_mode=config.slam_sync_mode)
    tracker: cuvslam.Tracker = cuvslam.Tracker(rig, tracker_cfg, slam_cfg)
    mode_str: str = "sync" if config.slam_sync_mode else "async"
    print(f"Tracker initialized (mode={tracker_cfg.odometry_mode}, slam=enabled ({mode_str}))")

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
                    rrb.TextLogView(name="SLAM Metrics", origin="slam_metrics"),
                ]
            ),
        )
    )
    rr.log("/", rr.ViewCoordinates.LFD, static=True)
    log_static_cameras_and_videos(dataset, timeline="video_time")
    log_rig_mesh(dataset.mesh_path)

    # Tracking loop
    odom_trajectory: list[Float32[np.ndarray, "3"]] = []
    slam_trajectory: list[Float32[np.ndarray, "3"]] = []
    loop_closure_points: list[Float32[np.ndarray, "3"]] = []
    prev_lc_count: int = 0
    n_cameras: int = len(dataset.cameras)

    for frame_idx in tqdm(range(dataset.n_frames), desc="SLAM Tracking"):
        images: list[UInt8[np.ndarray, "h w 3"]] = dataset.get_frame(frame_idx)
        timestamp_ns: int = int(dataset.video_timestamps_ns[frame_idx])

        rr.set_time("video_time", duration=timestamp_ns / 1e9)
        rr.set_time("frame", sequence=frame_idx)

        track_result: tuple[cuvslam.PoseEstimate, cuvslam.Pose | None] = tracker.track(timestamp_ns, images)
        odom_pose_estimate: cuvslam.PoseEstimate = track_result[0]
        slam_pose: cuvslam.Pose | None = track_result[1]

        if odom_pose_estimate.world_from_rig is None:
            tqdm.write(f"Warning: Failed to track frame {frame_idx}")
            continue

        odom_pose: cuvslam.Pose = odom_pose_estimate.world_from_rig.pose
        observations = [tracker.get_last_observations(i) for i in range(n_cameras)]
        landmarks = tracker.get_last_landmarks()

        # Collect trajectories
        odom_trajectory.append(odom_pose.translation)
        if slam_pose is not None:
            slam_trajectory.append(slam_pose.translation)

        # Loop closure detection
        current_lc_poses = tracker.get_loop_closure_poses()
        if current_lc_poses is not None and len(current_lc_poses) > prev_lc_count:
            new_count: int = len(current_lc_poses) - prev_lc_count
            for lc_pose in current_lc_poses[-new_count:]:
                loop_closure_points.append(lc_pose.pose.translation)
            prev_lc_count = len(current_lc_poses)

        # Re-log trajectories every 10 frames to avoid O(n^2) data transmission
        is_batch_frame: bool = frame_idx % 10 == 0 or frame_idx == dataset.n_frames - 1
        if is_batch_frame:
            rr.log("odom_trajectory", rr.LineStrips3D(odom_trajectory, colors=[[0, 200, 255]]))
            if slam_trajectory:
                rr.log("slam_trajectory", rr.LineStrips3D(slam_trajectory, colors=[[0, 255, 0]]))
            if loop_closure_points:
                rr.log("loop_closures", rr.Points3D(loop_closure_points, radii=0.05, colors=[[255, 0, 0]]))

            # Pose graph visualization
            pose_graph = tracker.get_pose_graph()
            if pose_graph is not None and pose_graph.nodes:
                node_positions = [n.node_pose.translation for n in pose_graph.nodes]
                rr.log("pose_graph/nodes", rr.Points3D(node_positions, radii=0.02, colors=[[255, 255, 0]]))
                if pose_graph.edges:
                    node_by_id: dict[int, cuvslam.Pose] = {n.id: n.node_pose for n in pose_graph.nodes}
                    edge_segments: list[list[Float32[np.ndarray, "3"]]] = []
                    for edge in pose_graph.edges:
                        if edge.node_from in node_by_id and edge.node_to in node_by_id:
                            edge_segments.append([
                                node_by_id[edge.node_from].translation,
                                node_by_id[edge.node_to].translation,
                            ])
                    if edge_segments:
                        rr.log("pose_graph/edges", rr.LineStrips3D(edge_segments, colors=[[180, 180, 180]]))

        log_frame_visuals(n_cameras, observations, landmarks, odom_pose)

        # SLAM metrics (every 100 frames)
        if frame_idx % 100 == 0:
            slam_metrics = tracker.get_slam_metrics()
            if slam_metrics is not None:
                rr.log(
                    "slam_metrics",
                    rr.TextLog(
                        f"frame={frame_idx} "
                        f"lc_status={slam_metrics.lc_status} "
                        f"pgo_status={slam_metrics.pgo_status} "
                        f"lc_selected={slam_metrics.lc_selected_landmarks_count} "
                        f"lc_tracked={slam_metrics.lc_tracked_landmarks_count} "
                        f"lc_pnp={slam_metrics.lc_pnp_landmarks_count} "
                        f"lc_good={slam_metrics.lc_good_landmarks_count}"
                    ),
                )

    log_final_landmarks(tracker)
    print(
        f"\nDone. Tracked {len(odom_trajectory)}/{dataset.n_frames} frames, "
        f"slam_poses={len(slam_trajectory)}, "
        f"loop_closures={len(loop_closure_points)}."
    )


def entrypoint() -> None:
    """CLI entrypoint for multicamera visual SLAM."""
    import tyro

    tyro.extras.set_accent_color("bright_cyan")
    main(tyro.cli(TrackSlamConfig, description="Run multicamera visual SLAM."))
