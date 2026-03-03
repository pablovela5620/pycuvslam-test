#!/usr/bin/env python3
"""Run PyCuVSLAM multicamera visual odometry on Robocap headset data.

Loads Robocap headset sequences (Kalibr calibrations + synchronized MP4 videos),
converts to PyCuVSLAM format, runs multicamera odometry, and visualizes in Rerun.

Usage:
    pixi run track-robocap
    pixi run python examples/robocap/track_robocap.py --root /mnt/8tb/data/robocap
"""

import argparse

import cuvslam
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Float32, UInt8

from robocap_utils import (
    add_common_args,
    beartype,
    log_final_landmarks,
    log_frame_visuals,
    log_static_camera_frustums,
    setup_cameras_and_reader,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@beartype
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PyCuVSLAM multicamera visual odometry on Robocap data",
    )
    add_common_args(parser)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@beartype
def main() -> None:
    args = parse_args()
    setup = setup_cameras_and_reader(args)

    # --- Set up tracker (odometry only) ---
    rig = cuvslam.Rig()
    rig.cameras = setup.cameras

    tracker_cfg = cuvslam.Tracker.OdometryConfig(
        enable_observations_export=True,
        enable_final_landmarks_export=True,
        horizontal_stereo_camera=False,
        odometry_mode=cuvslam.Tracker.OdometryMode.Multicamera,
    )

    tracker = cuvslam.Tracker(rig, tracker_cfg)
    print(f"Tracker initialized (mode={tracker_cfg.odometry_mode}, slam=disabled)")

    # --- Set up Rerun ---
    rr.init("robocap_odometry", spawn=True)
    cam_views = [
        rrb.Spatial2DView(origin=f"rig/cam{i}", name=setup.cam_order[i])
        for i in range(len(setup.cameras))
    ]
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.TimePanel(state="collapsed"),
            rrb.Vertical(
                contents=[
                    rrb.Horizontal(contents=cam_views),
                    rrb.Spatial3DView(
                        name="3D",
                        defaults=[rr.components.ImagePlaneDistance(0.025)],
                    ),
                ]
            ),
        )
    )
    rr.log("/", rr.ViewCoordinates.LFD, static=True)
    log_static_camera_frustums(setup.cameras)

    # --- Tracking loop ---
    trajectory: list[Float32[np.ndarray, "3"]] = []

    for frame_idx in range(setup.reader.n_frames):
        ok, frames = setup.reader.read()
        if not ok:
            break

        images: list[UInt8[np.ndarray, "h w 3"]] = [
            frames[name] for name in setup.cam_order
        ]

        timestamp_ns: int = frame_idx * setup.frame_dt_ns
        track_result: tuple[cuvslam.PoseEstimate, cuvslam.Pose | None] = tracker.track(
            timestamp_ns, images
        )
        odom_pose_estimate: cuvslam.PoseEstimate = track_result[0]

        if odom_pose_estimate.world_from_rig is None:
            print(f"Warning: Failed to track frame {frame_idx}")
            continue

        odom_pose: cuvslam.Pose = odom_pose_estimate.world_from_rig.pose
        observations = [
            tracker.get_last_observations(i) for i in range(len(setup.cameras))
        ]
        landmarks = tracker.get_last_landmarks()

        trajectory.append(odom_pose.translation)

        # --- Rerun visualization ---
        rr.set_time_sequence("frame", frame_idx)

        # Re-log trajectory every 10 frames to avoid O(n²) data transmission
        is_batch_frame: bool = (
            frame_idx % 10 == 0 or frame_idx == setup.reader.n_frames - 1
        )
        if is_batch_frame:
            rr.log(
                "trajectory",
                rr.LineStrips3D(trajectory, colors=[[0, 200, 255]]),
            )

        log_frame_visuals(setup.cameras, images, observations, landmarks, odom_pose)

        if frame_idx % 100 == 0:
            print(
                f"Frame {frame_idx}/{setup.reader.n_frames} — "
                f"tracked {len(trajectory)} frames"
            )

    setup.reader.release()
    log_final_landmarks(tracker)

    print(f"\nDone. Tracked {len(trajectory)}/{setup.reader.n_frames} frames.")


if __name__ == "__main__":
    main()
