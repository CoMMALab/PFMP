"""
Anchor a previously-calibrated extrinsics file to the robot/world frame.

Inputs:
  - extrinsics.npz produced by calibrate.py (UNANCHORED: world = some camera).
  - One synchronized capture of the table AprilTag at its known pose T_ARUCO
    (defined in camera.py). The tag must be visible to >=1 camera.

Output:
  - wf.npz with t_camN expressed in robot/world frame and t_aruco = T_ARUCO.

Algorithm:
  1. Localize the table tag in the OLD world (= ref-cam) frame via multi-view PnP
     across whichever cameras detected it. Camera intrinsics + (unanchored)
     extrinsics are fixed.
  2. Compute the rigid alignment that takes the OLD world into robot/world:
         T_oldworld_to_robot = T_ARUCO @ inv(T_table_in_oldworld)
  3. Re-express every camera extrinsic in robot/world:
         t_camN_in_robot = T_oldworld_to_robot @ extr.t_camN
"""

import time
import argparse
import numpy as np
from camera import *
from viz import print_world_coordinates, plot_worldframe

import matplotlib
matplotlib.use('TkAgg')

# Camera label -> WorldFrame slot index (cam1=top, cam2=corner, cam3=side).
CAM_TO_T = {
    "top":    lambda wf: wf.t_cam1,
    "corner": lambda wf: wf.t_cam2,
    "side":   lambda wf: wf.t_cam3,
}


def _detect_table_tag_observations(
    cams: list[str],
    pipelines: dict,
    intrinsics: dict,
    table_tag_size_m: float,
):
    """Capture one frame per camera and collect (cam_name, (4,2) corners) pairs."""
    observations = []
    poses_per_cam = {}  # cam -> T_table_to_cam (for diagnostics)
    time.sleep(1.0)
    for cam in cams:
        K, dist = intrinsics[cam]
        color_image, _ = get_rgbd_frames(pipelines[cam], cam, retries=4, timeout_ms=10000)
        detections, _ = detect_apriltags(color_image, K, dist, table_tag_size_m)
        corners_px = get_tag_corners_by_id(detections, TABLE_TAG_ID)
        t_table_to_cam = get_tag_transform_by_id(detections, TABLE_TAG_ID)
        if corners_px is None or t_table_to_cam is None:
            print(f"[ANCHOR][WARN] {cam}: table tag id={TABLE_TAG_ID} not detected")
            continue
        observations.append((cam, corners_px))
        poses_per_cam[cam] = t_table_to_cam
        print(f"[ANCHOR] {cam}: detected table tag, t={t_table_to_cam[:3, 3]}")
    return observations, poses_per_cam


def anchor(args):
    extr = WorldFrame.load(args.input_path)
    if not np.allclose(extr.t_aruco, np.eye(4), atol=1e-8):
        raise RuntimeError(
            f"{args.input_path}: t_aruco is not identity. "
            "This file appears to already be anchored; refusing to anchor again. "
            "Re-run calibrate.py to produce a fresh extrinsics file."
        )
    print(f"[ANCHOR] loaded unanchored extrinsics from {args.input_path}")

    pipelines, intrinsics = init_cameras(args.cams)
    try:
        observations, poses_per_cam = _detect_table_tag_observations(
            args.cams, pipelines, intrinsics, args.table_tag_size_m
        )
    finally:
        for pipe in pipelines.values():
            pipe.stop()

    if not observations:
        raise RuntimeError(
            "Anchor failed: table tag not visible to any selected camera."
        )

    # Localize the table tag in the OLD (unanchored) world frame.
    if args.no_multiview or len(observations) == 1:
        # Single-view fallback: use the first camera that detected the tag.
        cam0, _ = observations[0]
        t_table_in_oldworld = CAM_TO_T[cam0](extr) @ poses_per_cam[cam0]
        if args.no_multiview:
            print(f"[ANCHOR] --no-multiview: using single-view PnP from '{cam0}'")
        else:
            print(f"[ANCHOR] only 1 camera saw the tag: using single-view PnP from '{cam0}'")
    else:
        print(f"[ANCHOR] running multi-view PnP for table tag across {len(observations)} cameras")
        t_table_in_oldworld = multi_view_tag_pnp(
            extr,
            observations,
            tag_size_m=args.table_tag_size_m,
            yaw_only=False,
            verbose=True,
        )
        if t_table_in_oldworld is None:
            raise RuntimeError("Multi-view PnP for table tag failed during anchoring.")

    print(
        "[ANCHOR] table-tag pose in OLD world frame: "
        f"t={t_table_in_oldworld[:3, 3]}"
    )

    # Compute the rigid alignment: OLD world -> robot/world.
    t_oldworld_to_robot = T_ARUCO @ np.linalg.inv(t_table_in_oldworld)
    det = float(np.linalg.det(t_oldworld_to_robot[:3, :3]))
    if not np.isclose(det, 1.0, atol=1e-4):
        raise RuntimeError(
            f"Anchor alignment is not a proper rotation (det={det:.6f}); "
            "check T_ARUCO and that the input extrinsics are well-formed."
        )

    # Re-express every camera extrinsic in robot/world.
    def rebase(t_cam_in_oldworld):
        return t_oldworld_to_robot @ t_cam_in_oldworld

    wf = WorldFrame(
        t_aruco=T_ARUCO,
        K_cam1=extr.K_cam1,
        K_cam2=extr.K_cam2,
        K_cam3=extr.K_cam3,
        t_cam1=rebase(extr.t_cam1),
        t_cam2=rebase(extr.t_cam2),
        t_cam3=rebase(extr.t_cam3),
        t_pose=extr.t_pose,
    )
    print_world_coordinates(wf)

    if not args.no_save:
        wf.save(args.output_path)
        print(f"[ANCHOR] wrote {args.output_path}")

    if not args.no_plot:
        plot_worldframe(wf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Anchor an unanchored extrinsics.npz to the robot/world frame using "
            "the table AprilTag at its known pose T_ARUCO (defined in camera.py)."
        )
    )
    parser.add_argument("--cams", nargs="+", default=["top", "corner", "side"])
    parser.add_argument("--table-tag-size-m", type=float, default=0.1651)
    parser.add_argument("--input-path", type=str, default="extrinsics.npz")
    parser.add_argument("--output-path", type=str, default="wf.npz")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--no-multiview",
        action="store_true",
        help="Use single-view PnP from the first camera that sees the tag, instead of multi-view PnP.",
    )
    args = parser.parse_args()
    anchor(args)
