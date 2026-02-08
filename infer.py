#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import click
import numpy as np
import torch

from regformer_model import regformer_model
from tools.euler_tools import quat2mat

SUPPORTED_EXTENSIONS = (".pcd", ".laz", ".las")


def make_inference_args():
    args = types.SimpleNamespace()
    args.batch_size = 1
    args.H_input = 64
    args.W_input = 1792
    args.is_training = False
    args.num_points = 120000
    args.limit_or_filter = True
    return args


def load_point_cloud(path):
    """Read a point cloud file and return (N, 3) float32 XYZ array.

    Supported formats (detected by extension): .pcd, .las, .laz
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pcd":
        from pypcd4 import PointCloud as PCD
        pc = PCD.from_path(path)
        points = pc.numpy(("x", "y", "z")).astype(np.float32)
    elif ext in (".laz", ".las"):
        import laspy
        las = laspy.read(path)
        points = np.stack([las.x, las.y, las.z], axis=-1).astype(np.float32)
    else:
        raise click.ClickException(
            f"Unsupported file format '{ext}'. Use: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    if points.shape[0] == 0:
        raise click.ClickException(f"Point cloud '{path}' contains no points.")
    return points


def load_model(checkpoint_path, device):
    args = make_inference_args()
    model = regformer_model(args, batch_size=1, H_input=64, W_input=1792, is_training=False)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    cleaned = {}
    for k, v in state_dict.items():
        new_key = k[len("module."):] if k.startswith("module.") else k
        cleaned[new_key] = v

    model.load_state_dict(cleaned)
    model.to(device)
    model.eval()
    return model


def qt_to_matrix(q, t):
    """Convert quaternion (w,x,y,z) and translation to 4x4 matrix."""
    R = quat2mat(q)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def run_inference(model, source_pts, target_pts, device, verbose=False):
    # Model expects (target, source) order — matching train.py:167 model(pos2, pos1, ...)
    # where pos2 is the cloud being moved to align with pos1
    pc_target = torch.from_numpy(target_pts).float().to(device)
    pc_source = torch.from_numpy(source_pts).float().to(device)

    input_f1 = [pc_target]  # cloud being moved
    input_f2 = [pc_source]  # reference cloud

    T_gt = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0)
    T_trans = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0)
    T_trans_inv = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_f1, input_f2, T_gt, T_trans, T_trans_inv)

    # outputs: l0_q, l0_t, l1_q, l1_t, l2_q, l2_t, l3_q, l3_t, ...
    stages = {}
    for i, name in enumerate(["l0", "l1", "l2", "l3"]):
        q = outputs[i * 2].cpu().numpy().reshape(4)
        t = outputs[i * 2 + 1].cpu().numpy().reshape(3)
        stages[name] = qt_to_matrix(q, t)

    if verbose:
        # Print coarsest → finest
        for name in ["l3", "l2", "l1", "l0"]:
            T = stages[name]
            rot_deg = np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1)))
            trans_m = np.linalg.norm(T[:3, 3])
            click.echo(f"\n  {name} ({'coarsest' if name == 'l3' else 'finest' if name == 'l0' else 'mid'}):"
                        f"  rot={rot_deg:.3f} deg  trans={trans_m:.4f} m")
            click.echo(f"    t = [{T[0,3]:+.4f}, {T[1,3]:+.4f}, {T[2,3]:+.4f}]")
            click.echo(format_matrix(T))

    return stages["l0"]


def format_matrix(T):
    lines = []
    for row in T:
        lines.append("  ".join(f"{v:12.6f}" for v in row))
    return "\n".join(lines)


@click.command()
@click.argument("source", type=click.Path(exists=True))
@click.argument("target", type=click.Path(exists=True))
@click.option("-c", "--checkpoint", type=click.Path(exists=True),
              default="/host/regformer_KITTI_194_-11.957191.pth.tar",
              show_default=True, help="Path to RegFormer checkpoint.")
@click.option("-g", "--gpu", type=int, default=0, show_default=True,
              help="CUDA GPU device index.")
@click.option("-o", "--output", type=click.Path(), default=None,
              help="Save 4x4 matrix to file. If omitted, print to stdout.")
@click.option("-v", "--verbose", is_flag=True, default=False,
              help="Print predictions at all 4 pyramid levels (l3=coarsest to l0=finest).")
def main(source, target, checkpoint, gpu, output, verbose):
    """Estimate relative transformation between two point clouds using RegFormer.

    SOURCE is the reference (fixed) point cloud (.pcd, .las, .laz).
    TARGET is the moving point cloud (.pcd, .las, .laz).

    Outputs a 4x4 transformation matrix T that transforms TARGET into SOURCE frame.

    Model expects X=forward, Y=left, Z=up (KITTI Velodyne convention).
    """
    if not torch.cuda.is_available():
        raise click.ClickException("CUDA is not available. RegFormer requires a GPU.")

    device = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(device)

    click.echo("Loading point clouds...")
    pc_source = load_point_cloud(source)
    pc_target = load_point_cloud(target)
    click.echo(f"  Source: {pc_source.shape[0]} points")
    click.echo(f"  Target: {pc_target.shape[0]} points")

    click.echo(f"Loading model from {checkpoint}...")
    model = load_model(checkpoint, device)

    click.echo("Running inference...")
    T = run_inference(model, pc_source, pc_target, device, verbose=verbose)

    if output:
        np.savetxt(output, T, fmt="%.6f")
        click.echo(f"\nTransformation saved to {output}")
    else:
        click.echo("\nRelative transformation (4x4):")
        click.echo(format_matrix(T))


if __name__ == "__main__":
    main()
