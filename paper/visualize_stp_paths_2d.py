#!/usr/bin/env python3
"""
Visualize 2D diagonal-path (DP) and stochastic-threshold-path (STP) trajectories.

This script supports two modes:

1. Target-point mode (recommended for the theorem intuition):
   Specify a 2D interior point (mu1, mu2) in gate space. The script then
   constructs a family of valid STP paths that pass through that point using
   the proof construction in `paper/pathSampling.tex`:

       tau_star in (max(mu1, mu2), 1)
       u_i = (tau_star - mu_i) / (1 - mu_i)

2. Threshold mode:
   Specify (u1, u2) directly and visualize the corresponding single STP path.

Example:
    python paper/visualize_stp_paths_2d.py --mu1 0.2 --mu2 0.7
    python paper/visualize_stp_paths_2d.py --mu1 0.2 --mu2 0.7 --num-paths 7
    python paper/visualize_stp_paths_2d.py --u1 0.1 --u2 0.6 --out paper/stp_u_demo.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize 2D STP/DP integration paths."
    )
    parser.add_argument(
        "--mu1",
        type=float,
        default=0.25,
        help="Target point first coordinate in gate space (0, 1).",
    )
    parser.add_argument(
        "--mu2",
        type=float,
        default=0.70,
        help="Target point second coordinate in gate space (0, 1).",
    )
    parser.add_argument(
        "--u1",
        type=float,
        default=None,
        help="Directly specify the first STP threshold instead of target-point mode.",
    )
    parser.add_argument(
        "--u2",
        type=float,
        default=None,
        help="Directly specify the second STP threshold instead of target-point mode.",
    )
    parser.add_argument(
        "--tau-star",
        type=float,
        default=None,
        help="Use a specific tau* when constructing a target-point STP path.",
    )
    parser.add_argument(
        "--num-paths",
        type=int,
        default=5,
        help="How many valid STP paths to draw in target-point mode.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=400,
        help="How many tau samples to use per curve.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output image path. Defaults to paper/stp_paths_2d_mu_<mu1>_<mu2>.png.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also open the figure window after saving.",
    )
    return parser.parse_args()


def validate_open_unit_interval(name: str, value: float) -> None:
    if not (0.0 < value < 1.0):
        raise ValueError(f"{name} must be in (0, 1), got {value}.")


def validate_half_open_unit_interval(name: str, value: float) -> None:
    if not (0.0 <= value < 1.0):
        raise ValueError(f"{name} must be in [0, 1), got {value}.")


def alpha_stp(tau: np.ndarray, u: float) -> np.ndarray:
    if u >= 1.0:
        return np.zeros_like(tau)
    return np.clip((tau - u) / (1.0 - u), 0.0, 1.0)


def construct_u_from_target(mu: float, tau_star: float) -> float:
    return (tau_star - mu) / (1.0 - mu)


def choose_tau_values(mu1: float, mu2: float, num_paths: int) -> np.ndarray:
    lower = max(mu1, mu2) + 1e-3
    upper = 1.0 - 1e-3
    if lower >= upper:
        raise ValueError(
            "Target point is too close to the boundary; no stable tau* interval remains."
        )
    if num_paths <= 1:
        return np.array([(lower + upper) / 2.0], dtype=float)
    return np.linspace(lower, upper, num_paths, dtype=float)


def format_float(x: float) -> str:
    return f"{x:.3f}".rstrip("0").rstrip(".")


def build_paths(args: argparse.Namespace) -> tuple[list[dict], dict]:
    tau = np.linspace(0.0, 1.0, args.num_samples)

    if (args.u1 is None) != (args.u2 is None):
        raise ValueError("Please provide both --u1 and --u2, or neither of them.")

    if args.u1 is not None and args.u2 is not None:
        validate_half_open_unit_interval("u1", args.u1)
        validate_half_open_unit_interval("u2", args.u2)
        path = {
            "label": f"STP(u1={format_float(args.u1)}, u2={format_float(args.u2)})",
            "u1": args.u1,
            "u2": args.u2,
            "tau_star": None,
            "alpha1": alpha_stp(tau, args.u1),
            "alpha2": alpha_stp(tau, args.u2),
            "tau": tau,
        }
        meta = {"mode": "threshold", "target": None, "highlight": path}
        return [path], meta

    validate_open_unit_interval("mu1", args.mu1)
    validate_open_unit_interval("mu2", args.mu2)

    if args.tau_star is not None:
        tau_values = np.array([args.tau_star], dtype=float)
    else:
        tau_values = choose_tau_values(args.mu1, args.mu2, args.num_paths)

    paths = []
    for tau_star in tau_values:
        if not (max(args.mu1, args.mu2) < tau_star < 1.0):
            raise ValueError(
                f"tau_star must satisfy max(mu1, mu2) < tau_star < 1. Got {tau_star}."
            )
        u1 = construct_u_from_target(args.mu1, tau_star)
        u2 = construct_u_from_target(args.mu2, tau_star)
        validate_half_open_unit_interval("u1", u1)
        validate_half_open_unit_interval("u2", u2)
        paths.append(
            {
                "label": (
                    f"tau*={format_float(tau_star)}, "
                    f"u1={format_float(u1)}, u2={format_float(u2)}"
                ),
                "u1": u1,
                "u2": u2,
                "tau_star": tau_star,
                "alpha1": alpha_stp(tau, u1),
                "alpha2": alpha_stp(tau, u2),
                "tau": tau,
            }
        )

    highlight = paths[len(paths) // 2]
    meta = {
        "mode": "target",
        "target": (args.mu1, args.mu2),
        "highlight": highlight,
    }
    return paths, meta


def default_output_path(args: argparse.Namespace) -> Path:
    if args.out:
        return Path(args.out)
    if args.u1 is not None:
        return Path("paper") / (
            f"stp_paths_2d_u1_{format_float(args.u1)}_u2_{format_float(args.u2)}.png"
        )
    return Path("paper") / (
        f"stp_paths_2d_mu1_{format_float(args.mu1)}_mu2_{format_float(args.mu2)}.png"
    )


def plot_paths(paths: list[dict], meta: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax_space, ax_schedule = axes

    tau = np.linspace(0.0, 1.0, 400)
    ax_space.plot(tau, tau, linestyle="--", color="black", linewidth=2.0, label="DP")

    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(paths)))
    for idx, (path, color) in enumerate(zip(paths, colors)):
        lw = 3.0 if path is meta["highlight"] else 1.6
        alpha = 1.0 if path is meta["highlight"] else 0.75
        label = "STP family" if idx == 0 and meta["mode"] == "target" else path["label"]
        ax_space.plot(
            path["alpha1"],
            path["alpha2"],
            color=color,
            linewidth=lw,
            alpha=alpha,
            label=label,
        )

    if meta["target"] is not None:
        mu1, mu2 = meta["target"]
        ax_space.scatter(
            [mu1],
            [mu2],
            color="crimson",
            s=80,
            marker="*",
            zorder=5,
            label=f"target ({format_float(mu1)}, {format_float(mu2)})",
        )
        ax_space.annotate(
            r"$(\mu_1,\mu_2)$",
            xy=(mu1, mu2),
            xytext=(8, 8),
            textcoords="offset points",
            color="crimson",
            fontsize=11,
        )

    ax_space.set_xlim(0.0, 1.0)
    ax_space.set_ylim(0.0, 1.0)
    ax_space.set_aspect("equal")
    ax_space.set_xlabel(r"$\alpha_1$")
    ax_space.set_ylabel(r"$\alpha_2$")
    ax_space.set_title("Paths in 2D gate space")
    ax_space.grid(True, alpha=0.25)
    ax_space.legend(loc="lower right", fontsize=9)

    highlight = meta["highlight"]
    ax_schedule.plot(
        highlight["tau"],
        highlight["alpha1"],
        linewidth=2.5,
        color="#1f77b4",
        label=rf"$\alpha_1(\tau)$, $u_1={format_float(highlight['u1'])}$",
    )
    ax_schedule.plot(
        highlight["tau"],
        highlight["alpha2"],
        linewidth=2.5,
        color="#ff7f0e",
        label=rf"$\alpha_2(\tau)$, $u_2={format_float(highlight['u2'])}$",
    )
    ax_schedule.plot(
        tau,
        tau,
        linestyle="--",
        color="black",
        linewidth=1.8,
        label="DP schedule",
    )
    ax_schedule.axvline(
        highlight["u1"],
        linestyle=":",
        color="#1f77b4",
        linewidth=1.4,
        alpha=0.8,
    )
    ax_schedule.axvline(
        highlight["u2"],
        linestyle=":",
        color="#ff7f0e",
        linewidth=1.4,
        alpha=0.8,
    )
    if highlight["tau_star"] is not None:
        ax_schedule.axvline(
            highlight["tau_star"],
            linestyle="-.",
            color="crimson",
            linewidth=1.6,
            alpha=0.9,
            label=rf"$\tau^*={format_float(highlight['tau_star'])}$",
        )

    ax_schedule.set_xlim(0.0, 1.0)
    ax_schedule.set_ylim(0.0, 1.02)
    ax_schedule.set_xlabel(r"$\tau$")
    ax_schedule.set_ylabel(r"$\alpha(\tau)$")
    ax_schedule.set_title("Highlighted path schedule")
    ax_schedule.grid(True, alpha=0.25)
    ax_schedule.legend(loc="lower right", fontsize=9)

    if meta["mode"] == "target":
        mu1, mu2 = meta["target"]
        fig.suptitle(
            (
                "2D STP paths through a specified interior point "
                rf"$(\mu_1,\mu_2)=({format_float(mu1)}, {format_float(mu2)})$"
            ),
            fontsize=14,
        )
    else:
        fig.suptitle("2D STP path under directly specified thresholds", fontsize=14)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")


def main() -> None:
    args = parse_args()
    paths, meta = build_paths(args)
    out_path = default_output_path(args)
    plot_paths(paths, meta, out_path)

    print(f"Saved figure to: {out_path}")
    if meta["mode"] == "target":
        mu1, mu2 = meta["target"]
        print(f"Target point: mu1={mu1:.6f}, mu2={mu2:.6f}")
        print("Constructed valid STP paths through this point:")
        for path in paths:
            print(
                "  "
                f"tau*={path['tau_star']:.6f}, "
                f"u1={path['u1']:.6f}, "
                f"u2={path['u2']:.6f}"
            )
    else:
        highlight = meta["highlight"]
        print(
            "Direct threshold mode: "
            f"u1={highlight['u1']:.6f}, u2={highlight['u2']:.6f}"
        )

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
