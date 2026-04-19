"""Compute Gazebo LiftDrag plugin parameters from a Flow5 polar and plane.xml.

Fits pre-stall and post-stall aerodynamic slopes from the polar data and
produces a ready-to-paste SDF plugin snippet.

Required plugin parameters computed here:
  a0             — zero-lift angle of attack [rad]
  cla            — pre-stall dCL/dα [per rad]
  cda            — zero-lift drag coefficient CD₀ (at α = a0)
  cma            — pre-stall dCm/dα [per rad]
  alpha_stall    — stall angle [rad]
  cl_at_alpha_stall, cd_at_alpha_stall, cm_at_alpha_stall  — values at stall
  cla_stall      — post-stall dCL/dα [per rad]
  cda_stall      — post-stall dCD/dα [per rad]
  cma_stall      — post-stall dCm/dα [per rad]
  cp             — centre of pressure in link frame [m] (requires --span)
  forward/upward — orientation vectors in link frame

Parameters NOT in plane.xml that must be supplied by the user:
  --span         wingspan [m]  (required for AR, cp lateral component)
  --link-name    SDF link name this plugin will attach to
  --cp-offset    (x, y, z) offset from link origin to aerodynamic centre [m]
                 Defaults to (x_ac, 0, 0) from plane.xml if omitted.

Usage:
    py -3 gazebo_ald_params.py polars.txt --plane plane.xml \\
        --span 0.9 --link-name wing_link
    py -3 gazebo_ald_params.py polars.txt --span 0.9 --plot --show
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from plot_flow5 import load_plane_xml, load_polar


# ---------------------------------------------------------------------------
# Fitting helpers
# ---------------------------------------------------------------------------


def _linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Return (slope, intercept, R²) of a least-squares linear fit y = slope*x + intercept."""
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(slope), float(intercept), r2


def _prestall_mask(alpha_sorted: np.ndarray, cl_sorted: np.ndarray,
                   fit_alpha_min: float | None,
                   fit_alpha_max: float | None) -> np.ndarray:
    """Boolean mask selecting the pre-stall, optionally range-limited, data."""
    i_stall = int(np.argmax(cl_sorted))
    mask = np.zeros(len(alpha_sorted), dtype=bool)
    mask[: i_stall + 1] = True
    if fit_alpha_min is not None:
        mask &= alpha_sorted >= fit_alpha_min
    if fit_alpha_max is not None:
        mask &= alpha_sorted <= fit_alpha_max
    return mask


# ---------------------------------------------------------------------------
# Core parameter extraction
# ---------------------------------------------------------------------------


@dataclass
class AldParams:
    # Pre-stall
    a0_rad: float           # zero-lift AoA [rad]
    a0_deg: float           # zero-lift AoA [deg]
    cla_per_rad: float      # dCL/dα pre-stall [per rad]
    cla_per_deg: float      # dCL/dα pre-stall [per deg]  (informational)
    cla_r2: float           # goodness of fit
    cda: float              # CD at zero lift (interpolated at a0)
    cma_per_rad: float      # dCm/dα pre-stall [per rad]
    cma_per_deg: float
    cma_r2: float
    cm0: float              # Cm intercept (at α = 0)
    # Stall point
    alpha_stall_rad: float
    alpha_stall_deg: float
    cl_at_stall: float
    cd_at_stall: float
    cm_at_stall: float
    # Post-stall slopes  (per rad)
    cla_stall: float
    cda_stall: float
    cma_stall: float
    n_poststall: int        # number of post-stall data points used
    # Geometry
    area: float             # reference area [m²]
    span: float | None      # wingspan [m]; None if not provided
    chord: float | None     # mean chord [m]
    ar: float | None        # aspect ratio b²/S
    # Centre of pressure in link frame
    cp: tuple[float, float, float]
    # Orientation vectors
    forward: tuple[float, float, float]
    upward: tuple[float, float, float]
    # Link name
    link_name: str
    # Diagnostics
    warnings: list[str] = field(default_factory=list)


def compute_ald_params(
    polar: dict,
    plane: dict,
    span: float | None = None,
    link_name: str = "base_link",
    cp_offset: tuple[float, float, float] | None = None,
    forward: tuple[float, float, float] = (1.0, 0.0, 0.0),
    upward: tuple[float, float, float] = (0.0, 0.0, 1.0),
    fit_alpha_min: float | None = None,
    fit_alpha_max: float | None = None,
    n_poststall: int = 5,
) -> AldParams:
    """Extract all Gazebo LiftDrag plugin parameters from polar + plane data."""
    warnings: list[str] = []

    alpha_raw = np.asarray(polar["alpha"], dtype=float)
    cl_raw = np.asarray(polar["cl"], dtype=float)
    cd_raw = np.asarray(polar["cd"], dtype=float)
    cm_raw = np.asarray(polar["cm"], dtype=float)

    # Sort by alpha
    order = np.argsort(alpha_raw)
    alpha = alpha_raw[order]
    cl = cl_raw[order]
    cd = cd_raw[order]
    cm = cm_raw[order]

    # ------------------------------------------------------------------
    # Stall point
    # ------------------------------------------------------------------
    i_stall = int(np.argmax(cl))
    alpha_stall_deg = float(alpha[i_stall])
    alpha_stall_rad = np.radians(alpha_stall_deg)
    cl_stall = float(cl[i_stall])
    cd_stall = float(cd[i_stall])
    cm_stall = float(cm[i_stall])

    # ------------------------------------------------------------------
    # Pre-stall fit
    # ------------------------------------------------------------------
    mask = _prestall_mask(alpha, cl, fit_alpha_min, fit_alpha_max)
    if mask.sum() < 2:
        warnings.append("Fewer than 2 pre-stall points in fit range — using all pre-stall data.")
        mask = np.zeros(len(alpha), dtype=bool)
        mask[: i_stall + 1] = True

    alpha_pre = alpha[mask]
    cl_pre = cl[mask]
    cd_pre = cd[mask]
    cm_pre = cm[mask]

    # CL slope [per deg]
    cla_deg, cl0, cla_r2 = _linear_fit(alpha_pre, cl_pre)
    if cla_r2 < 0.98:
        warnings.append(
            f"Pre-stall CL fit R²={cla_r2:.4f} < 0.98 — consider narrowing "
            f"--fit-alpha-min / --fit-alpha-max to the linear region."
        )
    cla_per_deg = cla_deg
    cla_per_rad = cla_deg * (180.0 / np.pi)

    # Zero-lift angle: CL = 0 → a0 = -cl0 / cla_deg
    a0_deg = -cl0 / cla_deg if cla_deg != 0.0 else 0.0
    a0_rad = np.radians(a0_deg)

    # CD at zero lift (interpolate from data at a0_deg)
    cda = float(np.interp(a0_deg, alpha, cd))

    # Cm slope [per deg]
    cma_deg, cm0_int, cma_r2 = _linear_fit(alpha_pre, cm_pre)
    cma_per_deg = cma_deg
    cma_per_rad = cma_deg * (180.0 / np.pi)
    # cm0 at α = 0
    cm0 = float(np.interp(0.0, alpha_pre, cm_pre)) if alpha_pre[0] <= 0.0 <= alpha_pre[-1] else cm0_int

    if cma_r2 < 0.95:
        warnings.append(f"Pre-stall Cm fit R²={cma_r2:.4f} < 0.95.")

    # ------------------------------------------------------------------
    # Post-stall slopes
    # ------------------------------------------------------------------
    post_idx = np.where(alpha > alpha_stall_deg)[0]
    n_post = min(n_poststall, len(post_idx))

    if n_post >= 2:
        alpha_post = alpha[post_idx[:n_post]]
        cl_post = cl[post_idx[:n_post]]
        cd_post = cd[post_idx[:n_post]]
        cm_post = cm[post_idx[:n_post]]
        cla_stall_deg, _, _ = _linear_fit(alpha_post, cl_post)
        cda_stall_deg, _, _ = _linear_fit(alpha_post, cd_post)
        cma_stall_deg, _, _ = _linear_fit(alpha_post, cm_post)
        cla_stall = cla_stall_deg * (180.0 / np.pi)
        cda_stall = cda_stall_deg * (180.0 / np.pi)
        cma_stall = cma_stall_deg * (180.0 / np.pi)
    elif n_post == 1:
        warnings.append("Only 1 post-stall data point — post-stall slopes set to 0.")
        cla_stall = cda_stall = cma_stall = 0.0
    else:
        warnings.append(
            "No post-stall data in polar — post-stall slopes set to 0. "
            "Extend the polar to higher alpha for accurate stall modelling."
        )
        cla_stall = cda_stall = cma_stall = 0.0

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    area = plane.get("area")
    chord = plane.get("chord")
    if area is None:
        warnings.append("area not found in plane.xml — cannot compute AR or SDF area.")
    ar = (span ** 2 / area) if (span is not None and area is not None) else None

    # Centre of pressure
    if cp_offset is not None:
        cp = cp_offset
    else:
        x_ac = plane.get("ac")
        if x_ac is None:
            warnings.append(
                "ac (aerodynamic centre) not in plane.xml — cp set to (0, 0, 0). "
                "Provide --cp-offset x y z."
            )
            x_ac = 0.0
        cp = (x_ac, 0.0, 0.0)

    return AldParams(
        a0_rad=a0_rad,
        a0_deg=a0_deg,
        cla_per_rad=cla_per_rad,
        cla_per_deg=cla_per_deg,
        cla_r2=cla_r2,
        cda=cda,
        cma_per_rad=cma_per_rad,
        cma_per_deg=cma_per_deg,
        cma_r2=cma_r2,
        cm0=cm0,
        alpha_stall_rad=alpha_stall_rad,
        alpha_stall_deg=alpha_stall_deg,
        cl_at_stall=cl_stall,
        cd_at_stall=cd_stall,
        cm_at_stall=cm_stall,
        cla_stall=cla_stall,
        cda_stall=cda_stall,
        cma_stall=cma_stall,
        n_poststall=n_post,
        area=area,
        span=span,
        chord=chord,
        ar=ar,
        cp=cp,
        forward=forward,
        upward=upward,
        link_name=link_name,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_report(p: AldParams) -> None:
    print("=" * 70)
    print("GAZEBO LiftDrag Plugin Parameters")
    print("=" * 70)

    if p.warnings:
        print("\nWARNINGS:")
        for w in p.warnings:
            print(f"  ! {w}")

    print("\n--- Geometry ---")
    print(f"  Reference area S : {p.area:.5f} m²")
    if p.chord:
        print(f"  Mean chord c     : {p.chord:.4f} m")
    if p.span is not None:
        print(f"  Wingspan b       : {p.span:.4f} m")
        if p.ar is not None:
            print(f"  Aspect ratio AR  : {p.ar:.3f}  (b²/S)")
    else:
        print("  Wingspan b       : NOT PROVIDED  (pass --span)")
        print("  Aspect ratio AR  : NOT PROVIDED")
    print(f"  CP in link frame : ({p.cp[0]:.5f}, {p.cp[1]:.5f}, {p.cp[2]:.5f}) m")
    print(f"  Forward vector   : ({p.forward[0]:.1f}, {p.forward[1]:.1f}, {p.forward[2]:.1f})")
    print(f"  Upward vector    : ({p.upward[0]:.1f}, {p.upward[1]:.1f}, {p.upward[2]:.1f})")

    print("\n--- Pre-stall aerodynamics ---")
    print(f"  a0 (zero-lift α) : {p.a0_deg:+.4f}°  =  {p.a0_rad:+.6f} rad")
    print(f"  CLα pre-stall    : {p.cla_per_deg:.5f} /deg  =  {p.cla_per_rad:.5f} /rad  (R²={p.cla_r2:.4f})")
    print(f"  CDα=0 (drag@a0)  : {p.cda:.6f}  (CD at zero lift)")
    print(f"  Cmα pre-stall    : {p.cma_per_deg:.5f} /deg  =  {p.cma_per_rad:.5f} /rad  (R²={p.cma_r2:.4f})")
    print(f"  Cm₀ (at α=0)     : {p.cm0:+.5f}")

    print("\n--- Stall point ---")
    print(f"  α_stall          : {p.alpha_stall_deg:.2f}°  =  {p.alpha_stall_rad:.5f} rad")
    print(f"  CL at stall      : {p.cl_at_stall:.5f}")
    print(f"  CD at stall      : {p.cd_at_stall:.6f}")
    print(f"  Cm at stall      : {p.cm_at_stall:+.6f}")

    print(f"\n--- Post-stall slopes (fitted over {p.n_poststall} pts after stall) ---")
    print(f"  CLα post-stall   : {p.cla_stall:.5f} /rad")
    print(f"  CDα post-stall   : {p.cda_stall:.5f} /rad")
    print(f"  Cmα post-stall   : {p.cma_stall:.5f} /rad")

    print("=" * 70)


def generate_sdf(p: AldParams) -> str:
    """Return a ready-to-paste SDF plugin XML block."""
    area_str = f"{p.area:.6f}" if p.area is not None else "MISSING_AREA"
    cp_str = f"{p.cp[0]:.5f} {p.cp[1]:.5f} {p.cp[2]:.5f}"
    fwd_str = f"{p.forward[0]:.1f} {p.forward[1]:.1f} {p.forward[2]:.1f}"
    up_str = f"{p.upward[0]:.1f} {p.upward[1]:.1f} {p.upward[2]:.1f}"

    ar_comment = (f"<!-- AR = {p.ar:.3f} (b={p.span:.3f} m) -->"
                  if p.ar is not None else "<!-- AR: span not provided -->")

    lines = [
        '<plugin name="gz::sim::systems::LiftDrag"',
        '        filename="gz-sim-lift-drag-system">',
        f'  <!-- Link this plugin attaches to -->',
        f'  <link_name>{p.link_name}</link_name>',
        f'',
        f'  <!-- Reference area [m²] -->',
        f'  <area>{area_str}</area>  {ar_comment}',
        f'',
        f'  <!-- Air density [kg/m³] — match your world atmosphere -->',
        f'  <air_density>1.225</air_density>',
        f'',
        f'  <!-- Centre of pressure in link frame [m] (x y z) -->',
        f'  <cp>{cp_str}</cp>',
        f'',
        f'  <!-- Forward and upward unit vectors in link frame -->',
        f'  <forward>{fwd_str}</forward>',
        f'  <upward>{up_str}</upward>',
        f'',
        f'  <!-- Pre-stall: CL = cla * (alpha - a0) -->',
        f'  <a0>{p.a0_rad:.6f}</a0>       <!-- zero-lift AoA = {p.a0_deg:+.4f} deg -->',
        f'  <cla>{p.cla_per_rad:.6f}</cla>     <!-- dCL/dalpha = {p.cla_per_deg:.5f} /deg -->',
        f'',
        f'  <!-- Pre-stall drag: CD = cda + CL^2/(pi*AR*eff)  (set ar/eff below) -->',
        f'  <cda>{p.cda:.6f}</cda>     <!-- CD at zero lift (CD0) -->',
        (f'  <ar>{p.ar:.4f}</ar>          <!-- aspect ratio b^2/S -->'
         if p.ar is not None else
         f'  <!-- <ar>MISSING — pass --span</ar> -->'),
        f'  <eff>0.85</eff>          <!-- Oswald efficiency (typical 0.75–0.90) -->',
        f'',
        f'  <!-- Pre-stall pitching moment: Cm = cma*alpha + cm0 -->',
        f'  <cma>{p.cma_per_rad:.6f}</cma>     <!-- dCm/dalpha = {p.cma_per_deg:.5f} /deg -->',
        f'',
        f'  <!-- Stall breakpoint -->',
        f'  <alpha_stall>{p.alpha_stall_rad:.6f}</alpha_stall>  <!-- {p.alpha_stall_deg:.2f} deg -->',
        f'  <cl_at_alpha_stall>{p.cl_at_stall:.6f}</cl_at_alpha_stall>',
        f'  <cd_at_alpha_stall>{p.cd_at_stall:.6f}</cd_at_alpha_stall>',
        f'  <cm_at_alpha_stall>{p.cm_at_stall:.6f}</cm_at_alpha_stall>',
        f'',
        f'  <!-- Post-stall slopes (per rad): value at stall + slope*(alpha - alpha_stall) -->',
        f'  <cla_stall>{p.cla_stall:.6f}</cla_stall>',
        f'  <cda_stall>{p.cda_stall:.6f}</cda_stall>',
        f'  <cma_stall>{p.cma_stall:.6f}</cma_stall>',
        f'</plugin>',
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Optional diagnostic plots
# ---------------------------------------------------------------------------


def plot_fits(polar: dict, p: AldParams,
              save_path: str | None = None,
              show: bool = False) -> None:
    import matplotlib.pyplot as plt

    alpha_raw = np.asarray(polar["alpha"], dtype=float)
    cl_raw = np.asarray(polar["cl"], dtype=float)
    cd_raw = np.asarray(polar["cd"], dtype=float)
    cm_raw = np.asarray(polar["cm"], dtype=float)
    order = np.argsort(alpha_raw)
    alpha = alpha_raw[order]
    cl = cl_raw[order]
    cd = cd_raw[order]
    cm = cm_raw[order]

    a_fine = np.linspace(alpha[0], alpha[-1], 300)

    # Pre-stall CL model
    cl_model_pre = p.cla_per_deg * (a_fine - p.a0_deg)
    # Post-stall CL model (only for α > alpha_stall)
    a_post = a_fine[a_fine > p.alpha_stall_deg]
    cl_model_post = (p.cl_at_stall
                     + np.degrees(p.cla_stall) * (a_post - p.alpha_stall_deg))
    cd_model_post = (p.cd_at_stall
                     + np.degrees(p.cda_stall) * (a_post - p.alpha_stall_deg))
    cm_model_post = (p.cm_at_stall
                     + np.degrees(p.cma_stall) * (a_post - p.alpha_stall_deg))

    # Pre-stall CD (parabolic polar)
    cl_pre_fine = p.cla_per_deg * (a_fine[a_fine <= p.alpha_stall_deg] - p.a0_deg)
    if p.ar is not None:
        cd_model_pre = p.cda + cl_pre_fine ** 2 / (np.pi * p.ar * 0.85)
    else:
        cd_model_pre = np.full_like(cl_pre_fine, p.cda)

    # Cm model
    cm_model_pre = p.cma_per_deg * a_fine[a_fine <= p.alpha_stall_deg] + p.cm0

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Gazebo LiftDrag fit — Flow5 data vs plugin model", fontsize=11)

    # CL
    ax = axes[0]
    ax.plot(alpha, cl, "o", color="tab:blue", ms=5, label="Flow5 data")
    ax.plot(a_fine[a_fine <= p.alpha_stall_deg], cl_model_pre,
            "--", color="tab:orange", lw=2, label=f"Pre-stall fit (R²={p.cla_r2:.3f})")
    if len(a_post):
        ax.plot(a_post, cl_model_post, "--", color="tab:red", lw=2, label="Post-stall fit")
    ax.axvline(p.alpha_stall_deg, color="k", ls=":", lw=1, label=f"α_stall={p.alpha_stall_deg:.1f}°")
    ax.axvline(p.a0_deg, color="grey", ls=":", lw=1, label=f"a0={p.a0_deg:.2f}°")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("α [deg]"); ax.set_ylabel("CL")
    ax.set_title(f"CL  (CLα={p.cla_per_rad:.4f} /rad)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # CD
    ax = axes[1]
    ax.plot(alpha, cd, "s", color="tab:red", ms=5, label="Flow5 data")
    a_pre_fine = a_fine[a_fine <= p.alpha_stall_deg]
    ax.plot(a_pre_fine, cd_model_pre, "--", color="tab:orange", lw=2,
            label="Parabolic polar" + ("" if p.ar is not None else " (AR unknown → CD0 only)"))
    if len(a_post):
        ax.plot(a_post, cd_model_post, "--", color="tab:red", lw=2, label="Post-stall fit")
    ax.axvline(p.alpha_stall_deg, color="k", ls=":", lw=1)
    ax.set_xlabel("α [deg]"); ax.set_ylabel("CD")
    ax.set_title(f"CD  (CD0={p.cda:.5f})")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Cm
    ax = axes[2]
    ax.plot(alpha, cm, "^", color="tab:green", ms=5, label="Flow5 data")
    ax.plot(a_fine[a_fine <= p.alpha_stall_deg], cm_model_pre,
            "--", color="tab:orange", lw=2, label=f"Pre-stall fit (R²={p.cma_r2:.3f})")
    if len(a_post):
        ax.plot(a_post, cm_model_post, "--", color="tab:red", lw=2, label="Post-stall fit")
    ax.axvline(p.alpha_stall_deg, color="k", ls=":", lw=1)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("α [deg]"); ax.set_ylabel("Cm")
    ax.set_title(f"Cm  (Cmα={p.cma_per_rad:.4f} /rad)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved fit plot to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("polar_file", help="Flow5 polar .txt (alpha sweep)")
    ap.add_argument("--plane", default="plane.xml", help="Plane parameter XML")
    ap.add_argument("--span", type=float, default=None,
                    help="Wingspan [m] (not in plane.xml — required for AR and CP)")
    ap.add_argument("--link-name", default="base_link",
                    help="SDF link name for the plugin (default: base_link)")
    ap.add_argument("--cp-offset", type=float, nargs=3, metavar=("X", "Y", "Z"),
                    default=None,
                    help="Centre of pressure (x y z) in link frame [m]. "
                         "Defaults to (x_ac from plane.xml, 0, 0).")
    ap.add_argument("--forward", type=float, nargs=3, metavar=("X", "Y", "Z"),
                    default=(1.0, 0.0, 0.0),
                    help="Forward unit vector in link frame (default: 1 0 0)")
    ap.add_argument("--upward", type=float, nargs=3, metavar=("X", "Y", "Z"),
                    default=(0.0, 0.0, 1.0),
                    help="Upward unit vector in link frame (default: 0 0 1)")
    ap.add_argument("--fit-alpha-min", type=float, default=None,
                    help="Lower alpha bound [deg] for pre-stall linear fit")
    ap.add_argument("--fit-alpha-max", type=float, default=None,
                    help="Upper alpha bound [deg] for pre-stall linear fit")
    ap.add_argument("--n-poststall", type=int, default=5,
                    help="Number of post-stall data points used for slope fit (default 5)")
    ap.add_argument("--sdf-out", default=None,
                    help="Write SDF snippet to this file (default: print to stdout)")
    ap.add_argument("--plot", action="store_true", help="Save diagnostic fit plot")
    ap.add_argument("--plot-out", default=None,
                    help="Path for the fit plot (default: plots/<polar_stem>__ald_fit.png)")
    ap.add_argument("--show", action="store_true", help="Show plot interactively")
    args = ap.parse_args()

    polar = load_polar(Path(args.polar_file))
    plane = {"rho": 1.225, "gravity": 9.81}
    plane.update(load_plane_xml(Path(args.plane)))

    cp_offset = tuple(args.cp_offset) if args.cp_offset is not None else None

    params = compute_ald_params(
        polar=polar,
        plane=plane,
        span=args.span,
        link_name=args.link_name,
        cp_offset=cp_offset,
        forward=tuple(args.forward),
        upward=tuple(args.upward),
        fit_alpha_min=args.fit_alpha_min,
        fit_alpha_max=args.fit_alpha_max,
        n_poststall=args.n_poststall,
    )

    print_report(params)

    sdf = generate_sdf(params)
    if args.sdf_out:
        Path(args.sdf_out).write_text(sdf + "\n", encoding="utf-8")
        print(f"\nSDF snippet written to {args.sdf_out}")
    else:
        print("\n--- SDF Plugin Snippet ---\n")
        print(sdf)

    if args.plot or args.show:
        plot_path = args.plot_out
        if plot_path is None:
            stem = Path(args.polar_file).stem
            plot_path = os.path.join("plots", f"{stem}__ald_fit.png")
        plot_fits(polar, params, save_path=plot_path, show=args.show)


if __name__ == "__main__":
    main()
