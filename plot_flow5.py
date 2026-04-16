"""Generate report-quality plots from a Flow5 polar analysis text file.

Usage:
    py -3 plot_flow5.py [input.txt] [-o OUTPUT_DIR]

If no input file is given, every ``*.txt`` in the script directory is processed.
Plots for each input are written to ``OUTPUT_DIR/<input-stem>/`` (default
``OUTPUT_DIR`` is ``./plots``).
"""

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
DEFAULT_PLANE_XML = HERE / "plane.xml"


def load_plane_xml(path: Path) -> dict:
    if not path.exists():
        return {}
    root = ET.parse(path).getroot()

    def _float(el):
        if el is None or el.text is None:
            return None
        text = el.text.strip()
        return float(text) if text else None

    out = {}
    for key in ("mass", "area", "chord", "rho", "gravity", "arm", "dihedral", "cg", "ac"):
        val = _float(root.find(key))
        if val is not None:
            out[key] = val

    inertia_el = root.find("inertia")
    if inertia_el is not None:
        inertia = {}
        for key in ("Ixx", "Iyy", "Izz", "Ixz"):
            val = _float(inertia_el.find(key))
            if val is not None:
                inertia[key] = val
        if inertia:
            out["inertia"] = inertia

    return out

# Column indices (0-based) in the Flow5 data table.
COL_ALPHA = 1
COL_BETA = 2
COL_CL = 4
COL_CD = 5
COL_CY = 8
COL_CM = 9
COL_CL_ROLL = 12
COL_CN = 13
COL_XNP = 45
COL_MASS = 54
COL_COG_X = 55


def load_polar(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    title = lines[1].strip() if len(lines) > 1 else path.stem
    freestream = lines[2].strip() if len(lines) > 2 else ""

    rows = []
    for line in lines[6:]:
        parts = line.split()
        if not parts:
            continue
        try:
            rows.append([float(x) for x in parts])
        except ValueError:
            continue

    data = np.array(rows)
    ncols = data.shape[1]

    def _col(idx):
        return data[:, idx] if ncols > idx else np.full(len(data), np.nan)

    return {
        "title": title,
        "freestream": freestream,
        "alpha": data[:, COL_ALPHA],
        "beta": data[:, COL_BETA],
        "cl": data[:, COL_CL],
        "cd": data[:, COL_CD],
        "cy": data[:, COL_CY],
        "cm": data[:, COL_CM],
        "cl_roll": data[:, COL_CL_ROLL],
        "cn": data[:, COL_CN],
        "xnp": _col(COL_XNP),
        "mass_col": _col(COL_MASS),
        "cog_x": _col(COL_COG_X),
    }


def _sorted_xy(x, *ys):
    order = np.argsort(x)
    return (x[order],) + tuple(y[order] for y in ys)


def compute_cruise(polar, mass, area, rho, g):
    if mass is None or area is None or area <= 0 or mass <= 0:
        return None
    cl = polar["cl"]
    cd = polar["cd"]
    with np.errstate(divide="ignore", invalid="ignore"):
        ld = np.where(cd > 0, cl / cd, np.nan)
    if not np.any(np.isfinite(ld)):
        return None
    i_ld = int(np.nanargmax(ld))
    cl_opt = cl[i_ld]
    result = {
        "mass": mass,
        "area": area,
        "rho": rho,
        "g": g,
        "weight": mass * g,
        "i_ld": i_ld,
        "alpha_ld": polar["alpha"][i_ld],
        "cl_ld": cl_opt,
        "cd_ld": cd[i_ld],
        "ld_max": ld[i_ld],
    }
    if cl_opt > 0:
        result["v_maxrange"] = float(np.sqrt(2 * mass * g / (rho * area * cl_opt)))
    else:
        result["v_maxrange"] = float("nan")

    with np.errstate(divide="ignore", invalid="ignore"):
        endurance = np.where((cd > 0) & (cl > 0), cl ** 1.5 / cd, np.nan)
    if np.any(np.isfinite(endurance)):
        i_e = int(np.nanargmax(endurance))
        cl_e = cl[i_e]
        result["i_endurance"] = i_e
        result["alpha_endurance"] = polar["alpha"][i_e]
        result["cl_endurance"] = cl_e
        result["v_endurance"] = float(np.sqrt(2 * mass * g / (rho * area * cl_e)))
    return result


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        "axes.linewidth": 1.1,
        "lines.linewidth": 2.0,
        "lines.markersize": 5.5,
        "legend.frameon": False,
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


def annotate_subtitle(fig, title, freestream):
    # Intentionally left blank: file-name title and freestream speed are not
    # drawn on the figures.
    return


def plot_cl_vs_alpha(polar, out_path, annotate=True):
    a, cl = _sorted_xy(polar["alpha"], polar["cl"])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(a, cl, marker="o", color="#1f77b4", label=r"$C_L$ (data)")
    if annotate and len(a) >= 2:
        slope, intercept = np.polyfit(a, cl, 1)
        a_line = np.linspace(a.min(), a.max(), 100)
        ax.plot(a_line, slope * a_line + intercept, "--", color="#ff7f0e",
                linewidth=1.5,
                label=fr"fit: $C_L = {slope:.4f}\,\alpha {intercept:+.4f}$")
        a0 = -intercept / slope if slope != 0 else float("nan")
        ax.plot([], [], " ",
                label=fr"$C_{{L_\alpha}}$={slope:.4f} /deg, $\alpha_0$={a0:.2f}°")
    ax.axhline(0, color="#888888", linewidth=0.8)
    ax.axvline(0, color="#888888", linewidth=0.8)
    ax.set_xlabel(r"$\alpha$ (deg)")
    ax.set_ylabel(r"$C_L$")
    ax.set_title(r"$C_L$ vs. $\alpha$")
    ax.legend(loc="best")
    annotate_subtitle(fig, polar["title"], polar["freestream"])
    fig.savefig(out_path)
    plt.close(fig)


def plot_cd_vs_alpha(polar, out_path):
    a, cd = _sorted_xy(polar["alpha"], polar["cd"])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(a, cd, marker="s", color="#d62728", label=r"$C_D$")
    ax.axvline(0, color="#888888", linewidth=0.8)
    ax.set_xlabel(r"$\alpha$ (deg)")
    ax.set_ylabel(r"$C_D$")
    ax.set_title(r"$C_D$ vs. $\alpha$")
    ax.legend(loc="best")
    annotate_subtitle(fig, polar["title"], polar["freestream"])
    fig.savefig(out_path)
    plt.close(fig)


def plot_polar(polar, out_path, annotate=True):
    a, cl, cd = _sorted_xy(polar["alpha"], polar["cl"], polar["cd"])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(cd, cl, marker="o", color="#2ca02c",
            label=None if annotate else r"$C_L$")
    if annotate:
        ld = np.divide(cl, cd, out=np.zeros_like(cl), where=cd != 0)
        i_max = int(np.argmax(ld))
        ax.scatter([cd[i_max]], [cl[i_max]], s=80,
                   facecolor="none", edgecolor="#ff7f0e", linewidth=2,
                   label=fr"Max $L/D$ = {ld[i_max]:.2f} @ $\alpha$={a[i_max]:.1f}°")
    ax.axhline(0, color="#888888", linewidth=0.8)
    ax.set_xlabel(r"$C_D$")
    ax.set_ylabel(r"$C_L$")
    ax.set_title(r"$C_L$ vs. $C_D$")
    if annotate:
        ax.legend(loc="best")
    annotate_subtitle(fig, polar["title"], polar["freestream"])
    fig.savefig(out_path)
    plt.close(fig)


def plot_ld_vs_alpha(polar, out_path, annotate=True, cruise=None):
    a, cl, cd = _sorted_xy(polar["alpha"], polar["cl"], polar["cd"])
    ld = np.divide(cl, cd, out=np.zeros_like(cl), where=cd != 0)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(a, ld, marker="D", color="#9467bd", label=r"$L/D$")
    if annotate:
        i_max = int(np.argmax(ld))
        ax.scatter([a[i_max]], [ld[i_max]], s=80,
                   facecolor="none", edgecolor="#ff7f0e", linewidth=2,
                   label=fr"Max = {ld[i_max]:.2f} @ $\alpha$={a[i_max]:.1f}°")
        if cruise is not None and np.isfinite(cruise.get("v_maxrange", float("nan"))):
            ax.plot([], [], " ",
                    label=fr"$V_{{cruise}}$ = {cruise['v_maxrange']:.2f} m/s"
                          fr" (m={cruise['mass']:.2f} kg, S={cruise['area']:.3f} m²)")
            if "v_endurance" in cruise and np.isfinite(cruise["v_endurance"]):
                ax.plot([], [], " ",
                        label=fr"$V_{{endurance}}$ = {cruise['v_endurance']:.2f} m/s"
                              fr" @ $\alpha$={cruise['alpha_endurance']:.1f}°")
    ax.axhline(0, color="#888888", linewidth=0.8)
    ax.axvline(0, color="#888888", linewidth=0.8)
    ax.set_xlabel(r"$\alpha$ (deg)")
    ax.set_ylabel(r"$L/D$")
    ax.set_title(r"$L/D$ vs. $\alpha$")
    ax.legend(loc="best", fontsize=9)
    annotate_subtitle(fig, polar["title"], polar["freestream"])
    fig.savefig(out_path)
    plt.close(fig)


def plot_alpha_vs_speed(polar, out_path, mass, area, rho, g, annotate=True):
    """Angle of attack required for steady level flight (L = W) vs airspeed.

    Inverts the pre-stall CL(α) branch: CL_req = 2W/(ρV²S), α = CL⁻¹(CL_req).
    """
    alpha = np.asarray(polar["alpha"], dtype=float)
    cl = np.asarray(polar["cl"], dtype=float)
    order = np.argsort(alpha)
    alpha, cl = alpha[order], cl[order]
    i_stall = int(np.argmax(cl))
    cl_pre = cl[: i_stall + 1]
    a_pre = alpha[: i_stall + 1]
    o2 = np.argsort(cl_pre)
    cl_pre, a_pre = cl_pre[o2], a_pre[o2]

    cl_max = float(cl_pre[-1])
    cl_min = float(cl_pre[0])
    W = mass * g
    v_stall = float(np.sqrt(2 * W / (rho * area * cl_max)))
    v_max = (float(np.sqrt(2 * W / (rho * area * cl_min)))
             if cl_min > 0 else v_stall * 4.0)
    V = np.linspace(v_stall, v_max, 200)
    cl_req = 2 * W / (rho * area * V ** 2)
    a_req = np.interp(cl_req, cl_pre, a_pre)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(V, a_req, color="#8c6d31", linewidth=2.0,
            label=r"$\alpha$ for $L=W$")
    if annotate:
        ax.axvline(v_stall, color="#d62728", linestyle="--", linewidth=1.2,
                   label=fr"$V_{{stall}}$ = {v_stall:.2f} m/s"
                         fr" @ $\alpha$={a_pre[-1]:.1f}°")
        ax.plot([], [], " ",
                label=fr"m = {mass:.2f} kg, S = {area:.3f} m²,"
                      fr" $\rho$ = {rho:.3f}")
    ax.axhline(0, color="#888888", linewidth=0.8)
    ax.set_xlabel(r"Airspeed $V$ (m/s)")
    ax.set_ylabel(r"$\alpha$ (deg)")
    ax.set_title(r"Level-flight $\alpha$ vs. airspeed")
    if annotate:
        ax.legend(loc="best", fontsize=9)
    annotate_subtitle(fig, polar["title"], polar["freestream"])
    fig.savefig(out_path)
    plt.close(fig)


def plot_cm_vs_alpha(polar, out_path, annotate=True):
    a, cm, cl = _sorted_xy(polar["alpha"], polar["cm"], polar["cl"])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(a, cm, marker="^", color="#17becf", label=r"$C_m$ (data)")
    if annotate and len(a) >= 2:
        cm_slope, cm_int = np.polyfit(a, cm, 1)
        a_line = np.linspace(a.min(), a.max(), 100)
        ax.plot(a_line, cm_slope * a_line + cm_int, "--", color="#ff7f0e",
                linewidth=1.5,
                label=fr"fit: $C_m = {cm_slope:.4f}\,\alpha {cm_int:+.4f}$")

        a_trim = -cm_int / cm_slope if cm_slope != 0 else float("nan")
        if np.isfinite(a_trim) and a.min() <= a_trim <= a.max():
            ax.scatter([a_trim], [0.0], s=90, facecolor="none",
                       edgecolor="#d62728", linewidth=2,
                       label=fr"trim: $C_m=0$ @ $\alpha$={a_trim:.2f}°")

        cl_slope = np.polyfit(a, cl, 1)[0]
        if cl_slope != 0:
            sm = -cm_slope / cl_slope
            ax.plot([], [], " ",
                    label=fr"$-C_{{m_\alpha}}/C_{{L_\alpha}}$ = {sm:+.4f} $\bar c$")

            cog_x = polar.get("cog_x")
            if cog_x is not None and np.all(np.isfinite(cog_x)) and cog_x.size:
                x_cg = float(np.mean(cog_x))
                xnp_file = polar.get("xnp")
                if (xnp_file is not None and np.all(np.isfinite(xnp_file))
                        and xnp_file.size):
                    x_np_file = float(np.mean(xnp_file))
                    c_ref = (x_np_file - x_cg) / sm if sm != 0 else float("nan")
                    if np.isfinite(c_ref) and c_ref > 0:
                        ax.plot([], [], " ",
                                label=fr"$x_{{NP}}$ = {x_np_file:.4f} m"
                                      fr" ($x_{{CG}}$={x_cg:.4f}, $\bar c$={c_ref:.3f} m)")
                    else:
                        ax.plot([], [], " ",
                                label=fr"$x_{{NP}}$ = {x_np_file:.4f} m,"
                                      fr" $x_{{CG}}$ = {x_cg:.4f} m")

    ax.axhline(0, color="#888888", linewidth=0.8)
    ax.axvline(0, color="#888888", linewidth=0.8)
    ax.set_xlabel(r"$\alpha$ (deg)")
    ax.set_ylabel(r"$C_m$")
    ax.set_title(r"$C_m$ vs. $\alpha$")
    ax.legend(loc="best", fontsize=9)
    annotate_subtitle(fig, polar["title"], polar["freestream"])
    fig.savefig(out_path)
    plt.close(fig)


def plot_cy_vs_beta(polar, out_path, annotate=True):
    b, cy = _sorted_xy(polar["beta"], polar["cy"])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(b, cy, marker="o", color="#1f77b4", label=r"$C_Y$")
    if annotate and len(b) >= 2:
        slope = np.polyfit(b, cy, 1)[0]
        ax.plot([], [], " ", label=fr"$C_{{Y_\beta}}$ = {slope:.4f} /deg")
    ax.axhline(0, color="#888888", linewidth=0.8)
    ax.axvline(0, color="#888888", linewidth=0.8)
    ax.set_xlabel(r"$\beta$ (deg)")
    ax.set_ylabel(r"$C_Y$")
    ax.set_title(r"$C_Y$ vs. $\beta$")
    ax.legend(loc="best")
    annotate_subtitle(fig, polar["title"], polar["freestream"])
    fig.savefig(out_path)
    plt.close(fig)


def plot_cl_roll_vs_beta(polar, out_path, annotate=True):
    b, cl_roll = _sorted_xy(polar["beta"], polar["cl_roll"])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(b, cl_roll, marker="s", color="#2ca02c", label=r"$C_l$")
    if annotate and len(b) >= 2:
        slope = np.polyfit(b, cl_roll, 1)[0]
        ax.plot([], [], " ", label=fr"$C_{{l_\beta}}$ = {slope:.4f} /deg")
    ax.axhline(0, color="#888888", linewidth=0.8)
    ax.axvline(0, color="#888888", linewidth=0.8)
    ax.set_xlabel(r"$\beta$ (deg)")
    ax.set_ylabel(r"$C_l$")
    ax.set_title(r"$C_l$ vs. $\beta$")
    ax.legend(loc="best")
    annotate_subtitle(fig, polar["title"], polar["freestream"])
    fig.savefig(out_path)
    plt.close(fig)


def plot_cn_vs_beta(polar, out_path, annotate=True):
    b, cn = _sorted_xy(polar["beta"], polar["cn"])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(b, cn, marker="^", color="#d62728", label=r"$C_n$")
    if annotate and len(b) >= 2:
        slope = np.polyfit(b, cn, 1)[0]
        ax.plot([], [], " ", label=fr"$C_{{n_\beta}}$ = {slope:.4f} /deg")
    ax.axhline(0, color="#888888", linewidth=0.8)
    ax.axvline(0, color="#888888", linewidth=0.8)
    ax.set_xlabel(r"$\beta$ (deg)")
    ax.set_ylabel(r"$C_n$")
    ax.set_title(r"$C_n$ vs. $\beta$")
    ax.legend(loc="best")
    annotate_subtitle(fig, polar["title"], polar["freestream"])
    fig.savefig(out_path)
    plt.close(fig)


def plot_combined(polar, out_path, annotate=True):
    a, cl, cd = _sorted_xy(polar["alpha"], polar["cl"], polar["cd"])
    ld = np.divide(cl, cd, out=np.zeros_like(cl), where=cd != 0)
    i_max = int(np.argmax(ld))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    ax.plot(a, cl, marker="o", color="#1f77b4", label="data")
    if annotate and len(a) >= 2:
        slope, intercept = np.polyfit(a, cl, 1)
        a_line = np.linspace(a.min(), a.max(), 100)
        ax.plot(a_line, slope * a_line + intercept, "--", color="#ff7f0e",
                linewidth=1.5,
                label=fr"$C_L = {slope:.4f}\,\alpha {intercept:+.4f}$")
        ax.legend(loc="best", fontsize=9)
    ax.axhline(0, color="#888888", linewidth=0.8)
    ax.axvline(0, color="#888888", linewidth=0.8)
    ax.set_xlabel(r"$\alpha$ (deg)")
    ax.set_ylabel(r"$C_L$")
    ax.set_title(r"$C_L$ vs. $\alpha$")

    ax = axes[0, 1]
    ax.plot(a, cd, marker="s", color="#d62728")
    ax.axvline(0, color="#888888", linewidth=0.8)
    ax.set_xlabel(r"$\alpha$ (deg)")
    ax.set_ylabel(r"$C_D$")
    ax.set_title(r"$C_D$ vs. $\alpha$")

    ax = axes[1, 0]
    ax.plot(cd, cl, marker="o", color="#2ca02c")
    if annotate:
        ax.scatter([cd[i_max]], [cl[i_max]], s=80,
                   facecolor="none", edgecolor="#ff7f0e", linewidth=2)
    ax.axhline(0, color="#888888", linewidth=0.8)
    ax.set_xlabel(r"$C_D$")
    ax.set_ylabel(r"$C_L$")
    ax.set_title(r"$C_L$ vs. $C_D$")

    ax = axes[1, 1]
    ax.plot(a, ld, marker="D", color="#9467bd")
    if annotate:
        ax.scatter([a[i_max]], [ld[i_max]], s=80,
                   facecolor="none", edgecolor="#ff7f0e", linewidth=2,
                   label=fr"Max $L/D$ = {ld[i_max]:.2f} @ $\alpha$={a[i_max]:.1f}°")
        ax.legend(loc="best", fontsize=9)
    ax.axhline(0, color="#888888", linewidth=0.8)
    ax.axvline(0, color="#888888", linewidth=0.8)
    ax.set_xlabel(r"$\alpha$ (deg)")
    ax.set_ylabel(r"$L/D$")
    ax.set_title(r"$L/D$ vs. $\alpha$")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_combined_beta(polar, out_path):
    b, cy, cl_roll, cn = _sorted_xy(
        polar["beta"], polar["cy"], polar["cl_roll"], polar["cn"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    ax.plot(b, cy, marker="o", color="#1f77b4")
    ax.axhline(0, color="#888888", linewidth=0.8)
    ax.axvline(0, color="#888888", linewidth=0.8)
    ax.set_xlabel(r"$\beta$ (deg)")
    ax.set_ylabel(r"$C_Y$")
    ax.set_title(r"$C_Y$ vs. $\beta$")

    ax = axes[1]
    ax.plot(b, cl_roll, marker="s", color="#2ca02c")
    ax.axhline(0, color="#888888", linewidth=0.8)
    ax.axvline(0, color="#888888", linewidth=0.8)
    ax.set_xlabel(r"$\beta$ (deg)")
    ax.set_ylabel(r"$C_l$")
    ax.set_title(r"$C_l$ vs. $\beta$")

    ax = axes[2]
    ax.plot(b, cn, marker="^", color="#d62728")
    ax.axhline(0, color="#888888", linewidth=0.8)
    ax.axvline(0, color="#888888", linewidth=0.8)
    ax.set_xlabel(r"$\beta$ (deg)")
    ax.set_ylabel(r"$C_n$")
    ax.set_title(r"$C_n$ vs. $\beta$")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def process_file(data_file: Path, out_root: Path, params=None) -> None:
    polar = load_polar(data_file)
    stem = data_file.stem
    out_dir = out_root / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    alpha_varies = np.ptp(polar["alpha"]) > 1e-9
    beta_varies = np.ptp(polar["beta"]) > 1e-9

    params = params or {}
    mass = params.get("mass")
    if mass is None:
        mass_col = polar.get("mass_col")
        if mass_col is not None and np.all(np.isfinite(mass_col)) and mass_col.size:
            mass = float(np.mean(mass_col))
    area = params.get("area")
    chord = params.get("chord")
    rho = params.get("rho", 1.225)
    g = params.get("gravity", 9.81)

    cruise = compute_cruise(polar, mass, area, rho, g) if alpha_varies else None

    plain_dir = out_dir / "plain"
    plain_dir.mkdir(exist_ok=True)

    n_annotated = 0
    n_plain = 0

    if alpha_varies:
        plot_cl_vs_alpha(polar, out_dir / f"{stem}__CL_vs_alpha.png", annotate=True)
        plot_cd_vs_alpha(polar, out_dir / f"{stem}__CD_vs_alpha.png")
        plot_polar(polar, out_dir / f"{stem}__Polar.png", annotate=True)
        plot_ld_vs_alpha(polar, out_dir / f"{stem}__LD_vs_alpha.png",
                         annotate=True, cruise=cruise)
        plot_cm_vs_alpha(polar, out_dir / f"{stem}__Cm_vs_alpha.png", annotate=True)
        plot_combined(polar, out_dir / f"{stem}__Summary.png", annotate=True)
        n_annotated += 6

        if mass is not None and area is not None and area > 0 and mass > 0:
            plot_alpha_vs_speed(polar,
                                out_dir / f"{stem}__alpha_vs_V.png",
                                mass, area, rho, g, annotate=True)
            plot_alpha_vs_speed(polar,
                                plain_dir / f"{stem}__alpha_vs_V.png",
                                mass, area, rho, g, annotate=False)
            n_annotated += 1
            n_plain += 1

        plot_cl_vs_alpha(polar, plain_dir / f"{stem}__CL_vs_alpha.png", annotate=False)
        plot_polar(polar, plain_dir / f"{stem}__Polar.png", annotate=False)
        plot_ld_vs_alpha(polar, plain_dir / f"{stem}__LD_vs_alpha.png", annotate=False)
        plot_cm_vs_alpha(polar, plain_dir / f"{stem}__Cm_vs_alpha.png", annotate=False)
        plot_combined(polar, plain_dir / f"{stem}__Summary.png", annotate=False)
        n_plain += 5

    if beta_varies:
        plot_cy_vs_beta(polar, out_dir / f"{stem}__CY_vs_beta.png", annotate=True)
        plot_cl_roll_vs_beta(polar, out_dir / f"{stem}__Cl_vs_beta.png", annotate=True)
        plot_cn_vs_beta(polar, out_dir / f"{stem}__Cn_vs_beta.png", annotate=True)
        plot_combined_beta(polar, out_dir / f"{stem}__Summary_beta.png")
        n_annotated += 4

        plot_cy_vs_beta(polar, plain_dir / f"{stem}__CY_vs_beta.png", annotate=False)
        plot_cl_roll_vs_beta(polar, plain_dir / f"{stem}__Cl_vs_beta.png", annotate=False)
        plot_cn_vs_beta(polar, plain_dir / f"{stem}__Cn_vs_beta.png", annotate=False)
        n_plain += 3

    print(f"[{data_file.name}] wrote {n_annotated} annotated + {n_plain} plain plots to: {out_dir}")
    print(f"  Data points: {len(polar['alpha'])}")
    if alpha_varies:
        print(f"  Alpha range: {polar['alpha'].min():.1f}° to {polar['alpha'].max():.1f}°")
        ld = polar["cl"] / np.where(polar["cd"] == 0, np.nan, polar["cd"])
        i = int(np.nanargmax(ld))
        print(f"  Max L/D = {ld[i]:.2f} at alpha = {polar['alpha'][i]:.1f}°"
              f" (CL={polar['cl'][i]:.3f}, CD={polar['cd'][i]:.4f})")
        a_s, cl_s, cm_s = _sorted_xy(polar["alpha"], polar["cl"], polar["cm"])
        if len(a_s) >= 2:
            cl_slope, cl_int = np.polyfit(a_s, cl_s, 1)
            cm_slope, cm_int = np.polyfit(a_s, cm_s, 1)
            print(f"  CL fit: CL = {cl_slope:.4f} * alpha {cl_int:+.4f}"
                  f"   (CL_alpha={cl_slope:.4f} /deg)")
            print(f"  Cm fit: Cm = {cm_slope:.4f} * alpha {cm_int:+.4f}"
                  f"   (Cm_alpha={cm_slope:.4f} /deg)")
            if cl_slope != 0:
                sm = -cm_slope / cl_slope
                print(f"  Static margin SM = -Cm_alpha/CL_alpha = {sm:+.4f} c_ref"
                      f"  ({'stable' if sm > 0 else 'unstable'})")
                if cm_slope != 0:
                    a_trim = -cm_int / cm_slope
                    print(f"  Trim alpha (Cm=0): {a_trim:+.3f}°")
                cog_x = polar.get("cog_x")
                xnp_file = polar.get("xnp")
                if (cog_x is not None and np.all(np.isfinite(cog_x)) and cog_x.size
                        and xnp_file is not None and np.all(np.isfinite(xnp_file))
                        and xnp_file.size):
                    x_cg = float(np.mean(cog_x))
                    x_np = float(np.mean(xnp_file))
                    dx = x_np - x_cg
                    print(f"  Neutral point (from file): x_NP = {x_np:.4f} m,"
                          f" x_CG = {x_cg:.4f} m, x_NP - x_CG = {dx*1000:+.2f} mm")
                    if chord is not None and chord > 0:
                        dx_fit = sm * chord
                        print(f"  Neutral point (from fit, c={chord:.4f} m):"
                              f" x_NP - x_CG = {dx_fit*1000:+.2f} mm"
                              f"  -> x_NP = {x_cg + dx_fit:.4f} m")
                    else:
                        c_ref_impl = dx / sm if sm != 0 else float("nan")
                        if np.isfinite(c_ref_impl) and c_ref_impl > 0:
                            print(f"  Implied reference chord c_ref = {c_ref_impl:.4f} m"
                                  f" (pass --chord to override)")

        if cruise is not None:
            print(f"  --- Cruise (rho={rho:.3f} kg/m^3,"
                  f" m={cruise['mass']:.3f} kg, S={cruise['area']:.4f} m^2) ---")
            print(f"  Max range (max L/D): V = {cruise['v_maxrange']:.2f} m/s"
                  f" @ alpha={cruise['alpha_ld']:.1f}°"
                  f" (CL={cruise['cl_ld']:.3f}, L/D={cruise['ld_max']:.2f})")
            if "v_endurance" in cruise:
                print(f"  Max endurance (max CL^1.5/CD): V = {cruise['v_endurance']:.2f} m/s"
                      f" @ alpha={cruise['alpha_endurance']:.1f}°"
                      f" (CL={cruise['cl_endurance']:.3f})")
        elif area is None:
            print("  (pass --mass and --area to compute optimal cruise speed)")
    if beta_varies:
        print(f"  Beta range:  {polar['beta'].min():.1f}° to {polar['beta'].max():.1f}°")
        b_sorted, cy_s, cl_s, cn_s = _sorted_xy(
            polar["beta"], polar["cy"], polar["cl_roll"], polar["cn"])
        if len(b_sorted) >= 2:
            cy_b = np.polyfit(b_sorted, cy_s, 1)[0]
            cl_b = np.polyfit(b_sorted, cl_s, 1)[0]
            cn_b = np.polyfit(b_sorted, cn_s, 1)[0]
            print(f"  CY_beta = {cy_b:+.4f} /deg, "
                  f"Cl_beta = {cl_b:+.4f} /deg, "
                  f"Cn_beta = {cn_b:+.4f} /deg")


def parse_args(argv):
    p = argparse.ArgumentParser(
        description="Generate report-quality plots from Flow5 polar text files.")
    p.add_argument("inputs", nargs="*", type=Path,
                   help="Flow5 .txt result file(s). If omitted, all *.txt "
                        "files next to the script are processed.")
    p.add_argument("-o", "--output-dir", type=Path, default=HERE / "plots",
                   help="Directory to write plots into (default: ./plots). "
                        "Each input gets its own subfolder.")
    p.add_argument("--plane", type=Path, default=DEFAULT_PLANE_XML,
                   help="Plane parameter XML file (default: ./plane.xml). "
                        "Provides mass, area, chord, rho, gravity.")
    p.add_argument("--mass", type=float, default=None,
                   help="Aircraft mass in kg. Overrides plane.xml; "
                        "falls back to value from data file.")
    p.add_argument("--area", type=float, default=None,
                   help="Reference wing area S in m². Overrides plane.xml. "
                        "Required for V_cruise.")
    p.add_argument("--chord", type=float, default=None,
                   help="Reference chord c in m. Overrides plane.xml. "
                        "Used for absolute neutral-point location.")
    p.add_argument("--rho", type=float, default=None,
                   help="Air density in kg/m³. Overrides plane.xml (default 1.225).")
    p.add_argument("--gravity", type=float, default=None,
                   help="Gravitational acceleration in m/s². "
                        "Overrides plane.xml (default 9.81).")
    return p.parse_args(argv)


def resolve_inputs(inputs):
    if inputs:
        resolved = []
        for f in inputs:
            path = f if f.is_absolute() else (Path.cwd() / f)
            if not path.exists():
                # Fall back to script directory for convenience.
                alt = HERE / f.name
                if alt.exists():
                    path = alt
            if not path.exists():
                print(f"error: input file not found: {f}", file=sys.stderr)
                sys.exit(1)
            resolved.append(path)
        return resolved
    found = sorted(HERE.glob("*.txt"))
    if not found:
        print(f"error: no .txt files found in {HERE}", file=sys.stderr)
        sys.exit(1)
    return found


def main(argv=None):
    args = parse_args(argv)
    style()
    params = {"rho": 1.225, "gravity": 9.81}
    params.update(load_plane_xml(args.plane))
    for key in ("mass", "area", "chord", "rho", "gravity"):
        val = getattr(args, key)
        if val is not None:
            params[key] = val
    for data_file in resolve_inputs(args.inputs):
        process_file(data_file, args.output_dir, params=params)


if __name__ == "__main__":
    main()
