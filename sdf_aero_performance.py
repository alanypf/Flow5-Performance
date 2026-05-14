"""Aircraft performance from a Gazebo SDF aero model + provided powertrain.

Reads the AdvancedLiftDrag plugin block (and link mass) directly from the
SDF model file -- the same model that drives the Gazebo simulation -- and
couples it to a user-supplied (motor, propeller, battery) combination so
the resulting numbers can be compared against the simulation that uses a
different motor/prop integrated in the SDF.

Two cruise solves are produced for comparison:

  * "Gazebo default" -- the integrated propeller CSV from the SDF
    (read by PropellerPerformancePlugin) driven by an idealised
    velocity-controlled rotor: no battery / no electrical losses,
    only the max RPM implied by the ArduPilotPlugin <multiplier>.
    This is the upper-bound performance the simulator can produce.

  * "Provided powertrain" -- the same airframe with the user-supplied
    propeller (.txt), motor (.xml) and battery (.xml). Throttle is
    bisected against drag using the same torque-balance solver as
    motor_prop_performance.py, so this reports realistic electrical
    power, pack current, endurance and range.

Aero coefficients follow the AdvancedLiftDrag formulation
([gz-sim/src/systems/advanced_lift_drag/AdvancedLiftDrag.cc]):

    sigma(a) = sigmoid blend around +/- alpha_stall  (M = 15 default)
    CL(a)    = (1 - sigma) * (CL0 + CLa * a)
             + sigma * 2 * sign(a) * sin(a)^2 * cos(a)
    CD_fp    = 2 / (1 + exp(k1 + k2 * max(AR, 1/AR)))    k1=-0.224, k2=-0.115
    CD(a)    = (1 - sigma) * (CD0 + CL^2 / (pi * AR * eff))
             + sigma * | CD_fp * (0.5 - 0.5 * cos(2 a)) |
    Cm(a)    = Cem0 + Cema * a                              (pre-stall)
             = Cem0 + Cema*alpha_stall + CemaStall*(a - alpha_stall)  (post)

Steady level cruise:
    L = W                =>  CL_req(V) = 2 W / (rho S V^2)
    alpha_trim(V) such that CL(alpha) = CL_req(V) on the pre-stall branch
    D(V)                 = 0.5 rho V^2 S CD(alpha_trim)
    Thrust_req per motor = D / N_motors  (props axis along body x)

The motor + prop + battery model is shared with motor_prop_performance.py
(same torque-balance root-finder used by performance.py); only the
airframe side changes -- it comes from the SDF, not a Flow5 polar.

Usage:
    python sdf_aero_performance.py \
        --sdf model-aero-VITERNA-m.sdf \
        --prop PER3_7x11E.txt --motor motor.xml --battery battery.xml \
        --motors 4 --vmin 40 --vmax 90 --vstep 5 \
        --save plots/sdf_perf.png \
        --save-aero plots/sdf_aero.png \
        --save-compare plots/sdf_compare.png --no-show
"""

from __future__ import annotations

import argparse
import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

from motor_prop_performance import (
    Battery,
    Motor,
    OperatingPoint,
    PropPoint,
    Propeller,
    load_battery,
    load_motor,
    load_propeller,
    solve_operating_point,
)


# ---------------------------------------------------------------------------
# CSV propeller loader -- same format as PropellerPerformancePlugin reads
# (rpm, v_ms, thrust_N, torque_Nm), so the Propeller object built here is
# functionally identical to what the Gazebo plugin uses at runtime.
# ---------------------------------------------------------------------------


def load_propeller_csv(path: str) -> Propeller:
    """Load propeller performance from the SDF-side CSV format."""
    import csv
    points: list[PropPoint] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rpm = float(row["rpm"])
                v_ms = float(row["v_ms"])
                thrust_N = float(row["thrust_N"])
                torque_Nm = float(row["torque_Nm"])
            except (KeyError, ValueError):
                continue
            omega = rpm * 2.0 * math.pi / 60.0
            points.append(PropPoint(
                rpm=rpm, v_ms=v_ms, J=float("nan"),
                Ct=float("nan"), Cp=float("nan"),
                thrust_N=thrust_N, torque_Nm=torque_Nm,
                power_W=torque_Nm * omega,
            ))
    if not points:
        raise ValueError(f"No prop data parsed from {path}")
    name = os.path.splitext(os.path.basename(path))[0]
    # Extract diameter/pitch from filename like "PER3_7x11E"
    import re
    m = re.search(r"(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)", name)
    diameter_in = float(m.group(1)) if m else 0.0
    pitch_in = float(m.group(2)) if m else 0.0
    return Propeller(name=name, diameter_in=diameter_in,
                     pitch_in=pitch_in, points=points)


# ---------------------------------------------------------------------------
# SDF parsing -- AdvancedLiftDrag plugin parameters and link mass
# ---------------------------------------------------------------------------


@dataclass
class AeroModel:
    # geometry / reference
    rho: float
    area: float
    AR: float
    mac: float
    eff: float
    # linear (pre-stall) coefficients
    CL0: float
    CLa: float       # per rad
    CD0: float
    Cem0: float
    Cema: float      # per rad
    a0_rad: float    # zero-lift angle of attack (linear sense, info only)
    # stall / post-stall
    alpha_stall: float    # rad
    CLa_stall: float
    CDa_stall: float
    Cema_stall: float
    # sigmoid blend steepness (matches plugin default)
    M: float = 15.0
    # flat-plate drag sigmoid (matches plugin defaults)
    CD_fp_k1: float = -0.224
    CD_fp_k2: float = -0.115
    # link the plugin is attached to
    link_name: str = ""

    @property
    def CD_fp(self) -> float:
        ar_eff = max(self.AR, 1.0 / max(self.AR, 1e-9))
        return 2.0 / (1.0 + math.exp(self.CD_fp_k1 + self.CD_fp_k2 * ar_eff))


def _findtext(elem: ET.Element, tag: str, default: float | None = None,
              cast=float):
    e = elem.find(tag)
    if e is None or e.text is None or not e.text.strip():
        if default is None:
            raise ValueError(f"SDF missing <{tag}>")
        return default
    return cast(e.text.strip())


@dataclass
class SdfPropulsion:
    """Powertrain-related metadata read directly from the SDF."""
    perf_file_uri: str = ""        # e.g. "model://waterdrop/propellers/PER3_7x11E.csv"
    n_motors: int = 0
    max_rad_per_s: float = 0.0     # from |ArduPilotPlugin multiplier|

    @property
    def max_rpm(self) -> float:
        return self.max_rad_per_s * 60.0 / (2.0 * math.pi)


def load_sdf_model(path: str) -> tuple[AeroModel, float, SdfPropulsion, dict]:
    """Return (AeroModel, total_mass_kg, SdfPropulsion, info) from a Gazebo SDF file.

    Total mass sums every <inertial><mass> in the model -- so it includes
    the airframe plus motors/imu links.
    """
    tree = ET.parse(path)
    root = tree.getroot()
    model = root.find("model")
    if model is None:
        raise ValueError(f"{path} has no <model> element")

    # Sum link masses (airframe + propeller links + imu -- matches Gazebo).
    total_mass = 0.0
    link_masses: dict[str, float] = {}
    for link in model.findall("link"):
        m_elem = link.find("inertial/mass")
        if m_elem is not None and m_elem.text:
            m = float(m_elem.text.strip())
            total_mass += m
            link_masses[link.get("name", "?")] = m

    # Pull the AdvancedLiftDrag plugin (by filename or by Cl0/CLa presence).
    ald = None
    prop_perf_files: list[str] = []
    max_mult = 0.0
    for plug in model.findall("plugin"):
        fn = (plug.get("filename") or "").lower()
        if "advanced-lift-drag" in fn or "advancedliftdrag" in fn:
            ald = plug
        if "propellerperformanceplugin" in fn:
            pf = plug.find("performance_file")
            if pf is not None and pf.text:
                prop_perf_files.append(pf.text.strip())
        if "ardupilotplugin" in fn:
            for ctrl in plug.findall("control"):
                m_el = ctrl.find("multiplier")
                if m_el is not None and m_el.text:
                    try:
                        max_mult = max(max_mult, abs(float(m_el.text.strip())))
                    except ValueError:
                        pass
    if ald is None:
        raise ValueError(f"No AdvancedLiftDrag plugin found in {path}")

    aero = AeroModel(
        rho=_findtext(ald, "air_density", 1.225),
        area=_findtext(ald, "area"),
        AR=_findtext(ald, "AR"),
        mac=_findtext(ald, "mac"),
        eff=_findtext(ald, "eff", 1.0),
        CL0=_findtext(ald, "CL0", 0.0),
        CLa=_findtext(ald, "CLa"),
        CD0=_findtext(ald, "CD0", 0.0),
        Cem0=_findtext(ald, "Cem0", 0.0),
        Cema=_findtext(ald, "Cema", 0.0),
        a0_rad=_findtext(ald, "a0", 0.0),
        alpha_stall=_findtext(ald, "alpha_stall"),
        CLa_stall=_findtext(ald, "CLa_stall", 0.0),
        CDa_stall=_findtext(ald, "CDa_stall", 0.0),
        Cema_stall=_findtext(ald, "Cema_stall", 0.0),
        link_name=_findtext(ald, "link_name", "", cast=str),
    )

    prop_uri = prop_perf_files[0] if prop_perf_files else ""
    propulsion = SdfPropulsion(
        perf_file_uri=prop_uri,
        n_motors=len(prop_perf_files),
        max_rad_per_s=max_mult,
    )

    info = {
        "model_name": model.get("name", "?"),
        "link_masses": link_masses,
        "ald_present": True,
    }
    return aero, total_mass, propulsion, info


def resolve_sdf_uri(uri: str, sdf_path: str) -> str | None:
    """Resolve a model:// URI to an on-disk path next to the SDF.

    The waterdrop model lives at ardupilot_gazebo/models/waterdrop/ and the
    SDF here is a standalone file -- so we try a couple of fallbacks based
    on the workspace layout used in this repo.
    """
    if not uri:
        return None
    if uri.startswith("model://"):
        rel = uri[len("model://"):]
        # model://waterdrop/propellers/foo.csv -> waterdrop/propellers/foo.csv
        candidates = [
            os.path.join(os.path.dirname(os.path.abspath(sdf_path)),
                         "..", "ardupilot_gazebo", "models", rel),
            os.path.join("/home/alan/uav_sim_workspace",
                         "ardupilot_gazebo", "models", rel),
        ]
        for c in candidates:
            c = os.path.normpath(c)
            if os.path.exists(c):
                return c
        return None
    if os.path.isabs(uri) and os.path.exists(uri):
        return uri
    rel_to_sdf = os.path.join(os.path.dirname(os.path.abspath(sdf_path)), uri)
    return rel_to_sdf if os.path.exists(rel_to_sdf) else None


# ---------------------------------------------------------------------------
# Aero coefficient functions (vectorised over alpha in radians)
# ---------------------------------------------------------------------------


def _sigma(alpha: np.ndarray, alpha_stall: float, M: float) -> np.ndarray:
    """Blending function used by AdvancedLiftDrag (0 pre-stall, ~1 post-stall)."""
    # numerically stabilised version of the formula in the plugin
    num = 1.0 + np.exp(-M * (alpha - alpha_stall)) + np.exp(M * (alpha + alpha_stall))
    den = (1.0 + np.exp(-M * (alpha - alpha_stall))) * (1.0 + np.exp(M * (alpha + alpha_stall)))
    return num / den


def CL_of_alpha(aero: AeroModel, alpha: np.ndarray) -> np.ndarray:
    a = np.asarray(alpha, dtype=float)
    s = _sigma(a, aero.alpha_stall, aero.M)
    sin_a, cos_a = np.sin(a), np.cos(a)
    sign_a = np.where(a == 0.0, 1.0, np.sign(a))
    CL_pre = aero.CL0 + aero.CLa * a
    CL_post = 2.0 * sign_a * sin_a * sin_a * cos_a
    return (1.0 - s) * CL_pre + s * CL_post


def CD_of_alpha(aero: AeroModel, alpha: np.ndarray) -> np.ndarray:
    a = np.asarray(alpha, dtype=float)
    s = _sigma(a, aero.alpha_stall, aero.M)
    CL = CL_of_alpha(aero, a)
    CD_pre = aero.CD0 + (CL * CL) / (math.pi * aero.AR * aero.eff)
    CD_post = np.abs(aero.CD_fp * (0.5 - 0.5 * np.cos(2.0 * a)))
    return (1.0 - s) * CD_pre + s * CD_post


def Cm_of_alpha(aero: AeroModel, alpha: np.ndarray) -> np.ndarray:
    """Pitching-moment coefficient about the link origin (per plugin)."""
    a = np.asarray(alpha, dtype=float)
    aS = aero.alpha_stall
    Cm = aero.Cem0 + aero.Cema * a
    above = a > aS
    below = a < -aS
    if np.any(above):
        Cm = np.where(above,
                      aero.Cem0 + aero.Cema * aS
                      + aero.Cema_stall * (a - aS),
                      Cm)
    if np.any(below):
        Cm = np.where(below,
                      aero.Cem0 - aero.Cema * aS
                      + aero.Cema_stall * (a + aS),
                      Cm)
    return Cm


# ---------------------------------------------------------------------------
# Trim solve: alpha such that CL(alpha) = CL_req on the pre-stall branch
# ---------------------------------------------------------------------------


def alpha_trim(aero: AeroModel, CL_req: float) -> float:
    """Pre-stall alpha [rad] giving CL_req. NaN if outside [CL_min, CL_max]."""
    # Sample pre-stall branch densely; the plugin equations are monotone
    # up to alpha_stall so a tabulate + interp inversion is robust.
    a_lo = -aero.alpha_stall + 1e-3
    a_hi = aero.alpha_stall - 1e-3
    alphas = np.linspace(a_lo, a_hi, 401)
    CLs = CL_of_alpha(aero, alphas)
    cl_min, cl_max = float(CLs.min()), float(CLs.max())
    if CL_req < cl_min or CL_req > cl_max:
        return float("nan")
    # CL is monotone increasing pre-stall -> sort just in case of numerical noise
    order = np.argsort(CLs)
    return float(np.interp(CL_req, CLs[order], alphas[order]))


@dataclass
class StallSpeeds:
    """1g level-flight stall speeds under three different CL_max definitions.

    * blended     -- CL_max taken as the first local maximum of the
                     AdvancedLiftDrag blended CL(alpha) curve as alpha rises
                     from 0. This is what the *simulator* can sustain in
                     steady level flight, because at lower speeds cruise_sweep
                     can no longer find a trim alpha. Use this for stall-margin
                     parameters (e.g. AIRSPEED_MIN = 1.2 * blended).
    * linear      -- CL_max from the textbook linear extrapolation
                     CL0 + CLa * alpha_stall. This is what a wing-only polar
                     or a hand calc that ignores post-stall behaviour gives.
                     Reported as a diagnostic; do not size AIRSPEED_MIN from it.
    * analytical  -- Closed-form evaluation of the ALD blended formula AT
                     alpha = alpha_stall. By symmetry of the sigmoid the
                     blend weight is sigma = 0.5 there, so
                       CL_max = 0.5*(CL0 + CLa*alpha_stall)
                              + 0.5*(2*sin^2(alpha_stall)*cos(alpha_stall))
                     This is the pen-and-paper textbook analytical answer that
                     accounts for the plugin's blend without numerical search.
                     Typically falls between the linear and the blended-peak
                     values, and provides a useful sanity check.
    """
    blended: float
    linear: float
    analytical: float
    CL_max_blended: float
    CL_max_linear: float
    CL_max_analytical: float
    alpha_at_blended_peak_rad: float


def stall_speed(aero: AeroModel, mass: float, g: float = 9.81) -> StallSpeeds:
    """Return the simulator-accurate, textbook-linear, and textbook-analytical V_stall."""
    W = mass * g
    qSinv = 1.0 / (0.5 * aero.rho * aero.area)
    # Linear / textbook CL_max -- evaluated at alpha_stall on the pure
    # pre-stall slope, ignoring the sigmoid blend.
    cl_lin = aero.CL0 + aero.CLa * aero.alpha_stall
    # Closed-form analytical CL_max -- ALD blended formula at alpha = alpha_stall,
    # where sigma = 1/2 by symmetry. The flat-plate post-stall lift there is
    # 2 * sin^2(alpha_stall) * cos(alpha_stall).
    cl_post_at_stall = 2.0 * math.sin(aero.alpha_stall) ** 2 * math.cos(aero.alpha_stall)
    cl_analytical = 0.5 * cl_lin + 0.5 * cl_post_at_stall
    # Blended CL_max -- scan alpha from 0 past alpha_stall by a few sigmoid
    # half-widths and pick the FIRST local maximum. This is robust to changes
    # in M (sigmoid sharpness) and alpha_stall, and avoids accidentally
    # latching onto the rising post-stall flat-plate plateau (which is not
    # a stall lift -- the wing has long since departed by that AoA).
    a_hi = aero.alpha_stall + 6.0 / max(aero.M, 1.0)
    alphas = np.linspace(0.0, a_hi, 8001)
    CLs = CL_of_alpha(aero, alphas)
    # First index where CL starts decreasing (numerically robust to noise).
    drops = np.where(np.diff(CLs) < 0.0)[0]
    if drops.size:
        peak_idx = int(drops[0])
    else:
        peak_idx = int(np.argmax(CLs))
    cl_blend = float(CLs[peak_idx])
    alpha_blend = float(alphas[peak_idx])

    def _vs(cl_max: float) -> float:
        if cl_max <= 0 or not math.isfinite(cl_max):
            return float("nan")
        return math.sqrt(W * qSinv / cl_max)

    return StallSpeeds(
        blended=_vs(cl_blend),
        linear=_vs(cl_lin),
        analytical=_vs(cl_analytical),
        CL_max_blended=cl_blend,
        CL_max_linear=cl_lin,
        CL_max_analytical=cl_analytical,
        alpha_at_blended_peak_rad=alpha_blend,
    )


# ---------------------------------------------------------------------------
# Throttle bisector (per-motor thrust target)
# ---------------------------------------------------------------------------


def _throttle_for_thrust(motor: Motor, prop: Propeller, battery: Battery,
                        V: float, T_req: float,
                        soc: float = 1.0) -> tuple[float, OperatingPoint] | None:
    op_hi = solve_operating_point(motor, prop, battery, 1.0, V, soc=soc)
    if op_hi is None or op_hi.thrust_N < T_req:
        return None
    op_lo = solve_operating_point(motor, prop, battery, 0.0, V, soc=soc)
    if op_lo is not None and op_lo.thrust_N >= T_req:
        return 0.0, op_lo
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        op = solve_operating_point(motor, prop, battery, mid, V, soc=soc)
        T = op.thrust_N if op is not None else 0.0
        if T < T_req:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-4:
            break
    t = 0.5 * (lo + hi)
    op = solve_operating_point(motor, prop, battery, t, V, soc=soc)
    return (t, op) if op is not None else None


# ---------------------------------------------------------------------------
# Cruise sweep
# ---------------------------------------------------------------------------


@dataclass
class CruisePoint:
    V: float
    alpha_deg: float
    CL: float
    CD: float
    Cm_link: float
    drag_N: float
    L_over_D: float
    thrust_per_motor_N: float
    throttle: float
    rpm: float
    current_per_motor_A: float
    P_elec_per_motor_W: float
    P_elec_total_W: float
    endurance_h: float
    range_km: float
    feasible: bool       # full-throttle thrust >= drag (level-flight capable)


def cruise_sweep(aero: AeroModel, mass: float, V_arr: np.ndarray,
                 motor: Motor, prop: Propeller, battery: Battery,
                 n_motors: int, g: float = 9.81, soc: float = 1.0,
                 usable_fraction: float = 0.8) -> list[CruisePoint]:
    W = mass * g
    E_Wh = battery.V_nominal * battery.capacity_Ah * usable_fraction
    out: list[CruisePoint] = []
    for V in V_arr:
        V = float(V)
        if V <= 0:
            continue
        CL_req = 2.0 * W / (aero.rho * aero.area * V * V)
        a = alpha_trim(aero, CL_req)
        if not math.isfinite(a):
            # Below stall -- record an "infeasible" point so plots show the gap
            out.append(CruisePoint(
                V=V, alpha_deg=float("nan"), CL=CL_req, CD=float("nan"),
                Cm_link=float("nan"), drag_N=float("nan"),
                L_over_D=float("nan"),
                thrust_per_motor_N=float("nan"), throttle=float("nan"),
                rpm=float("nan"), current_per_motor_A=float("nan"),
                P_elec_per_motor_W=float("nan"), P_elec_total_W=float("nan"),
                endurance_h=float("nan"), range_km=float("nan"),
                feasible=False,
            ))
            continue
        CD = float(CD_of_alpha(aero, a))
        Cm = float(Cm_of_alpha(aero, a))
        D = 0.5 * aero.rho * V * V * aero.area * CD
        T_per = D / n_motors
        res = _throttle_for_thrust(motor, prop, battery, V, T_per, soc=soc)
        if res is None:
            out.append(CruisePoint(
                V=V, alpha_deg=math.degrees(a), CL=CL_req, CD=CD,
                Cm_link=Cm, drag_N=D, L_over_D=CL_req / CD if CD > 0 else float("nan"),
                thrust_per_motor_N=T_per, throttle=float("nan"),
                rpm=float("nan"), current_per_motor_A=float("nan"),
                P_elec_per_motor_W=float("nan"), P_elec_total_W=float("nan"),
                endurance_h=float("nan"), range_km=float("nan"),
                feasible=False,
            ))
            continue
        t, op = res
        P_total = op.P_elec_W * n_motors
        endurance_h = E_Wh / P_total if P_total > 0 else float("nan")
        range_km = V * endurance_h * 3.6
        out.append(CruisePoint(
            V=V, alpha_deg=math.degrees(a), CL=CL_req, CD=CD,
            Cm_link=Cm, drag_N=D,
            L_over_D=CL_req / CD if CD > 0 else float("nan"),
            thrust_per_motor_N=op.thrust_N, throttle=t,
            rpm=op.rpm, current_per_motor_A=op.current_A,
            P_elec_per_motor_W=op.P_elec_W, P_elec_total_W=P_total,
            endurance_h=endurance_h, range_km=range_km,
            feasible=True,
        ))
    return out


# ---------------------------------------------------------------------------
# Gazebo "default setup" cruise: same aero, same prop CSV, idealised rotor.
# No motor / battery model -- just find RPM that meets the thrust demand and
# report mechanical (shaft) power.  RPM is capped by the ArduPilotPlugin
# multiplier from the SDF (rad/s -> RPM).
# ---------------------------------------------------------------------------


@dataclass
class GazeboCruisePoint:
    V: float
    alpha_deg: float
    CL: float
    CD: float
    drag_N: float
    thrust_per_motor_N: float
    rpm: float
    torque_per_motor_Nm: float
    P_shaft_per_motor_W: float
    P_shaft_total_W: float
    feasible: bool


def _rpm_for_thrust(prop: Propeller, V: float, T_req: float,
                    rpm_max: float) -> tuple[float, PropPoint] | None:
    """Find RPM in [rpm_min, rpm_max] where prop_thrust(rpm, V) = T_req."""
    rpms = prop.rpms
    rpm_lo = float(rpms[0])
    rpm_hi = float(min(rpm_max, rpms[-1])) if rpm_max > 0 else float(rpms[-1])
    if rpm_hi <= rpm_lo:
        return None
    T_lo = prop.at(rpm_lo, V).thrust_N
    T_hi = prop.at(rpm_hi, V).thrust_N
    if T_hi < T_req:
        return None
    if T_lo >= T_req:
        return rpm_lo, prop.at(rpm_lo, V)
    lo, hi = rpm_lo, rpm_hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        T = prop.at(mid, V).thrust_N
        if T < T_req:
            lo = mid
        else:
            hi = mid
        if hi - lo < 0.5:
            break
    rpm = 0.5 * (lo + hi)
    return rpm, prop.at(rpm, V)


def cruise_sweep_gazebo(aero: AeroModel, mass: float, V_arr: np.ndarray,
                        prop: Propeller, n_motors: int, rpm_max: float,
                        g: float = 9.81) -> list[GazeboCruisePoint]:
    W = mass * g
    out: list[GazeboCruisePoint] = []
    for V in V_arr:
        V = float(V)
        if V <= 0:
            continue
        CL_req = 2.0 * W / (aero.rho * aero.area * V * V)
        a = alpha_trim(aero, CL_req)
        if not math.isfinite(a):
            out.append(GazeboCruisePoint(
                V=V, alpha_deg=float("nan"), CL=CL_req, CD=float("nan"),
                drag_N=float("nan"), thrust_per_motor_N=float("nan"),
                rpm=float("nan"), torque_per_motor_Nm=float("nan"),
                P_shaft_per_motor_W=float("nan"), P_shaft_total_W=float("nan"),
                feasible=False,
            ))
            continue
        CD = float(CD_of_alpha(aero, a))
        D = 0.5 * aero.rho * V * V * aero.area * CD
        T_per = D / n_motors
        res = _rpm_for_thrust(prop, V, T_per, rpm_max)
        if res is None:
            out.append(GazeboCruisePoint(
                V=V, alpha_deg=math.degrees(a), CL=CL_req, CD=CD,
                drag_N=D, thrust_per_motor_N=T_per,
                rpm=float("nan"), torque_per_motor_Nm=float("nan"),
                P_shaft_per_motor_W=float("nan"), P_shaft_total_W=float("nan"),
                feasible=False,
            ))
            continue
        rpm, pp = res
        omega = rpm * 2.0 * math.pi / 60.0
        P_per = pp.torque_Nm * omega
        out.append(GazeboCruisePoint(
            V=V, alpha_deg=math.degrees(a), CL=CL_req, CD=CD,
            drag_N=D, thrust_per_motor_N=pp.thrust_N,
            rpm=rpm, torque_per_motor_Nm=pp.torque_Nm,
            P_shaft_per_motor_W=P_per, P_shaft_total_W=P_per * n_motors,
            feasible=True,
        ))
    return out


# ---------------------------------------------------------------------------
# Diagnostic speeds derived from the trim sweep
# ---------------------------------------------------------------------------


@dataclass
class CruiseSummary:
    V_stall: float                # simulator-accurate (blended CL_max)
    V_stall_linear: float         # textbook CL0 + CLa*alpha_stall (info only)
    V_best_LD: float
    V_best_endurance: float
    V_best_range: float
    V_min_power: float
    V_max: float


def cruise_summary(aero: AeroModel, mass: float,
                   pts: list[CruisePoint], g: float = 9.81) -> CruiseSummary:
    V = np.array([p.V for p in pts])
    feas = np.array([p.feasible for p in pts])
    LD = np.array([p.L_over_D for p in pts])
    end_h = np.array([p.endurance_h for p in pts])
    rng_km = np.array([p.range_km for p in pts])
    P_tot = np.array([p.P_elec_total_W for p in pts])

    def argmax_safe(arr: np.ndarray) -> int | None:
        valid = np.isfinite(arr) & feas
        if not np.any(valid):
            return None
        idxs = np.where(valid)[0]
        return int(idxs[np.argmax(arr[valid])])

    def argmin_safe(arr: np.ndarray) -> int | None:
        valid = np.isfinite(arr) & feas
        if not np.any(valid):
            return None
        idxs = np.where(valid)[0]
        return int(idxs[np.argmin(arr[valid])])

    i_ld = argmax_safe(LD)
    i_e = argmax_safe(end_h)
    i_r = argmax_safe(rng_km)
    i_pmin = argmin_safe(P_tot)
    feas_idx = np.where(feas & np.isfinite(P_tot))[0]
    V_max = float(V[feas_idx[-1]]) if feas_idx.size else float("nan")

    vs = stall_speed(aero, mass, g)
    return CruiseSummary(
        V_stall=vs.blended,
        V_stall_linear=vs.linear,
        V_best_LD=float(V[i_ld]) if i_ld is not None else float("nan"),
        V_best_endurance=float(V[i_e]) if i_e is not None else float("nan"),
        V_best_range=float(V[i_r]) if i_r is not None else float("nan"),
        V_min_power=float(V[i_pmin]) if i_pmin is not None else float("nan"),
        V_max=V_max,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _annotate_v(ax, V: float, label: str, color: str = "k") -> None:
    if not math.isfinite(V):
        return
    ax.axvline(V, color=color, linestyle="--", linewidth=0.8, alpha=0.7)
    ymin, ymax = ax.get_ylim()
    ax.text(V, ymin + 0.05 * (ymax - ymin), f" {label}={V:.1f}",
            rotation=90, va="bottom", ha="left", fontsize=8, color=color)


def plot_aero(aero: AeroModel, save_path: str | None, show: bool) -> None:
    import matplotlib.pyplot as plt
    alpha_deg = np.linspace(-25, 60, 400)
    a = np.deg2rad(alpha_deg)
    CL = CL_of_alpha(aero, a)
    CD = CD_of_alpha(aero, a)
    Cm = Cm_of_alpha(aero, a)
    with np.errstate(divide="ignore", invalid="ignore"):
        LD = np.where(CD > 0, CL / CD, np.nan)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    ax = axes[0, 0]
    ax.plot(alpha_deg, CL)
    ax.axvline(math.degrees(aero.alpha_stall), color="r", ls="--", lw=0.7,
               label=f"alpha_stall={math.degrees(aero.alpha_stall):.1f} deg")
    ax.set_ylabel("CL"); ax.grid(alpha=0.3); ax.legend()
    ax.set_title("Lift coefficient")

    ax = axes[0, 1]
    ax.plot(alpha_deg, CD)
    ax.set_ylabel("CD"); ax.grid(alpha=0.3)
    ax.set_title("Drag coefficient")

    ax = axes[1, 0]
    ax.plot(alpha_deg, LD)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("CL/CD"); ax.set_xlabel("alpha [deg]")
    ax.grid(alpha=0.3); ax.set_title("Aerodynamic efficiency")

    ax = axes[1, 1]
    ax.plot(alpha_deg, Cm)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("Cm"); ax.set_xlabel("alpha [deg]")
    ax.grid(alpha=0.3); ax.set_title("Pitching moment (about link origin)")

    fig.suptitle("SDF AdvancedLiftDrag aerodynamic model", fontsize=12)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_cruise(aero: AeroModel, mass: float, pts: list[CruisePoint],
                summary: CruiseSummary, motor: Motor, prop: Propeller,
                battery: Battery, n_motors: int,
                save_path: str | None, show: bool) -> None:
    import matplotlib.pyplot as plt

    V = np.array([p.V for p in pts])
    a_deg = np.array([p.alpha_deg for p in pts])
    CL = np.array([p.CL for p in pts])
    CD = np.array([p.CD for p in pts])
    LD = np.array([p.L_over_D for p in pts])
    drag = np.array([p.drag_N for p in pts])
    thr = np.array([p.throttle for p in pts])
    rpm = np.array([p.rpm for p in pts])
    I_per = np.array([p.current_per_motor_A for p in pts])
    P_tot = np.array([p.P_elec_total_W for p in pts])
    end_h = np.array([p.endurance_h for p in pts])
    rng_km = np.array([p.range_km for p in pts])

    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    title = (f"SDF aero ({aero.area:.3f} m^2, AR={aero.AR:.2f}, m={mass:.2f} kg)"
             f"  +  {prop.name}  +  {motor.name} x{n_motors}"
             f"  +  {battery.series}S{battery.parallel}P {battery.chemistry}")
    fig.suptitle(title, fontsize=11)

    # 1. Trimmed AOA
    ax = axes[0, 0]
    ax.plot(V, a_deg, color="C0")
    ax.axhline(math.degrees(aero.alpha_stall), color="r", ls="--", lw=0.7,
               label="alpha_stall")
    ax.set_ylabel("alpha_trim [deg]"); ax.set_xlabel("V [m/s]")
    ax.set_title("Trimmed angle of attack")
    ax.grid(alpha=0.3); ax.legend()
    _annotate_v(ax, summary.V_stall, "V_stall", "r")

    # 2. CL required
    ax = axes[0, 1]
    ax.plot(V, CL, color="C0")
    alphas = np.linspace(-aero.alpha_stall + 1e-3, aero.alpha_stall - 1e-3, 401)
    ax.axhline(float(CL_of_alpha(aero, alphas).max()), color="r", ls="--",
               lw=0.7, label="CL_max (pre-stall)")
    ax.set_ylabel("CL required (L=W)"); ax.set_xlabel("V [m/s]")
    ax.set_title("Lift coefficient required"); ax.grid(alpha=0.3); ax.legend()

    # 3. Drag and required thrust
    ax = axes[0, 2]
    ax.plot(V, drag, color="C0", label="Drag = Thrust_total")
    ax.plot(V, drag / n_motors, color="C3", ls=":", label="Thrust per motor")
    ax.set_ylabel("Force [N]"); ax.set_xlabel("V [m/s]")
    ax.set_title("Drag / required thrust"); ax.grid(alpha=0.3); ax.legend()
    _annotate_v(ax, summary.V_best_LD, "V_LDmax", "g")

    # 4. L/D
    ax = axes[1, 0]
    ax.plot(V, LD, color="C0")
    ax.set_ylabel("L / D"); ax.set_xlabel("V [m/s]")
    ax.set_title("Aerodynamic efficiency"); ax.grid(alpha=0.3)
    _annotate_v(ax, summary.V_best_LD, "V_LDmax", "g")

    # 5. Throttle
    ax = axes[1, 1]
    ax.plot(V, thr, color="C0")
    ax.axhline(1.0, color="r", ls="--", lw=0.7, label="full throttle")
    ax.set_ylabel("Throttle"); ax.set_xlabel("V [m/s]")
    ax.set_title("Throttle setting (cruise trim)"); ax.grid(alpha=0.3)
    ax.legend()
    _annotate_v(ax, summary.V_max, "V_max", "r")

    # 6. RPM
    ax = axes[1, 2]
    ax.plot(V, rpm, color="C0")
    ax.set_ylabel("RPM"); ax.set_xlabel("V [m/s]")
    ax.set_title("Propeller RPM at trim"); ax.grid(alpha=0.3)

    # 7. Total electrical power
    ax = axes[2, 0]
    ax.plot(V, P_tot, color="C0", label="Pack power")
    ax.set_ylabel("Power [W]"); ax.set_xlabel("V [m/s]")
    ax.set_title("Battery power required"); ax.grid(alpha=0.3)
    _annotate_v(ax, summary.V_min_power, "V_Pmin", "m")
    ax.legend()

    # 8. Endurance
    ax = axes[2, 1]
    ax.plot(V, end_h * 60.0, color="C0")
    ax.set_ylabel("Endurance [min]"); ax.set_xlabel("V [m/s]")
    ax.set_title("Endurance"); ax.grid(alpha=0.3)
    _annotate_v(ax, summary.V_best_endurance, "V_E", "m")

    # 9. Range
    ax = axes[2, 2]
    ax.plot(V, rng_km, color="C0")
    ax.set_ylabel("Range [km]"); ax.set_xlabel("V [m/s]")
    ax.set_title("Range"); ax.grid(alpha=0.3)
    _annotate_v(ax, summary.V_best_range, "V_R", "g")

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_comparison(aero: AeroModel, mass: float,
                    gz_pts: list[GazeboCruisePoint],
                    pw_pts: list[CruisePoint],
                    summary: CruiseSummary,
                    motor: Motor, prop_provided: Propeller,
                    prop_gazebo: Propeller, battery: Battery,
                    n_motors: int, rpm_max: float,
                    save_path: str | None, show: bool) -> None:
    """Overlay Gazebo-default and provided-powertrain cruise curves."""
    import matplotlib.pyplot as plt

    V_gz = np.array([p.V for p in gz_pts])
    rpm_gz = np.array([p.rpm for p in gz_pts])
    T_gz = np.array([p.thrust_per_motor_N for p in gz_pts])
    P_gz = np.array([p.P_shaft_total_W for p in gz_pts])
    a_gz = np.array([p.alpha_deg for p in gz_pts])
    drag_gz = np.array([p.drag_N for p in gz_pts])

    V_pw = np.array([p.V for p in pw_pts])
    rpm_pw = np.array([p.rpm for p in pw_pts])
    T_pw = np.array([p.thrust_per_motor_N for p in pw_pts])
    P_pw_elec = np.array([p.P_elec_total_W for p in pw_pts])
    thr_pw = np.array([p.throttle for p in pw_pts])
    a_pw = np.array([p.alpha_deg for p in pw_pts])
    drag_pw = np.array([p.drag_N for p in pw_pts])
    end_pw_min = np.array([p.endurance_h * 60.0 for p in pw_pts])
    rng_pw_km = np.array([p.range_km for p in pw_pts])

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    title = (f"Gazebo-default vs provided powertrain  --  "
             f"airframe: SDF aero (m={mass:.2f} kg, S={aero.area:.3f} m^2)")
    fig.suptitle(title, fontsize=11)

    # 1. Trimmed AOA (aero-only, identical for both -- shown for reference)
    ax = axes[0, 0]
    ax.plot(V_gz, a_gz, color="C2", label="Gazebo default")
    ax.plot(V_pw, a_pw, color="C0", ls="--", label="Provided powertrain")
    ax.axhline(math.degrees(aero.alpha_stall), color="r", ls=":", lw=0.7,
               label="alpha_stall")
    ax.set_ylabel("alpha_trim [deg]"); ax.set_xlabel("V [m/s]")
    ax.set_title("Trimmed angle of attack")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    _annotate_v(ax, summary.V_stall, "V_stall", "r")

    # 2. Drag and per-motor required thrust (aero-driven, identical)
    ax = axes[0, 1]
    ax.plot(V_gz, drag_gz, color="k", label="Drag (= total thrust)")
    ax.plot(V_gz, T_gz, color="C2", label="Gazebo per-motor thrust achieved")
    ax.plot(V_pw, T_pw, color="C0", ls="--", label="Provided per-motor thrust")
    ax.set_ylabel("Force [N]"); ax.set_xlabel("V [m/s]")
    ax.set_title("Cruise thrust / drag balance")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # 3. RPM comparison
    ax = axes[0, 2]
    ax.plot(V_gz, rpm_gz, color="C2", label=f"Gazebo ({prop_gazebo.name})")
    ax.plot(V_pw, rpm_pw, color="C0", ls="--",
            label=f"Provided ({prop_provided.name})")
    if rpm_max and math.isfinite(rpm_max):
        ax.axhline(rpm_max, color="r", ls=":", lw=0.7,
                   label=f"SDF max RPM = {rpm_max:.0f}")
    ax.set_ylabel("RPM"); ax.set_xlabel("V [m/s]")
    ax.set_title("Propeller RPM at trim")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # 4. Shaft vs electrical power
    ax = axes[1, 0]
    ax.plot(V_gz, P_gz, color="C2", label="Gazebo shaft power (ideal)")
    ax.plot(V_pw, P_pw_elec, color="C0", ls="--",
            label="Provided electrical (pack) power")
    ax.set_ylabel("Power [W]"); ax.set_xlabel("V [m/s]")
    ax.set_title("Power required")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    _annotate_v(ax, summary.V_min_power, "V_Pmin", "m")

    # 5. Throttle for the provided powertrain
    ax = axes[1, 1]
    ax.plot(V_pw, thr_pw, color="C0")
    ax.axhline(1.0, color="r", ls="--", lw=0.7, label="full throttle")
    ax.set_ylabel("Throttle (provided PT)"); ax.set_xlabel("V [m/s]")
    ax.set_title("Throttle setting -- realistic powertrain")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    _annotate_v(ax, summary.V_max, "V_max", "r")

    # 6. Endurance + range (provided powertrain only -- needs battery)
    ax = axes[1, 2]
    ax2 = ax.twinx()
    l1, = ax.plot(V_pw, end_pw_min, color="C0", label="Endurance [min]")
    l2, = ax2.plot(V_pw, rng_pw_km, color="C3", ls="--", label="Range [km]")
    ax.set_xlabel("V [m/s]")
    ax.set_ylabel("Endurance [min]", color="C0")
    ax2.set_ylabel("Range [km]", color="C3")
    ax.set_title("Mission performance (provided powertrain)")
    ax.grid(alpha=0.3)
    ax.legend([l1, l2], [l1.get_label(), l2.get_label()], fontsize=8, loc="best")
    _annotate_v(ax, summary.V_best_endurance, "V_E", "m")
    _annotate_v(ax, summary.V_best_range, "V_R", "g")

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_summary(aero: AeroModel, mass: float, motor: Motor,
                   prop_provided: Propeller, prop_gazebo: Propeller,
                   battery: Battery, n_motors: int,
                   propulsion: SdfPropulsion,
                   summary: CruiseSummary, pts: list[CruisePoint],
                   gz_pts: list[GazeboCruisePoint]) -> None:
    print()
    print("=== Inputs ===")
    print(f"  SDF aero: area={aero.area:.4f} m^2  AR={aero.AR:.3f}  mac={aero.mac:.4f} m")
    print(f"            CL0={aero.CL0:+.4f}  CLa={aero.CLa:.3f}/rad  CD0={aero.CD0:.4f}")
    print(f"            alpha_stall={math.degrees(aero.alpha_stall):.2f} deg")
    print(f"            eff={aero.eff:.3f}  rho={aero.rho:.3f} kg/m^3")
    print(f"  Mass (sum of SDF link masses): {mass:.3f} kg")
    print(f"  SDF propulsion: {propulsion.n_motors} motors, "
          f"integrated prop = {propulsion.perf_file_uri or '(none found)'}")
    print(f"                  max RPM from ArduPilotPlugin multiplier: "
          f"{propulsion.max_rpm:.0f} ({propulsion.max_rad_per_s:.1f} rad/s)")
    print(f"  Provided powertrain: {motor.name} x{n_motors}, prop {prop_provided.name},")
    print(f"                       battery {battery.series}S{battery.parallel}P "
          f"{battery.chemistry}, V_pack(nom)={battery.V_nominal:.2f} V, "
          f"capacity={battery.capacity_Ah:.2f} Ah")

    print()
    print("=== Cruise summary (provided powertrain) ===")
    print(f"  V_stall            : {summary.V_stall:6.2f} m/s")
    print(f"  V at L/D max       : {summary.V_best_LD:6.2f} m/s  (best range, no propulsion)")
    print(f"  V at min pack power: {summary.V_min_power:6.2f} m/s")
    print(f"  V_best_endurance   : {summary.V_best_endurance:6.2f} m/s")
    print(f"  V_best_range       : {summary.V_best_range:6.2f} m/s")
    print(f"  V_max              : {summary.V_max:6.2f} m/s")

    print()
    print("=== Gazebo default setup cruise (idealised motor, prop CSV from SDF) ===")
    print(f"{'V[m/s]':>7} {'a[deg]':>7} {'CL':>6} {'CD':>6} {'D[N]':>6} "
          f"{'RPM':>6} {'Q[Nm]':>6} {'P_shaft[W]':>11}")
    for p in gz_pts:
        if not p.feasible:
            continue
        print(f"{p.V:7.2f} {p.alpha_deg:7.2f} {p.CL:6.3f} {p.CD:6.3f} "
              f"{p.drag_N:6.2f} {p.rpm:6.0f} {p.torque_per_motor_Nm:6.3f} "
              f"{p.P_shaft_total_W:11.1f}")

    print()
    print("=== Provided powertrain cruise table ===")
    print(f"{'V[m/s]':>7} {'a[deg]':>7} {'CL':>6} {'CD':>6} {'L/D':>6} "
          f"{'D[N]':>6} {'thr':>5} {'RPM':>6} {'I/m[A]':>7} "
          f"{'P[W]':>7} {'t[min]':>7} {'R[km]':>7}")
    for p in pts:
        if not p.feasible:
            continue
        print(f"{p.V:7.2f} {p.alpha_deg:7.2f} {p.CL:6.3f} {p.CD:6.3f} "
              f"{p.L_over_D:6.2f} {p.drag_N:6.2f} {p.throttle:5.2f} "
              f"{p.rpm:6.0f} {p.current_per_motor_A:7.2f} "
              f"{p.P_elec_total_W:7.1f} {p.endurance_h * 60:7.1f} "
              f"{p.range_km:7.2f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sdf", default="model-aero-VITERNA-m.sdf",
                   help="SDF model file with AdvancedLiftDrag plugin (default: %(default)s)")
    p.add_argument("--prop", default="PER3_7x11E.txt",
                   help="APC-format propeller file (default: %(default)s)")
    p.add_argument("--motor", default="motor.xml",
                   help="Motor XML (default: %(default)s)")
    p.add_argument("--battery", default="battery.xml",
                   help="Battery XML (default: %(default)s)")
    p.add_argument("--motors", type=int, default=4,
                   help="Number of motors on the airframe (default: %(default)s)")
    p.add_argument("--mass", type=float, default=None,
                   help="Override total mass [kg] (default: sum of SDF link masses)")
    p.add_argument("--vmin", type=float, default=1.0)
    p.add_argument("--vmax", type=float, default=60.0)
    p.add_argument("--vstep", type=float, default=1.0)
    p.add_argument("--soc", type=float, default=1.0,
                   help="Battery state of charge 0..1 (default: %(default)s)")
    p.add_argument("--usable-fraction", type=float, default=0.8,
                   help="Usable battery energy fraction (default: %(default)s)")
    p.add_argument("--save", default=None,
                   help="Save the cruise figure to this path (e.g. plots/sdf_perf.png)")
    p.add_argument("--save-aero", default=None,
                   help="Save the aero-model figure (CL/CD/Cm vs alpha) here")
    p.add_argument("--save-compare", default=None,
                   help="Save the Gazebo-vs-provided comparison figure here")
    p.add_argument("--gazebo-prop", default=None,
                   help="Override propeller CSV used by the SDF "
                        "(default: resolve <performance_file> from the SDF)")
    p.add_argument("--no-show", action="store_true",
                   help="Skip plt.show() (useful when only saving figures)")
    args = p.parse_args()

    aero, sdf_mass, propulsion, info = load_sdf_model(args.sdf)
    mass = args.mass if args.mass is not None else sdf_mass

    prop_provided = load_propeller(args.prop)
    motor = load_motor(args.motor)
    battery = load_battery(args.battery)

    # Resolve the propeller CSV that the SDF actually uses, falling back to
    # the provided .txt prop (still the same physical 7x11E data) if the CSV
    # can't be located on disk.
    csv_path = args.gazebo_prop or resolve_sdf_uri(propulsion.perf_file_uri, args.sdf)
    if csv_path and os.path.exists(csv_path):
        prop_gazebo = load_propeller_csv(csv_path)
        print(f"Gazebo prop CSV resolved -> {csv_path}")
    else:
        prop_gazebo = prop_provided
        if propulsion.perf_file_uri:
            print(f"WARNING: could not resolve {propulsion.perf_file_uri}; "
                  f"falling back to provided prop {prop_provided.name}")

    n_motors_gazebo = propulsion.n_motors or args.motors
    rpm_max = propulsion.max_rpm

    V_arr = np.arange(args.vmin, args.vmax + 1e-9, args.vstep)

    # Gazebo default setup: aero + integrated prop + idealised rotor
    gz_pts = cruise_sweep_gazebo(aero, mass, V_arr, prop_gazebo,
                                 n_motors=n_motors_gazebo, rpm_max=rpm_max)

    # Provided powertrain: aero + provided prop + motor + battery
    pts = cruise_sweep(aero, mass, V_arr, motor, prop_provided, battery,
                       n_motors=args.motors, soc=args.soc,
                       usable_fraction=args.usable_fraction)
    summary = cruise_summary(aero, mass, pts)

    _print_summary(aero, mass, motor, prop_provided, prop_gazebo, battery,
                   args.motors, propulsion, summary, pts, gz_pts)

    show = not args.no_show
    if args.save_aero or show:
        plot_aero(aero, args.save_aero,
                  show=False if (args.save or args.save_compare) and not show else show)
    plot_cruise(aero, mass, pts, summary, motor, prop_provided, battery,
                n_motors=args.motors, save_path=args.save, show=show)
    plot_comparison(aero, mass, gz_pts, pts, summary, motor, prop_provided,
                    prop_gazebo, battery, n_motors=args.motors,
                    rpm_max=rpm_max, save_path=args.save_compare, show=show)


if __name__ == "__main__":
    main()
