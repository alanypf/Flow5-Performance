"""Recommended ArduPilot TECS / airspeed parameters from SDF + powertrain.

Re-uses the same aerodynamic SDF model and motor/propeller/battery objects
loaded by [sdf_aero_performance.py](sdf_aero_performance.py) and converts
the cruise / climb / glide envelope into starting values for the fixed-wing
TECS controller and the surrounding airspeed / throttle / pitch limits.

The values are *starting points*. ArduPilot's TECS gains (TIME_CONST, damping,
integrator) are not derivable from a steady performance polar -- they need
flight-log tuning. The parameters that ARE derivable from this analysis are
the airspeed envelope, cruise throttle, max climb rate, min sink rate, and
the pitch authority needed to fly the predicted climb angle. Defaults are
emitted unchanged for the gain-style parameters so the file can be loaded as
a baseline without overriding hand-tuned values you may already have.

Derivation:

    AIRSPEED_MIN     = 1.2 * V_stall                # stall margin
    AIRSPEED_CRUISE  = V at minimum pack-power      # best endurance for electric
    AIRSPEED_MAX     = 0.9 * V_max                  # leave thrust headroom
    TECS_LAND_ARSPD  = 1.3 * V_stall                # standard approach margin

    TRIM_THROTTLE    = throttle at AIRSPEED_CRUISE  # from the cruise sweep
    TECS_CLMB_MAX    = max over V of (T_full - D)*V / W   (specific excess power)
    TECS_SINK_MIN    = min over V of D * V / W            (best glide sink rate)
    TECS_PITCH_MAX   = round(climb_angle_max + alpha_trim + margin), capped
    PTCH_LIM_MAX/MIN = absolute pitch attitude limits (wider than TECS_PITCH_*)

References:
    https://ardupilot.org/plane/docs/tecs-total-energy-control-system-for-speed-height-tuning-guide.html
    https://ardupilot.org/plane/docs/airspeed-config.html

Output is a Mission-Planner-compatible .param file (one "NAME,VALUE" per line)
with header comments explaining the source of each value.

Usage:
    python ardupilot_tecs_params.py \\
        --sdf model-aero-VITERNA-m.sdf \\
        --prop PER3_7x11E.txt --motor motor.xml --battery battery.xml \\
        --motors 1 --vmin 1 --vmax 60 --vstep 1 \\
        --save tecs_recommended.param
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from motor_prop_performance import (
    Battery,
    Motor,
    Propeller,
    load_battery,
    load_motor,
    load_propeller,
    solve_operating_point,
)
from sdf_aero_performance import (
    AeroModel,
    CruisePoint,
    CD_of_alpha,
    CL_of_alpha,
    StallSpeeds,
    alpha_trim,
    cruise_summary,
    cruise_sweep,
    load_sdf_model,
    stall_speed,
)


GRAVITY = 9.81


# ---------------------------------------------------------------------------
# Climb / glide envelope (full-throttle thrust vs drag at each airspeed)
# ---------------------------------------------------------------------------


@dataclass
class ClimbEnvelope:
    V_best_roc: float = math.nan       # m/s, gives max climb rate
    max_roc: float = math.nan          # m/s, specific excess power / W
    V_best_angle: float = math.nan     # m/s, gives steepest climb
    max_angle_deg: float = math.nan    # deg, sin^-1((T-D)/W)
    alpha_at_best_angle_deg: float = math.nan


def climb_envelope(aero: AeroModel, mass: float, V_arr: Iterable[float],
                   motor: Motor, prop: Propeller, battery: Battery,
                   n_motors: int, soc: float = 1.0) -> ClimbEnvelope:
    """Sweep airspeed at full throttle; track max climb rate and steepest climb."""
    W = mass * GRAVITY
    out = ClimbEnvelope()
    best_roc = -math.inf
    best_angle = -math.inf
    for V in V_arr:
        V = float(V)
        if V <= 0:
            continue
        op = solve_operating_point(motor, prop, battery, 1.0, V, soc=soc)
        if op is None:
            continue
        T_total = op.thrust_N * n_motors
        CL_req = 2.0 * W / (aero.rho * aero.area * V * V)
        a = alpha_trim(aero, CL_req)
        if not math.isfinite(a):
            continue
        CD = float(CD_of_alpha(aero, a))
        D = 0.5 * aero.rho * V * V * aero.area * CD
        if T_total <= D:
            continue
        excess = T_total - D
        roc = excess * V / W
        if roc > best_roc:
            best_roc = roc
            out.V_best_roc = V
            out.max_roc = roc
        sin_gamma = max(-1.0, min(1.0, excess / W))
        gamma_deg = math.degrees(math.asin(sin_gamma))
        if gamma_deg > best_angle:
            best_angle = gamma_deg
            out.V_best_angle = V
            out.max_angle_deg = gamma_deg
            out.alpha_at_best_angle_deg = math.degrees(a)
    return out


@dataclass
class ClimbAtSpeed:
    """Steady-climb force balance at one (airspeed, throttle) point."""
    V: float = math.nan
    throttle: float = math.nan
    thrust_total_N: float = math.nan
    drag_N: float = math.nan
    alpha_deg: float = math.nan
    ROC: float = math.nan
    gamma_deg: float = math.nan
    pitch_attitude_deg: float = math.nan
    feasible: bool = False


def climb_at(aero: AeroModel, mass: float, V: float,
             motor: Motor, prop: Propeller, battery: Battery,
             n_motors: int, throttle: float = 1.0,
             soc: float = 1.0) -> ClimbAtSpeed:
    """Steady-state climb at a specified airspeed and throttle setting.

    Uses the shallow-climb force balance:
        L = W cos(γ) ≈ W     (cos γ ≈ 1 for moderate γ)
        T − D = W sin(γ)
        ROC = V sin(γ) = (T − D) V / W
        θ   = γ + α_trim     (aircraft pitch attitude)

    This is the rate / pitch that ArduPilot's TECS guide says
    TECS_CLMB_MAX and PTCH_LIM_MAX_DEG must respect when V = AIRSPEED_CRUISE
    and throttle = THR_MAX/100.
    """
    if V <= 0 or not math.isfinite(V):
        return ClimbAtSpeed(V=V, throttle=throttle)
    W = mass * GRAVITY
    CL_req = 2.0 * W / (aero.rho * aero.area * V * V)
    a = alpha_trim(aero, CL_req)
    if not math.isfinite(a):
        return ClimbAtSpeed(V=V, throttle=throttle)
    CD = float(CD_of_alpha(aero, a))
    D = 0.5 * aero.rho * V * V * aero.area * CD
    op = solve_operating_point(motor, prop, battery, throttle, V, soc=soc)
    if op is None:
        return ClimbAtSpeed(V=V, throttle=throttle, drag_N=D,
                            alpha_deg=math.degrees(a))
    T_total = op.thrust_N * n_motors
    sin_gamma = max(-1.0, min(1.0, (T_total - D) / W))
    gamma = math.asin(sin_gamma)
    roc = V * sin_gamma
    return ClimbAtSpeed(
        V=V, throttle=throttle, thrust_total_N=T_total, drag_N=D,
        alpha_deg=math.degrees(a),
        ROC=roc, gamma_deg=math.degrees(gamma),
        pitch_attitude_deg=math.degrees(gamma + a),
        feasible=T_total > D,
    )


def min_sink_rate(pts: list[CruisePoint], mass: float) -> tuple[float, float]:
    """Best-glide sink rate. With T = 0 in steady flight, sink = D*V/W."""
    W = mass * GRAVITY
    best = math.inf
    V_best = math.nan
    for p in pts:
        if not p.feasible or not math.isfinite(p.drag_N):
            continue
        sink = p.drag_N * p.V / W
        if sink < best:
            best = sink
            V_best = p.V
    if not math.isfinite(best):
        return math.nan, math.nan
    return best, V_best


def throttle_at_speed(pts: list[CruisePoint], V_target: float) -> float:
    """Interp the trim throttle (0..1) at V_target from the cruise sweep."""
    if not math.isfinite(V_target):
        return math.nan
    Vs, thrs = [], []
    for p in pts:
        if p.feasible and math.isfinite(p.throttle):
            Vs.append(p.V)
            thrs.append(p.throttle)
    if not Vs:
        return math.nan
    Vs_arr = np.array(Vs)
    thrs_arr = np.array(thrs)
    order = np.argsort(Vs_arr)
    return float(np.interp(V_target, Vs_arr[order], thrs_arr[order]))


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------


@dataclass
class TecsRecommendation:
    # --- diagnostics (not ArduPilot parameters; printed for traceability) ---
    V_stall_mps: float = math.nan         # blended / simulator-accurate
    V_stall_linear_mps: float = math.nan  # textbook (CL0 + CLa*alpha_stall)
    V_max_mps: float = math.nan
    V_best_LD_mps: float = math.nan
    V_min_power_mps: float = math.nan
    V_best_range_mps: float = math.nan
    V_best_climb_mps: float = math.nan
    V_best_glide_mps: float = math.nan
    max_climb_rate_mps: float = math.nan
    min_sink_rate_mps: float = math.nan
    max_climb_angle_deg: float = math.nan
    cruise_throttle_frac: float = math.nan
    # --- ArduPilot parameters (units in attribute names) ---
    params: dict[str, float] = field(default_factory=dict)
    # full stall-speed bundle (kept for plotting / traceability)
    stall: StallSpeeds | None = None
    # climb performance at AIRSPEED_CRUISE / THR_MAX (drives TECS_CLMB_MAX
    # and PTCH_LIM_MAX_DEG per the ArduPilot TECS tuning guide)
    cruise_climb: ClimbAtSpeed | None = None
    # which limit was binding when sizing TECS_CLMB_MAX (thrust vs pitch cap)
    climb_binding: str = ""


def _round1(x: float) -> float:
    return round(float(x), 1) if math.isfinite(x) else math.nan


def _round2(x: float) -> float:
    return round(float(x), 2) if math.isfinite(x) else math.nan


def recommend_tecs(aero: AeroModel, mass: float, motor: Motor,
                   prop: Propeller, battery: Battery, n_motors: int,
                   V_arr: np.ndarray, soc: float = 1.0,
                   usable_fraction: float = 0.8,
                   cruise_objective: str = "min_power",
                   thr_max_pct: int = 100,
                   pitch_margin_deg: float = 5.0,
                   pitch_cap_deg: float = 30.0,
                   ) -> tuple[TecsRecommendation, list[CruisePoint]]:
    """Compute recommended TECS / airspeed parameters from steady-state sweeps.

    cruise_objective:
        "min_power" -- AIRSPEED_CRUISE = V at minimum pack power (best endurance)
        "best_ld"   -- AIRSPEED_CRUISE = V at best L/D (best aero range)
        "best_range"-- AIRSPEED_CRUISE = V maximising actual battery range

    thr_max_pct:
        THR_MAX value (0..100). Caps the throttle available to TECS and
        therefore the climb rate the airframe can sustain at AIRSPEED_CRUISE,
        which in turn caps TECS_CLMB_MAX and PTCH_LIM_MAX_DEG.

    pitch_margin_deg:
        Headroom added on top of the pitch attitude needed to fly
        TECS_CLMB_MAX at AIRSPEED_CRUISE, before setting PTCH_LIM_MAX_DEG.

    pitch_cap_deg:
        Maximum allowed steady-climb pitch attitude (deg). For overpowered
        airframes the thrust-limited climb angle can exceed practical FW
        pitch attitudes (>30°); when that happens TECS_CLMB_MAX is reduced
        to whatever climb fits inside pitch_cap_deg, keeping
        TECS_CLMB_MAX / AIRSPEED_CRUISE / PTCH_LIM_MAX_DEG mutually
        consistent. Set higher (e.g. 45) for aerobatic airframes.
    """
    pts = cruise_sweep(aero, mass, V_arr, motor, prop, battery,
                       n_motors=n_motors, soc=soc,
                       usable_fraction=usable_fraction)
    summary = cruise_summary(aero, mass, pts)
    climb = climb_envelope(aero, mass, V_arr, motor, prop, battery,
                           n_motors, soc=soc)
    sink_min, V_sink = min_sink_rate(pts, mass)
    stalls = stall_speed(aero, mass)

    V_stall = summary.V_stall
    V_max = summary.V_max

    if cruise_objective == "best_ld":
        V_cruise = summary.V_best_LD
    elif cruise_objective == "best_range":
        V_cruise = summary.V_best_range
    else:
        V_cruise = summary.V_min_power
    if not math.isfinite(V_cruise):
        # Cascading fallbacks if the preferred objective is unreachable.
        for candidate in (summary.V_min_power, summary.V_best_LD,
                          summary.V_best_range, summary.V_best_endurance):
            if math.isfinite(candidate):
                V_cruise = candidate
                break

    aspd_min = 1.2 * V_stall if math.isfinite(V_stall) else math.nan
    aspd_max = 0.9 * V_max if math.isfinite(V_max) else math.nan
    land_aspd = max(1.3 * V_stall, aspd_min) if math.isfinite(V_stall) else math.nan

    # Cruise must live inside [AIRSPEED_MIN, AIRSPEED_MAX]; if the chosen
    # objective lands outside (common when the drag bucket is close to stall),
    # clamp upward to AIRSPEED_MIN. Drop AIRSPEED_MIN if even that is above
    # V_max -- in that degenerate case the powertrain can't sustain cruise.
    if math.isfinite(V_cruise) and math.isfinite(aspd_min) and V_cruise < aspd_min:
        V_cruise = aspd_min
    if math.isfinite(V_cruise) and math.isfinite(aspd_max) and V_cruise > aspd_max:
        V_cruise = aspd_max

    cruise_thr = throttle_at_speed(pts, V_cruise)

    # --- TECS chain: TECS_CLMB_MAX, PTCH_LIM_MAX_DEG must respect what the
    # airframe can do at AIRSPEED_CRUISE with throttle limited to THR_MAX.
    # See https://ardupilot.org/plane/docs/tecs-total-energy-control-system-for-speed-height-tuning-guide.html
    thr_max_pct = max(0, min(100, int(round(thr_max_pct))))
    thr_max_frac = thr_max_pct / 100.0
    cruise_climb = climb_at(aero, mass, V_cruise, motor, prop, battery,
                            n_motors, throttle=thr_max_frac, soc=soc)

    # Make the chain mutually consistent. Two limits can be binding:
    #
    #   (a) Thrust limit: ROC_phys = V_cruise · sin(γ_phys) where
    #       sin(γ_phys) = (T(V_cruise,THR_MAX) - D) / W
    #       Pitch attitude required: θ_phys = γ_phys + α_trim
    #
    #   (b) Pitch limit: θ_cap (user-set, default 30°)
    #       Allowed climb angle: γ_cap = θ_cap - α_trim
    #       Allowed climb rate:  ROC_cap = V_cruise · sin(γ_cap)
    #
    # We must pick the smaller of ROC_phys, ROC_cap for TECS_CLMB_MAX, and
    # set PTCH_LIM_MAX_DEG to match. Otherwise TECS will demand a climb
    # the airframe can't fly within its pitch envelope, and the result is
    # airspeed droop or over-speed.
    if cruise_climb.feasible and cruise_climb.ROC > 0:
        pitch_phys = cruise_climb.pitch_attitude_deg
        if pitch_phys > pitch_cap_deg:
            gamma_allowed_deg = max(0.0, pitch_cap_deg - cruise_climb.alpha_deg)
            roc_allowed = V_cruise * math.sin(math.radians(gamma_allowed_deg))
            tecs_clmb_max = max(_round1(roc_allowed), 0.5)
            tecs_pitch_max = int(math.ceil(pitch_cap_deg))
            climb_binding = (f"pitch-limited at θ_cap={pitch_cap_deg:.0f}° "
                             f"(thrust would give {cruise_climb.ROC:.1f} m/s "
                             f"needing θ={pitch_phys:.1f}°)")
        else:
            tecs_clmb_max = _round1(cruise_climb.ROC)
            tecs_pitch_max = int(max(10, math.ceil(pitch_phys)))
            climb_binding = (f"thrust-limited at AIRSPEED_CRUISE / THR_MAX "
                             f"(needs θ={pitch_phys:.1f}°)")
        ptch_lim_max = int(tecs_pitch_max + int(round(pitch_margin_deg)))
    else:
        tecs_clmb_max = 1.0
        tecs_pitch_max = 15
        ptch_lim_max = 25
        climb_binding = "infeasible at AIRSPEED_CRUISE / THR_MAX"
    tecs_pitch_min = -15
    ptch_lim_min = -25

    # TECS_SINK_MAX: practical descent limit (m/s). No closed-form derivation
    # from steady performance -- defaults to a conservative value scaled with
    # the airframe's drag bucket. Use 1.5 x best-glide sink as a soft cap so a
    # slow / draggy airframe gets a smaller value than a slick one.
    if math.isfinite(sink_min) and sink_min > 0:
        tecs_sink_max = max(3.0, min(8.0, round(1.5 * sink_min * 2) / 2))
    else:
        tecs_sink_max = 5.0

    params: dict[str, float] = {
        # ----- airspeed envelope (newer m/s names) -----
        "AIRSPEED_MIN":     _round1(aspd_min),
        "AIRSPEED_CRUISE":  _round1(V_cruise),
        "AIRSPEED_MAX":     _round1(aspd_max),
        # ----- legacy airspeed names (kept for older firmware) -----
        "ARSPD_FBW_MIN":    _round1(aspd_min),
        "ARSPD_FBW_MAX":    _round1(aspd_max),
        "TRIM_ARSPD_CM":    int(round(V_cruise * 100.0)) if math.isfinite(V_cruise) else 0,
        # ----- throttle -----
        "TRIM_THROTTLE":    int(round(100.0 * cruise_thr)) if math.isfinite(cruise_thr) else 50,
        "THR_MIN":          0,
        "THR_MAX":          thr_max_pct,
        "THR_SLEWRATE":     100,
        # ----- TECS energy controller -----
        "TECS_CLMB_MAX":    tecs_clmb_max,
        "TECS_SINK_MIN":    _round2(sink_min) if math.isfinite(sink_min) else 2.0,
        "TECS_SINK_MAX":    float(tecs_sink_max),
        "TECS_LAND_ARSPD":  _round1(land_aspd),
        "TECS_LAND_THR":    0,
        "TECS_PITCH_MAX":   tecs_pitch_max,
        "TECS_PITCH_MIN":   tecs_pitch_min,
        # ----- TECS gain-style params: ArduPilot defaults; tune in flight -----
        "TECS_SPDWEIGHT":   1.0,
        "TECS_TIME_CONST":  5.0,
        "TECS_THR_DAMP":    0.5,
        "TECS_INTEG_GAIN":  0.3,
        "TECS_PTCH_DAMP":   0.3,
        "TECS_RLL2THR":     10.0,
        "TECS_VERT_ACC":    7.0,
        # ----- attitude / stall safety -----
        "PTCH_LIM_MAX_DEG": int(ptch_lim_max),
        "PTCH_LIM_MIN_DEG": int(ptch_lim_min),
        "STALL_PREVENTION": 1,
    }

    rec = TecsRecommendation(
        V_stall_mps=V_stall,
        V_stall_linear_mps=summary.V_stall_linear,
        V_max_mps=V_max,
        V_best_LD_mps=summary.V_best_LD,
        V_min_power_mps=summary.V_min_power,
        V_best_range_mps=summary.V_best_range,
        V_best_climb_mps=climb.V_best_roc,
        V_best_glide_mps=V_sink,
        max_climb_rate_mps=climb.max_roc,
        min_sink_rate_mps=sink_min,
        max_climb_angle_deg=climb.max_angle_deg,
        cruise_throttle_frac=cruise_thr,
        params=params,
        stall=stalls,
        cruise_climb=cruise_climb,
        climb_binding=climb_binding,
    )
    return rec, pts


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


PARAM_NOTES: dict[str, str] = {
    "AIRSPEED_MIN":     "1.2 * V_stall (autopilot lower speed limit, m/s)",
    "AIRSPEED_CRUISE":  "trim/cruise airspeed at min pack power (m/s)",
    "AIRSPEED_MAX":     "0.9 * V_max (autopilot upper speed limit, m/s)",
    "ARSPD_FBW_MIN":    "legacy alias for AIRSPEED_MIN (pre-4.4 firmware)",
    "ARSPD_FBW_MAX":    "legacy alias for AIRSPEED_MAX (pre-4.4 firmware)",
    "TRIM_ARSPD_CM":    "legacy AIRSPEED_CRUISE expressed in cm/s",
    "TRIM_THROTTLE":    "throttle (%) needed for level flight at cruise speed",
    "THR_MIN":          "minimum throttle in AUTO modes (%)",
    "THR_MAX":          "maximum throttle in AUTO modes (%)",
    "THR_SLEWRATE":     "throttle rate of change (% / s)",
    "TECS_CLMB_MAX":    "max sustained climb rate, full throttle (m/s)",
    "TECS_SINK_MIN":    "best-glide sink rate, throttle = 0 (m/s)",
    "TECS_SINK_MAX":    "max commanded descent rate (m/s)",
    "TECS_LAND_ARSPD":  "approach airspeed = 1.3 * V_stall (m/s)",
    "TECS_LAND_THR":    "throttle on final approach (%)",
    "TECS_PITCH_MAX":   "max pitch TECS commands in climb (deg)",
    "TECS_PITCH_MIN":   "min pitch TECS commands in descent (deg)",
    "TECS_SPDWEIGHT":   "1=balance speed/height, 2=hold speed only (default 1)",
    "TECS_TIME_CONST":  "energy filter time constant (s) -- tune in flight",
    "TECS_THR_DAMP":    "throttle damping gain -- tune in flight",
    "TECS_INTEG_GAIN":  "integrator gain -- tune in flight",
    "TECS_PTCH_DAMP":   "pitch damping gain -- tune in flight",
    "TECS_RLL2THR":     "throttle increase in turns (default 10)",
    "TECS_VERT_ACC":    "vertical accel limit (m/s^2, default 7)",
    "PTCH_LIM_MAX_DEG": "absolute max nose-up attitude (deg)",
    "PTCH_LIM_MIN_DEG": "absolute max nose-down attitude (deg)",
    "STALL_PREVENTION": "1=enable airspeed-based stall prevention",
}


PARAM_ORDER: list[str] = [
    "AIRSPEED_MIN", "AIRSPEED_CRUISE", "AIRSPEED_MAX",
    "ARSPD_FBW_MIN", "ARSPD_FBW_MAX", "TRIM_ARSPD_CM",
    "TRIM_THROTTLE", "THR_MIN", "THR_MAX", "THR_SLEWRATE",
    "TECS_CLMB_MAX", "TECS_SINK_MIN", "TECS_SINK_MAX",
    "TECS_LAND_ARSPD", "TECS_LAND_THR",
    "TECS_PITCH_MAX", "TECS_PITCH_MIN",
    "TECS_SPDWEIGHT", "TECS_TIME_CONST", "TECS_THR_DAMP",
    "TECS_INTEG_GAIN", "TECS_PTCH_DAMP",
    "TECS_RLL2THR", "TECS_VERT_ACC",
    "PTCH_LIM_MAX_DEG", "PTCH_LIM_MIN_DEG",
    "STALL_PREVENTION",
]


def format_param_value(v: float) -> str:
    """Format like Mission Planner: integers without a decimal, floats with up to 3."""
    if isinstance(v, int) or (isinstance(v, float) and v.is_integer()):
        return str(int(v))
    return f"{v:.3f}".rstrip("0").rstrip(".")


def write_param_file(rec: TecsRecommendation, path: str,
                     header_lines: list[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in header_lines:
            f.write(f"# {line}\n")
        f.write("#\n")
        for k in PARAM_ORDER:
            if k not in rec.params:
                continue
            v = rec.params[k]
            note = PARAM_NOTES.get(k, "")
            f.write(f"{k},{format_param_value(v)}")
            if note:
                f.write(f"   # {note}")
            f.write("\n")


def plot_stall_comparison(aero: AeroModel, mass: float, vs: StallSpeeds,
                          V_max: float | None, save_path: str | None,
                          show: bool) -> None:
    """Two-panel illustration of the two stall-speed definitions.

    Left:  CL vs alpha. Plots both the linear pre-stall extrapolation
           (CL0 + CLa*alpha, the textbook polar) and the AdvancedLiftDrag
           blended curve (linear * (1-sigma) + flat-plate * sigma) that the
           simulator actually evaluates. Markers show CL_max under each
           definition and the alpha where each CL_max occurs.

    Right: CL required for L = W as a function of airspeed (CL_req = 2W/(rho S V^2))
           with horizontal lines at each CL_max and vertical lines at the
           corresponding V_stall. The shaded region is where CL_req exceeds
           the blended CL_max -- the simulator cannot trim there even if a
           textbook polar would still produce lift.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    aS_deg = math.degrees(aero.alpha_stall)
    aP_deg = math.degrees(vs.alpha_at_blended_peak_rad)

    alpha_deg = np.linspace(-5.0, aS_deg + 20.0, 800)
    a_rad = np.deg2rad(alpha_deg)
    CL_blend = CL_of_alpha(aero, a_rad)
    CL_lin = aero.CL0 + aero.CLa * a_rad

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    # --- Left: CL vs alpha ----------------------------------------------
    ax1.plot(alpha_deg, CL_lin, color="C3", linestyle="--", linewidth=1.6,
             alpha=0.75, label="Linear: CL0 + CLa·α (textbook)")
    ax1.plot(alpha_deg, CL_blend, color="C0", linewidth=2.2,
             label="ALD blended (simulator)")
    ax1.axvline(aS_deg, color="grey", linestyle=":", linewidth=0.9,
                label=f"α_stall = {aS_deg:.1f}°")

    # Linear CL_max marker
    ax1.plot(aS_deg, vs.CL_max_linear, marker="o", color="C3", markersize=9,
             linestyle="none",
             label=f"Linear CL_max = {vs.CL_max_linear:.3f}")
    ax1.plot([aS_deg, aS_deg], [0.0, vs.CL_max_linear], color="C3",
             linestyle=":", linewidth=0.7)
    # Analytical (closed-form ALD at alpha_stall, sigma=0.5) marker
    ax1.plot(aS_deg, vs.CL_max_analytical, marker="D", color="C2",
             markersize=10, linestyle="none",
             label=f"Analytical CL_max = {vs.CL_max_analytical:.3f} "
                   f"(σ=½ at α_stall)")
    ax1.plot([aS_deg, aS_deg], [0.0, vs.CL_max_analytical], color="C2",
             linestyle=":", linewidth=0.7)
    # Blended CL_max marker
    ax1.plot(aP_deg, vs.CL_max_blended, marker="s", color="C0", markersize=9,
             linestyle="none",
             label=f"Blended CL_max = {vs.CL_max_blended:.3f} @ α={aP_deg:.1f}°")
    ax1.plot([aP_deg, aP_deg], [0.0, vs.CL_max_blended], color="C0",
             linestyle=":", linewidth=0.7)

    ax1.set_xlabel("α [deg]")
    ax1.set_ylabel("CL")
    ax1.set_title("CL(α): textbook vs simulator")
    ax1.grid(alpha=0.3)
    ax1.set_ylim(bottom=min(0.0, float(CL_blend.min())))
    ax1.legend(fontsize=8, loc="upper left")

    # --- Right: CL_req vs V ---------------------------------------------
    W = mass * 9.81
    V_top = V_max if V_max and math.isfinite(V_max) else 2.0 * vs.blended
    V_lo = max(0.4 * vs.linear, 1.0)
    V_arr = np.linspace(V_lo, V_top, 600)
    CL_req = 2.0 * W / (aero.rho * aero.area * V_arr ** 2)

    ax2.plot(V_arr, CL_req, color="k", linewidth=2.0,
             label=r"CL required (L = W): $2W/(\rho S V^2)$")
    ax2.axhline(vs.CL_max_linear, color="C3", linestyle="--", linewidth=1.5,
                label=f"Linear CL_max = {vs.CL_max_linear:.3f}")
    ax2.axhline(vs.CL_max_analytical, color="C2", linestyle="-.", linewidth=1.5,
                label=f"Analytical CL_max = {vs.CL_max_analytical:.3f}")
    ax2.axhline(vs.CL_max_blended, color="C0", linestyle="-", linewidth=1.5,
                label=f"Blended CL_max = {vs.CL_max_blended:.3f}")
    ax2.axvline(vs.linear, color="C3", linestyle=":", linewidth=0.9)
    ax2.axvline(vs.analytical, color="C2", linestyle=":", linewidth=0.9)
    ax2.axvline(vs.blended, color="C0", linestyle=":", linewidth=0.9)
    ax2.plot(vs.linear, vs.CL_max_linear, marker="o", color="C3",
             markersize=11, linestyle="none",
             label=f"V_stall(linear)     = {vs.linear:.1f} m/s")
    ax2.plot(vs.analytical, vs.CL_max_analytical, marker="D", color="C2",
             markersize=11, linestyle="none",
             label=f"V_stall(analytical) = {vs.analytical:.1f} m/s")
    ax2.plot(vs.blended, vs.CL_max_blended, marker="s", color="C0",
             markersize=11, linestyle="none",
             label=f"V_stall(blended)    = {vs.blended:.1f} m/s")

    # Shade the region where the simulator cannot trim (CL_req > CL_max_blended)
    mask = CL_req > vs.CL_max_blended
    if np.any(mask):
        ax2.fill_between(V_arr, vs.CL_max_blended, CL_req, where=mask,
                         color="C0", alpha=0.12,
                         label="Simulator cannot trim")

    ax2.set_xlabel("V [m/s]")
    ax2.set_ylabel("CL")
    ax2.set_title("Level-flight CL_required vs airspeed")
    ax2.grid(alpha=0.3)
    cl_top = max(2.2 * vs.CL_max_linear, 1.0)
    ax2.set_ylim(0.0, cl_top)
    ax2.set_xlim(V_lo, V_top)
    ax2.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        f"Stall-speed comparison  --  m={mass:.2f} kg, S={aero.area:.3f} m², "
        f"AR={aero.AR:.2f}, α_stall={aS_deg:.1f}°  (M={aero.M:.0f})",
        fontsize=11,
    )
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def print_recommendation(rec: TecsRecommendation) -> None:
    print()
    print("=== Performance-derived speeds & rates ===")
    print(f"  V_stall (sim-accurate)  : {rec.V_stall_mps:6.2f} m/s   "
          "(blended ALD CL_max -- used for AIRSPEED_MIN)")
    print(f"  V_stall (textbook linear): {rec.V_stall_linear_mps:6.2f} m/s   "
          "(CL0+CLa*α_stall; info only)")
    if rec.stall is not None:
        print(f"  V_stall (analytical)    : {rec.stall.analytical:6.2f} m/s   "
              "(closed-form ALD at α_stall, σ=½; info only)")
    print(f"  V_max (level)        : {rec.V_max_mps:6.2f} m/s")
    print(f"  V at best L/D        : {rec.V_best_LD_mps:6.2f} m/s   (best aero range)")
    print(f"  V at min pack power  : {rec.V_min_power_mps:6.2f} m/s   (best endurance)")
    print(f"  V at best range      : {rec.V_best_range_mps:6.2f} m/s   (battery range)")
    print(f"  V at best climb rate : {rec.V_best_climb_mps:6.2f} m/s")
    print(f"  V at min sink (glide): {rec.V_best_glide_mps:6.2f} m/s")
    print(f"  Max climb rate (any V): {rec.max_climb_rate_mps:6.2f} m/s   "
          f"@ V={rec.V_best_climb_mps:.1f} m/s, full throttle  (diagnostic)")
    print(f"  Max climb angle       : {rec.max_climb_angle_deg:6.2f} deg")
    print(f"  Min sink rate (glide) : {rec.min_sink_rate_mps:6.2f} m/s")
    print(f"  Cruise throttle       : {rec.cruise_throttle_frac * 100.0:6.1f} %")

    cc = rec.cruise_climb
    if cc is not None:
        print()
        print("=== TECS chain (THR_MAX → climb at AIRSPEED_CRUISE → pitch limit) ===")
        print(f"  AIRSPEED_CRUISE        : {cc.V:6.2f} m/s")
        print(f"  THR_MAX                : {int(round(cc.throttle * 100)):3d} %")
        print(f"  Thrust available       : {cc.thrust_total_N:6.2f} N")
        print(f"  Drag at trim α         : {cc.drag_N:6.2f} N")
        print(f"  α at cruise (L=W·cosγ) : {cc.alpha_deg:6.2f} deg")
        print(f"  γ = asin((T−D)/W)      : {cc.gamma_deg:6.2f} deg")
        print(f"  ROC = V·sinγ           : {cc.ROC:6.2f} m/s   "
              "→ TECS_CLMB_MAX")
        print(f"  Pitch attitude (γ + α) : {cc.pitch_attitude_deg:6.2f} deg  "
              "→ PTCH_LIM_MAX_DEG floor")
        if rec.climb_binding:
            print(f"  Binding constraint     : {rec.climb_binding}")
        print(f"  -> TECS_CLMB_MAX={rec.params.get('TECS_CLMB_MAX'):.1f} m/s, "
              f"TECS_PITCH_MAX={rec.params.get('TECS_PITCH_MAX')}°, "
              f"PTCH_LIM_MAX_DEG={rec.params.get('PTCH_LIM_MAX_DEG')}°")

    print()
    print("=== Recommended ArduPilot parameters ===")
    name_w = max(len(k) for k in rec.params)
    for k in PARAM_ORDER:
        if k not in rec.params:
            continue
        v = rec.params[k]
        note = PARAM_NOTES.get(k, "")
        print(f"  {k:<{name_w}}  {format_param_value(v):>8}   # {note}")
    print()
    print("Notes:")
    print("  * TECS_TIME_CONST and the *_DAMP / *_GAIN values are firmware")
    print("    defaults -- they require in-flight tuning, not steady-state")
    print("    analysis.")
    print("  * Both new (AIRSPEED_*) and legacy (ARSPD_FBW_*, TRIM_ARSPD_CM)")
    print("    names are emitted; load only the set your firmware uses.")
    print("  * Always re-validate AIRSPEED_MIN against the actual stall in")
    print("    flight tests before relying on it.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--sdf", default="model-aero-VITERNA-m.sdf",
                   help="SDF model with AdvancedLiftDrag plugin")
    p.add_argument("--prop", default="PER3_7x11E.txt",
                   help="APC-format propeller .txt")
    p.add_argument("--motor", default="motor.xml")
    p.add_argument("--battery", default="battery.xml")
    p.add_argument("--motors", type=int, default=1,
                   help="Number of motors providing thrust (default 1 for FW)")
    p.add_argument("--mass", type=float, default=None,
                   help="Override mass [kg] (default: sum SDF link masses)")
    p.add_argument("--vmin", type=float, default=1.0)
    p.add_argument("--vmax", type=float, default=60.0)
    p.add_argument("--vstep", type=float, default=1.0)
    p.add_argument("--soc", type=float, default=1.0)
    p.add_argument("--usable-fraction", type=float, default=0.8)
    p.add_argument("--cruise-objective",
                   choices=("min_power", "best_ld", "best_range"),
                   default="min_power",
                   help="Which speed to use for AIRSPEED_CRUISE")
    p.add_argument("--thr-max", type=int, default=100,
                   help="THR_MAX percent (default 100). Caps the climb rate "
                        "the airframe can sustain at AIRSPEED_CRUISE and "
                        "therefore TECS_CLMB_MAX / PTCH_LIM_MAX_DEG.")
    p.add_argument("--pitch-margin", type=float, default=5.0,
                   help="Degrees of pitch headroom over the steady-climb "
                        "attitude when sizing PTCH_LIM_MAX_DEG (default 5).")
    p.add_argument("--pitch-cap", type=float, default=30.0,
                   help="Practical max steady-climb pitch attitude (deg, "
                        "default 30). If thrust would push pitch above this, "
                        "TECS_CLMB_MAX is reduced to fit instead.")
    p.add_argument("--save", default=None,
                   help="Write recommended .param file here")
    p.add_argument("--save-stall", default=None,
                   help="Save the stall-speed comparison figure here "
                        "(blended vs textbook CL_max)")
    p.add_argument("--show", action="store_true",
                   help="Display plots interactively (default: save only)")
    args = p.parse_args()

    aero, sdf_mass, propulsion, _info = load_sdf_model(args.sdf)
    mass = args.mass if args.mass is not None else sdf_mass

    prop = load_propeller(args.prop)
    motor = load_motor(args.motor)
    battery = load_battery(args.battery)

    V_arr = np.arange(args.vmin, args.vmax + 1e-9, args.vstep)

    rec, _pts = recommend_tecs(
        aero, mass, motor, prop, battery,
        n_motors=args.motors, V_arr=V_arr,
        soc=args.soc, usable_fraction=args.usable_fraction,
        cruise_objective=args.cruise_objective,
        thr_max_pct=args.thr_max,
        pitch_margin_deg=args.pitch_margin,
        pitch_cap_deg=args.pitch_cap,
    )

    print(f"Inputs : sdf={args.sdf}  prop={prop.name}  motor={motor.name}")
    print(f"         battery={battery.series}S{battery.parallel}P "
          f"{battery.chemistry}  mass={mass:.3f} kg  motors={args.motors}")
    print(f"         cruise_objective={args.cruise_objective}")
    print_recommendation(rec)

    if args.save_stall or args.show:
        plot_stall_comparison(aero, mass, rec.stall, V_max=rec.V_max_mps,
                              save_path=args.save_stall, show=args.show)
        if args.save_stall:
            print(f"Wrote {args.save_stall}")

    if args.save:
        header = [
            "ArduPilot fixed-wing TECS / airspeed starting points",
            f"Source SDF      : {os.path.basename(args.sdf)}",
            f"Propeller       : {prop.name}",
            f"Motor           : {motor.name} x {args.motors}",
            f"Battery         : {battery.series}S{battery.parallel}P "
            f"{battery.chemistry}  V_nom={battery.V_nominal:.2f}V  "
            f"cap={battery.capacity_Ah:.2f}Ah",
            f"Mass            : {mass:.3f} kg",
            f"V_stall (blended): {rec.V_stall_mps:.2f} m/s  "
            "(simulator-accurate CL_max; used to size AIRSPEED_MIN)",
            f"V_stall (linear) : {rec.V_stall_linear_mps:.2f} m/s  "
            "(textbook CL0+CLa*alpha_stall; info only)",
            f"V_max           : {rec.V_max_mps:.2f} m/s",
            f"V_cruise (used) : {rec.params.get('AIRSPEED_CRUISE', 0.0):.2f} m/s "
            f"({args.cruise_objective})",
            f"Cruise throttle : {rec.cruise_throttle_frac * 100.0:.1f} %",
            f"Max climb rate  : {rec.max_climb_rate_mps:.2f} m/s "
            f"@ {rec.V_best_climb_mps:.1f} m/s  (diagnostic; absolute capability)",
            (f"TECS chain      : AIRSPEED_CRUISE={rec.cruise_climb.V:.1f} m/s, "
             f"THR_MAX={int(round(rec.cruise_climb.throttle*100))}%, "
             f"thrust-limited ROC={rec.cruise_climb.ROC:.2f} m/s at "
             f"θ={rec.cruise_climb.pitch_attitude_deg:.1f}°"
             if rec.cruise_climb and rec.cruise_climb.feasible
             else "TECS chain      : not feasible at AIRSPEED_CRUISE / THR_MAX"),
            (f"Binding         : {rec.climb_binding}"
             if rec.climb_binding else "Binding         : n/a"),
            (f"-> TECS_CLMB_MAX={rec.params.get('TECS_CLMB_MAX')} m/s, "
             f"TECS_PITCH_MAX={rec.params.get('TECS_PITCH_MAX')}°, "
             f"PTCH_LIM_MAX_DEG={rec.params.get('PTCH_LIM_MAX_DEG')}°"),
            f"Min sink rate   : {rec.min_sink_rate_mps:.2f} m/s "
            f"@ {rec.V_best_glide_mps:.1f} m/s",
            "",
            "Format: NAME,VALUE  -- compatible with Mission Planner /",
            "QGroundControl 'Load from file'. Lines starting with '#' are",
            "comments. Numeric values follow the standard ArduPilot units.",
        ]
        write_param_file(rec, args.save, header)
        print(f"Wrote {args.save}")


if __name__ == "__main__":
    main()
