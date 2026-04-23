"""Coupled airframe + propulsion performance estimator for tailsitters.

Combines the Flow5 aerodynamic polar (parsed by plot_flow5) with the
motor/prop/battery torque-balance solver (from motor_prop_performance)
to compute the quantities that matter for a tailsitter:

    Hover   — trim throttle, rpm, per-motor current, electrical power,
              hover endurance from usable battery energy.
    Cruise  — at each airspeed V the required thrust equals the airframe
              drag D(V) (assuming L = W). Bisects throttle to hit that
              thrust, then reports endurance (h) and range (km).
    Climb   — at full throttle, RoC(V) = V·(T_max − D)/W.

Assumptions worth knowing:
  * Steady level flight: lift = weight, no pitch offset. CD is looked up
    from the pre-stall branch of the polar at CL_req.
  * Throttle is PWM duty on battery terminal voltage (same model as
    motor_prop_performance.solve_operating_point).
  * Multiple motors are treated identically; total thrust = N · single.
  * Usable battery energy = V_nominal · capacity · usable_fraction (0.8
    by default — LiPo typical DoD before sag dominates).

Usage:
    py -3 performance.py PER3_7x15E.txt motor.xml battery.xml polar.txt \\
        --plane plane.xml --n-motors 1 --vmin 5 --vmax 40 --vstep 1
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from motor_prop_performance import (
    Battery,
    Motor,
    OperatingPoint,
    Propeller,
    load_battery,
    load_motor,
    load_propeller,
    solve_operating_point,
)
from plot_flow5 import load_plane_xml, load_polar


# ---------------------------------------------------------------------------
# Airframe drag from polar
# ---------------------------------------------------------------------------


def _polar_prestall(polar: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (CL, CD) sorted by CL over the pre-stall branch of the polar."""
    alpha = np.asarray(polar["alpha"], dtype=float)
    cl = np.asarray(polar["cl"], dtype=float)
    cd = np.asarray(polar["cd"], dtype=float)
    order = np.argsort(alpha)
    alpha, cl, cd = alpha[order], cl[order], cd[order]
    i_stall = int(np.argmax(cl))
    cl_pre = cl[: i_stall + 1]
    cd_pre = cd[: i_stall + 1]
    o2 = np.argsort(cl_pre)
    return cl_pre[o2], cd_pre[o2]


def airframe_drag(polar: dict, mass: float, area: float, rho: float,
                  g: float, V: np.ndarray) -> np.ndarray:
    """Drag [N] for steady level flight at each airspeed in V (m/s).

    L = W so CL_req = 2W/(ρV²S); CD is interpolated from the pre-stall
    branch. Returns NaN where the airframe cannot hold weight (V below
    stall or above a polar that doesn't reach low-enough CL).
    """
    cl_m, cd_m = _polar_prestall(polar)
    W = mass * g
    V = np.asarray(V, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        cl_req = np.where(V > 0, 2.0 * W / (rho * area * V**2), np.inf)
    cl_lo, cl_hi = float(cl_m[0]), float(cl_m[-1])
    in_range = (cl_req >= cl_lo) & (cl_req <= cl_hi)
    cd_here = np.interp(cl_req, cl_m, cd_m)
    D = 0.5 * rho * area * V**2 * cd_here
    return np.where(in_range, D, np.nan)


def stall_speed(polar: dict, mass: float, area: float,
                rho: float, g: float) -> float:
    cl_m, _ = _polar_prestall(polar)
    cl_max = float(cl_m[-1])
    return float(np.sqrt(2.0 * mass * g / (rho * area * cl_max)))


# ---------------------------------------------------------------------------
# ISA atmosphere & altitude-dependent characteristic speeds
# ---------------------------------------------------------------------------


def isa_density(altitude_m: float) -> float:
    """ISA air density [kg/m³] for altitude below 11 000 m (troposphere)."""
    T0 = 288.15        # sea-level temperature [K]
    L = 0.0065         # lapse rate [K/m]
    rho0 = 1.225       # sea-level density [kg/m³]
    g0 = 9.80665
    R = 287.0528       # specific gas constant for air [J/(kg·K)]
    T = T0 - L * altitude_m
    return rho0 * (T / T0) ** (g0 / (L * R) - 1.0)


@dataclass
class AltitudeSpeedPoint:
    altitude_m: float
    rho: float
    V_stall: float
    V_best_range: float
    V_best_endurance: float


def altitude_speed_sweep(polar: dict, mass: float, area: float,
                         g: float = 9.81,
                         alt_min: float = 0.0,
                         alt_max: float = 3000.0,
                         alt_step: float = 100.0,
                         ) -> list[AltitudeSpeedPoint]:
    """Compute stall, best-range, and best-endurance speeds vs altitude."""
    cl_pre, cd_pre = _polar_prestall(polar)
    cl_max = float(cl_pre[-1])

    cl = np.asarray(polar["cl"], dtype=float)
    cd = np.asarray(polar["cd"], dtype=float)
    W = mass * g

    # Best range: max L/D → CL at max CL/CD
    with np.errstate(divide="ignore", invalid="ignore"):
        ld = np.where(cd > 0, cl / cd, np.nan)
    i_ld = int(np.nanargmax(ld))
    cl_range = cl[i_ld]

    # Best endurance: max CL^1.5 / CD
    with np.errstate(divide="ignore", invalid="ignore"):
        endur_metric = np.where((cd > 0) & (cl > 0), cl ** 1.5 / cd, np.nan)
    i_e = int(np.nanargmax(endur_metric))
    cl_endur = cl[i_e]

    out: list[AltitudeSpeedPoint] = []
    alt = alt_min
    while alt <= alt_max + 1e-9:
        rho = isa_density(alt)
        v_stall = float(np.sqrt(2 * W / (rho * area * cl_max)))
        v_range = float(np.sqrt(2 * W / (rho * area * cl_range))) if cl_range > 0 else float("nan")
        v_endur = float(np.sqrt(2 * W / (rho * area * cl_endur))) if cl_endur > 0 else float("nan")
        out.append(AltitudeSpeedPoint(
            altitude_m=alt, rho=rho,
            V_stall=v_stall,
            V_best_range=v_range,
            V_best_endurance=v_endur,
        ))
        alt += alt_step
    return out


def alpha_level_flight(polar: dict, mass: float, area: float, rho: float,
                       g: float, V: np.ndarray) -> np.ndarray:
    """Angle of attack [deg] for steady level flight at each V (m/s).

    Inverts the pre-stall CL(α) branch to find α such that
    CL = 2W/(ρV²S). Returns NaN where CL_req is outside the polar.
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

    W = mass * g
    V = np.asarray(V, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        cl_req = np.where(V > 0, 2.0 * W / (rho * area * V**2), np.inf)
    in_range = (cl_req >= cl_pre[0]) & (cl_req <= cl_pre[-1])
    a = np.interp(cl_req, cl_pre, a_pre)
    return np.where(in_range, a, np.nan)


# ---------------------------------------------------------------------------
# Throttle root-finder shared by hover and cruise
# ---------------------------------------------------------------------------


def _find_throttle_for_thrust(motor: Motor, prop: Propeller, battery: Battery,
                              V: float, T_req: float,
                              soc: float) -> tuple[float, OperatingPoint] | None:
    """Bisect throttle ∈ [0, 1] so that per-motor prop thrust at V equals T_req.

    Returns (throttle, operating_point) or None if full throttle can't
    reach T_req at this airspeed.
    """
    op_hi = solve_operating_point(motor, prop, battery, 1.0, V, soc=soc)
    if op_hi is None or op_hi.thrust_N < T_req:
        return None
    op_lo = solve_operating_point(motor, prop, battery, 0.0, V, soc=soc)
    if op_lo is not None and op_lo.thrust_N >= T_req:
        return (0.0, op_lo)

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
# Hover
# ---------------------------------------------------------------------------


@dataclass
class HoverPoint:
    throttle: float
    rpm: float
    thrust_per_motor_N: float
    current_per_motor_A: float
    P_elec_per_motor_W: float
    P_elec_total_W: float
    endurance_min: float
    thrust_to_weight_max: float


def solve_hover(motor: Motor, prop: Propeller, battery: Battery,
                mass: float, g: float = 9.81, n_motors: int = 1,
                soc: float = 1.0,
                usable_fraction: float = 0.8) -> HoverPoint | None:
    W = mass * g
    T_req = W / n_motors
    res = _find_throttle_for_thrust(motor, prop, battery, 0.0, T_req, soc)
    if res is None:
        return None
    t, op = res
    op_full = solve_operating_point(motor, prop, battery, 1.0, 0.0, soc=soc)
    twr = (op_full.thrust_N * n_motors) / W if op_full else float("nan")
    P_total = op.P_elec_W * n_motors
    E_Wh = battery.V_nominal * battery.capacity_Ah * usable_fraction
    endurance_min = (E_Wh / P_total) * 60.0 if P_total > 0 else float("nan")
    return HoverPoint(
        throttle=t,
        rpm=op.rpm,
        thrust_per_motor_N=op.thrust_N,
        current_per_motor_A=op.current_A,
        P_elec_per_motor_W=op.P_elec_W,
        P_elec_total_W=P_total,
        endurance_min=endurance_min,
        thrust_to_weight_max=twr,
    )


# ---------------------------------------------------------------------------
# Cruise trim sweep
# ---------------------------------------------------------------------------


@dataclass
class CruisePoint:
    V: float
    CL_req: float
    drag_N: float
    thrust_per_motor_N: float
    torque_per_motor_Nm: float
    throttle: float
    rpm: float
    current_per_motor_A: float
    P_elec_per_motor_W: float
    P_elec_total_W: float
    endurance_h: float
    range_km: float


def cruise_sweep(polar: dict, plane: dict, motor: Motor, prop: Propeller,
                 battery: Battery, V_array: np.ndarray,
                 n_motors: int = 1, soc: float = 1.0,
                 usable_fraction: float = 0.8) -> list[CruisePoint | None]:
    mass = plane["mass"]
    area = plane["area"]
    rho = plane.get("rho", 1.225)
    g = plane.get("gravity", 9.81)
    D_arr = airframe_drag(polar, mass, area, rho, g, V_array)
    E_Wh = battery.V_nominal * battery.capacity_Ah * usable_fraction

    out: list[CruisePoint | None] = []
    for V, D in zip(V_array, D_arr):
        V = float(V)
        if V <= 0 or not np.isfinite(D):
            out.append(None)
            continue
        T_per = D / n_motors
        res = _find_throttle_for_thrust(motor, prop, battery, V, T_per, soc)
        if res is None:
            out.append(None)
            continue
        t, op = res
        P_total = op.P_elec_W * n_motors
        endurance_h = E_Wh / P_total if P_total > 0 else float("nan")
        range_km = V * endurance_h * 3.6  # V[m/s] · t[h] · 3600/1000
        out.append(CruisePoint(
            V=V,
            CL_req=2.0 * mass * g / (rho * V * V * area),
            drag_N=float(D),
            thrust_per_motor_N=op.thrust_N,
            torque_per_motor_Nm=op.torque_Nm,
            throttle=t,
            rpm=op.rpm,
            current_per_motor_A=op.current_A,
            P_elec_per_motor_W=op.P_elec_W,
            P_elec_total_W=P_total,
            endurance_h=endurance_h,
            range_km=range_km,
        ))
    return out


# ---------------------------------------------------------------------------
# Climb (full throttle excess thrust → rate of climb)
# ---------------------------------------------------------------------------


@dataclass
class ClimbPoint:
    V: float
    T_max_total_N: float
    drag_N: float
    excess_thrust_N: float
    roc_ms: float
    TW_max: float          # full-throttle thrust / weight at this V
    TW_cruise: float       # drag / weight (trim thrust fraction) at this V
    rpm_max: float         # full-throttle (= max) prop rpm at this V
    torque_per_motor_Nm: float  # full-throttle (= max available) prop torque at this V


def climb_sweep(polar: dict, plane: dict, motor: Motor, prop: Propeller,
                battery: Battery, V_array: np.ndarray,
                n_motors: int = 1,
                soc: float = 1.0) -> list[ClimbPoint | None]:
    mass = plane["mass"]
    area = plane["area"]
    rho = plane.get("rho", 1.225)
    g = plane.get("gravity", 9.81)
    W = mass * g
    D_arr = airframe_drag(polar, mass, area, rho, g, V_array)

    out: list[ClimbPoint | None] = []
    for V, D in zip(V_array, D_arr):
        V = float(V)
        if V <= 0 or not np.isfinite(D):
            out.append(None)
            continue
        op = solve_operating_point(motor, prop, battery, 1.0, V, soc=soc)
        if op is None:
            out.append(None)
            continue
        T_total = op.thrust_N * n_motors
        excess = T_total - float(D)
        roc = V * excess / W  # steady climb, small angle: RoC = V(T-D)/W
        out.append(ClimbPoint(
            V=V,
            T_max_total_N=T_total,
            drag_N=float(D),
            excess_thrust_N=excess,
            roc_ms=roc,
            TW_max=T_total / W,
            TW_cruise=float(D) / W,
            rpm_max=op.rpm,
            torque_per_motor_Nm=op.torque_Nm,
        ))
    return out


# ---------------------------------------------------------------------------
# V_max — powertrain-limited maximum speed
# ---------------------------------------------------------------------------


@dataclass
class VmaxPoint:
    V: float
    drag_N: float
    thrust_total_N: float


def solve_vmax(polar: dict, plane: dict, motor: Motor, prop: Propeller,
               battery: Battery, n_motors: int = 1,
               soc: float = 1.0) -> VmaxPoint | None:
    """Find the maximum airspeed where full-throttle thrust >= drag.

    Bisects airspeed between stall speed and an upper bound where thrust
    can no longer sustain level flight.  Returns None if no feasible
    crossing exists within the polar's valid CL range.
    """
    mass = plane["mass"]
    area = plane["area"]
    rho = plane.get("rho", 1.225)
    g = plane.get("gravity", 9.81)

    vs = stall_speed(polar, mass, area, rho, g)

    # excess(V) = full-throttle total thrust − drag at V (level flight)
    def excess(V: float) -> float | None:
        D_arr = airframe_drag(polar, mass, area, rho, g, np.array([V]))
        D = float(D_arr[0])
        if not np.isfinite(D):
            return None
        op = solve_operating_point(motor, prop, battery, 1.0, V, soc=soc)
        if op is None:
            return None
        return op.thrust_N * n_motors - D

    # Find an upper bound where excess < 0.  Start from a reasonable
    # ceiling and step outward if needed.
    V_lo = vs + 0.5
    e_lo = excess(V_lo)
    if e_lo is None or e_lo < 0:
        return None  # can't even fly just above stall

    V_hi = vs + 1.0
    for _ in range(200):
        V_hi += 1.0
        e_hi = excess(V_hi)
        if e_hi is None or e_hi < 0:
            break
    else:
        return None  # thrust always exceeds drag — no crossing found

    # Bisect to find the crossing: excess(V) = 0
    for _ in range(50):
        V_mid = 0.5 * (V_lo + V_hi)
        e_mid = excess(V_mid)
        if e_mid is None or e_mid < 0:
            V_hi = V_mid
        else:
            V_lo = V_mid
        if V_hi - V_lo < 0.01:
            break

    V_max = 0.5 * (V_lo + V_hi)
    D_arr = airframe_drag(polar, mass, area, rho, g, np.array([V_max]))
    D = float(D_arr[0])
    op = solve_operating_point(motor, prop, battery, 1.0, V_max, soc=soc)
    T_total = op.thrust_N * n_motors if op else D

    return VmaxPoint(V=V_max, drag_N=D, thrust_total_N=T_total)


# ---------------------------------------------------------------------------
# Control authority (cruise, differential thrust on X-quad)
# ---------------------------------------------------------------------------


@dataclass
class ControlAuthorityPoint:
    V: float
    T_trim_per_motor_N: float
    T_max_per_motor_N: float
    dT_max_N: float            # max ΔT per motor (symmetric about trim)
    M_roll_Nm: float           # max roll moment (left/right pair differential)
    M_pitch_Nm: float          # max pitch moment (top/bottom pair differential)
    alpha_roll_rads2: float    # max roll angular acceleration [rad/s²]
    alpha_pitch_rads2: float   # max pitch angular acceleration [rad/s²]
    cp_roll: float             # roll control power ratio M / (q·S·c)
    cp_pitch: float            # pitch control power ratio M / (q·S·c)
    M_aero_pitch_Nm: float     # aerodynamic pitching moment about CG (L × (x_cg - x_ac))
    M_pitch_net_Nm: float      # net pitch authority after trimming aero moment
    static_margin: float       # (x_ac - x_cg) / chord  (positive = stable)


def control_authority_sweep(
    polar: dict, plane: dict,
    motor: Motor, prop: Propeller, battery: Battery,
    V_array: np.ndarray,
    n_motors: int = 4,
    soc: float = 1.0,
) -> list[ControlAuthorityPoint | None]:
    """Estimate differential-thrust control authority at each cruise airspeed.

    Assumes an X-quad layout with motor arm length ``plane["arm"]`` and
    half-angle ``plane["dihedral"]`` (degrees) from the longitudinal axis.
    A symmetric X has dihedral=45; a stretched X (more pitch authority)
    uses a smaller angle.

    For each V the available ΔT per motor is limited by whichever is smaller:
    the headroom to full throttle or the room to reduce to zero thrust.

    Roll moment uses the lateral offset (arm·sin(dihedral)) and pitch moment
    uses the longitudinal offset (arm·cos(dihedral)).
    """
    mass = plane["mass"]
    area = plane["area"]
    chord = plane.get("chord", 1.0)
    rho = plane.get("rho", 1.225)
    g = plane.get("gravity", 9.81)
    arm = plane.get("arm")
    inertia = plane.get("inertia", {})
    Ixx = inertia.get("Ixx")
    Iyy = inertia.get("Iyy")
    dihedral_deg = plane.get("dihedral", 45.0)  # half-angle from longitudinal axis
    x_cg = plane.get("cg")           # CG position from LE [m]
    x_ac = plane.get("ac")           # AC position from LE [m]

    if arm is None or n_motors < 2:
        return [None] * len(V_array)

    # Static margin: (x_ac - x_cg) / chord.  Positive ⇒ CG ahead of AC ⇒ stable.
    if x_cg is not None and x_ac is not None:
        sm = (x_ac - x_cg) / chord
        dx_cg_ac = x_cg - x_ac          # moment arm for L about CG (positive = CG aft of AC)
    else:
        sm = float("nan")
        dx_cg_ac = 0.0                   # no offset info → ignore aero moment

    # Stretched X-quad: each motor arm makes angle `dihedral` with the
    # longitudinal (pitch) axis.
    #   lateral offset  (roll arm)  = arm · sin(dihedral)
    #   longitudinal offset (pitch arm) = arm · cos(dihedral)
    dihedral_rad = np.radians(dihedral_deg)
    l_lat = arm * np.sin(dihedral_rad)   # roll moment arm
    l_lon = arm * np.cos(dihedral_rad)   # pitch moment arm

    W = mass * g
    D_arr = airframe_drag(polar, mass, area, rho, g, V_array)

    out: list[ControlAuthorityPoint | None] = []
    for V, D in zip(V_array, D_arr):
        V = float(V)
        if V <= 0 or not np.isfinite(D):
            out.append(None)
            continue

        T_trim_per = float(D) / n_motors
        op_full = solve_operating_point(motor, prop, battery, 1.0, V, soc=soc)
        if op_full is None:
            out.append(None)
            continue
        T_max_per = op_full.thrust_N

        # Symmetric ΔT budget: can't exceed T_max or go below 0
        dT = min(T_max_per - T_trim_per, T_trim_per)

        # Roll: 2 motors on each side at lateral offset l_lat.
        # Increase right pair by ΔT, decrease left pair by ΔT.
        # M_roll = 4·ΔT·l_lat
        M_roll = 4.0 * dT * l_lat

        # Pitch: top/bottom pairs at longitudinal offset l_lon.
        # M_pitch = 4·ΔT·l_lon
        M_pitch = 4.0 * dT * l_lon

        # Aerodynamic pitching moment about CG due to CG-AC offset.
        # In level flight L = W, so: M_aero = L · (x_cg - x_ac) = W · dx_cg_ac
        # Positive M_aero ⇒ nose-up (CG aft of AC, destabilising).
        # The differential-thrust pitch moment must overcome this to trim and
        # still have authority left for manoeuvring.
        M_aero_pitch = W * dx_cg_ac

        # Net pitch authority: what remains after trimming the aero moment.
        # |M_pitch_available| - |M_aero_trim_needed|.  Sign preserved for insight.
        M_pitch_net = M_pitch - abs(M_aero_pitch)

        q = 0.5 * rho * V * V
        qSc = q * area * chord

        alpha_roll = M_roll / Ixx if Ixx else float("nan")
        alpha_pitch = M_pitch / Iyy if Iyy else float("nan")
        cp_roll = M_roll / qSc if qSc > 0 else float("nan")
        cp_pitch = M_pitch / qSc if qSc > 0 else float("nan")

        out.append(ControlAuthorityPoint(
            V=V,
            T_trim_per_motor_N=T_trim_per,
            T_max_per_motor_N=T_max_per,
            dT_max_N=dT,
            M_roll_Nm=M_roll,
            M_pitch_Nm=M_pitch,
            alpha_roll_rads2=alpha_roll,
            alpha_pitch_rads2=alpha_pitch,
            cp_roll=cp_roll,
            cp_pitch=cp_pitch,
            M_aero_pitch_Nm=M_aero_pitch,
            M_pitch_net_Nm=M_pitch_net,
            static_margin=sm,
        ))
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _first_valid(seq):
    return next((x for x in seq if x is not None), None)


def print_report(plane: dict, polar: dict, motor: Motor, prop: Propeller,
                 battery: Battery, n_motors: int,
                 hover: HoverPoint | None,
                 cruise: list[CruisePoint | None],
                 climb: list[ClimbPoint | None],
                 usable_fraction: float,
                 ctrl: list[ControlAuthorityPoint | None] | None = None) -> None:
    mass = plane["mass"]; area = plane["area"]
    rho = plane.get("rho", 1.225); g = plane.get("gravity", 9.81)
    W = mass * g
    E_Wh = battery.V_nominal * battery.capacity_Ah * usable_fraction

    print("=" * 70)
    print(f"Propeller : {prop.name}  ({prop.diameter_in}\" x {prop.pitch_in}\")")
    print(f"Motor     : {motor.name}  Kv={motor.Kv:.0f} rpm/V  R={motor.R*1000:.1f} mohm")
    print(f"Battery   : {battery.name}  {battery.series}S{battery.parallel}P  "
          f"{battery.capacity_Ah:.2f} Ah  V_nom={battery.V_nominal:.1f} V")
    x_cg = plane.get("cg")
    x_ac = plane.get("ac")
    chord = plane.get("chord", 1.0)
    print(f"Airframe  : m={mass:.2f} kg  W={W:.2f} N  S={area:.4f} m^2  "
          f"rho={rho:.3f} kg/m^3  n_motors={n_motors}")
    if x_cg is not None:
        print(f"CG pos    : {x_cg*1000:.1f} mm from LE  "
              f"({x_cg/chord*100:.1f}% chord)")
    if x_ac is not None:
        print(f"AC pos    : {x_ac*1000:.1f} mm from LE  "
              f"({x_ac/chord*100:.1f}% chord)")
    if x_cg is not None and x_ac is not None:
        sm = (x_ac - x_cg) / chord
        print(f"Static mrg: {sm*100:+.1f}% chord  "
              f"({'stable' if sm > 0 else 'unstable' if sm < 0 else 'neutral'}, "
              f"CG-AC = {(x_cg - x_ac)*1000:+.1f} mm)")
    print(f"Usable E  : {E_Wh:.1f} Wh  ({usable_fraction*100:.0f}% of "
          f"{battery.V_nominal*battery.capacity_Ah:.1f} Wh nominal)")

    try:
        vs = stall_speed(polar, mass, area, rho, g)
        print(f"V_stall   : {vs:.2f} m/s  (from polar CL_max)")
    except Exception:
        pass

    print("\n--- HOVER ---")
    if hover is None:
        op_full = solve_operating_point(motor, prop, battery, 1.0, 0.0)
        T_full = (op_full.thrust_N * n_motors) if op_full else 0.0
        print(f"  INFEASIBLE - full-throttle static thrust {T_full:.2f} N "
              f"< weight {W:.2f} N")
        print(f"  Thrust-to-weight at full throttle: {T_full/W:.3f}")
    else:
        h = hover
        print(f"  Thrust/Weight (full throttle) : {h.thrust_to_weight_max:.2f}")
        print(f"  Trim throttle                 : {h.throttle*100:.1f} %")
        print(f"  RPM                           : {h.rpm:.0f}")
        print(f"  Current per motor             : {h.current_per_motor_A:.2f} A")
        print(f"  Electrical power total        : {h.P_elec_total_W:.0f} W")
        print(f"  Hover endurance               : {h.endurance_min:.1f} min")
        if motor.Imax is not None and h.current_per_motor_A > motor.Imax:
            print(f"  WARNING: hover current {h.current_per_motor_A:.1f} A "
                  f"> motor Imax {motor.Imax:.1f} A")
        if battery.I_max is not None:
            I_pack = h.current_per_motor_A * n_motors
            if I_pack > battery.I_max:
                print(f"  WARNING: pack current {I_pack:.0f} A > "
                      f"battery C-limit {battery.I_max:.0f} A")

    print("\n--- CRUISE SWEEP ---")
    valid = [c for c in cruise if c is not None]
    if not valid:
        print("  No feasible cruise points in range.")
    else:
        hdr = ("   V    CL_req   Drag   thr%    RPM    I     P_elec   "
               "endur    range")
        units = ("  m/s           N             rpm    A       W       h"
                 "       km")
        print(hdr)
        print(units)
        print("  " + "-" * (len(hdr) - 2))
        for c in valid:
            print(f"  {c.V:4.1f}  {c.CL_req:6.3f}  {c.drag_N:5.2f}  "
                  f"{c.throttle*100:4.1f}  {c.rpm:6.0f}  "
                  f"{c.current_per_motor_A:4.1f}  {c.P_elec_total_W:6.0f}  "
                  f"{c.endurance_h:5.2f}  {c.range_km:6.1f}")

        # Key cruise figures of merit
        i_end = int(np.argmin([c.P_elec_total_W for c in valid]))
        i_rng = int(np.argmax([c.range_km for c in valid]))
        print()
        print(f"  Best endurance : {valid[i_end].endurance_h:.2f} h  @ "
              f"V = {valid[i_end].V:.1f} m/s  (P_elec = "
              f"{valid[i_end].P_elec_total_W:.0f} W)")
        print(f"  Best range     : {valid[i_rng].range_km:.1f} km  @ "
              f"V = {valid[i_rng].V:.1f} m/s  (throttle = "
              f"{valid[i_rng].throttle*100:.0f}%)")

        # V_max: largest V where cruise trim is feasible (throttle ≤ 1)
        V_max = max(c.V for c in valid)
        print(f"  V_max (cruise) : {V_max:.1f} m/s  (largest trim-feasible V "
              f"in sweep)")

    # V_max: powertrain-limited maximum speed
    vmax_pt = solve_vmax(polar, plane, motor, prop, battery,
                         n_motors=n_motors, soc=1.0)
    if vmax_pt is not None:
        print(f"\n  V_max (powertrain) : {vmax_pt.V:.2f} m/s  "
              f"(full-throttle thrust = drag = {vmax_pt.drag_N:.2f} N, "
              f"throttle 100%)")
    else:
        print("\n  V_max (powertrain) : could not determine "
              "(thrust never meets drag in the valid polar range)")

    print("\n--- CLIMB (full throttle) ---")
    valid_climb = [c for c in climb if c is not None]
    if valid_climb:
        i = int(np.argmax([c.roc_ms for c in valid_climb]))
        best = valid_climb[i]
        print(f"  Max RoC : {best.roc_ms:.2f} m/s  @ V = {best.V:.1f} m/s  "
              f"(excess thrust {best.excess_thrust_N:.2f} N)")
    else:
        print("  No feasible climb points.")

    print("\n--- THRUST / WEIGHT vs AIRSPEED ---")
    if hover is not None:
        print(f"  V = 0 (hover)  T/W_max = {hover.thrust_to_weight_max:.2f}  "
              f"T/W_trim = 1.00")
    if valid_climb:
        print("    V     T/W_max   T/W_trim (= D/W)   margin")
        print("   m/s                                (T_max - D)/W")
        print("  " + "-" * 52)
        for c in valid_climb:
            margin = c.TW_max - c.TW_cruise
            print(f"  {c.V:5.1f}   {c.TW_max:6.2f}     {c.TW_cruise:6.3f}"
                  f"           {margin:+.2f}")

    if ctrl is not None:
        valid_ctrl = [c for c in ctrl if c is not None]
        if valid_ctrl:
            arm = plane.get("arm")
            inertia = plane.get("inertia", {})
            print(f"\n--- CONTROL AUTHORITY (differential thrust, X-quad) ---")
            dihedral_deg = plane.get("dihedral", 45.0)
            dihedral_rad = np.radians(dihedral_deg)
            if arm is not None:
                print(f"  Motor arm       : {arm:.3f} m  "
                      f"(dihedral = {dihedral_deg:.0f}°, "
                      f"lateral = {arm*np.sin(dihedral_rad):.3f} m, "
                      f"longitudinal = {arm*np.cos(dihedral_rad):.3f} m)")
            if inertia:
                print(f"  Inertia         : Ixx={inertia.get('Ixx','?')}  "
                      f"Iyy={inertia.get('Iyy','?')}  "
                      f"Izz={inertia.get('Izz','?')} kg*m^2")
            x_cg_v = plane.get("cg")
            x_ac_v = plane.get("ac")
            if x_cg_v is not None and x_ac_v is not None:
                sm0 = valid_ctrl[0].static_margin
                print(f"  CG              : {x_cg_v*1000:.1f} mm from LE  "
                      f"({x_cg_v/chord*100:.1f}% chord)")
                print(f"  AC              : {x_ac_v*1000:.1f} mm from LE  "
                      f"({x_ac_v/chord*100:.1f}% chord)")
                print(f"  Static margin   : {sm0*100:+.1f}% chord  "
                      f"({'stable' if sm0 > 0 else 'unstable' if sm0 < 0 else 'neutral'})")
                print(f"  M_aero (pitch)  : {valid_ctrl[0].M_aero_pitch_Nm:+.3f} N·m  "
                      f"(L×(CG-AC), {'nose-up' if valid_ctrl[0].M_aero_pitch_Nm > 0 else 'nose-down'})")
            hdr = ("   V    T_trim  T_max   dT_max  M_roll  M_pitch  M_aero  M_net   "
                   "a_roll   a_pitch  CP_roll  CP_pitch")
            units = ("  m/s    N/mot  N/mot    N/mot   N*m     N*m     N*m     N*m   "
                     "rad/s^2  rad/s^2")
            print(hdr)
            print(units)
            print("  " + "-" * (len(hdr) - 2))
            for c in valid_ctrl:
                print(f"  {c.V:4.1f}  {c.T_trim_per_motor_N:5.2f}  "
                      f"{c.T_max_per_motor_N:5.1f}   {c.dT_max_N:5.2f}  "
                      f"{c.M_roll_Nm:6.2f}  {c.M_pitch_Nm:7.2f}  "
                      f"{c.M_aero_pitch_Nm:6.3f}  {c.M_pitch_net_Nm:6.2f}  "
                      f"{c.alpha_roll_rads2:7.1f}  {c.alpha_pitch_rads2:8.1f}  "
                      f"{c.cp_roll:7.1f}  {c.cp_pitch:8.1f}")
            # Summary: minimum control authority in sweep
            min_roll = min(c.alpha_roll_rads2 for c in valid_ctrl)
            min_pitch = min(c.alpha_pitch_rads2 for c in valid_ctrl)
            min_cp = min(c.cp_roll for c in valid_ctrl)
            v_min_cp = min(valid_ctrl, key=lambda c: c.cp_roll).V
            min_net = min(c.M_pitch_net_Nm for c in valid_ctrl)
            v_min_net = min(valid_ctrl, key=lambda c: c.M_pitch_net_Nm).V
            print()
            print(f"  Min roll  accel : {min_roll:.1f} rad/s^2  "
                  f"({np.degrees(min_roll):.0f} deg/s^2)")
            print(f"  Min pitch accel : {min_pitch:.1f} rad/s^2  "
                  f"({np.degrees(min_pitch):.0f} deg/s^2)")
            print(f"  Min CP ratio    : {min_cp:.1f}  @ V = {v_min_cp:.1f} m/s  "
                  f"(M_avail / q*S*c)")
            print(f"  Min net pitch   : {min_net:+.2f} N·m  @ V = {v_min_net:.1f} m/s  "
                  f"({'OK' if min_net > 0 else 'INSUFFICIENT — aero moment exceeds thrust authority'})")

    print("=" * 70)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_altitude_speeds(polar: dict, mass: float, area: float,
                         g: float = 9.81,
                         save_path: str | None = None,
                         show: bool = False) -> None:
    """Plot stall, best-range, and best-endurance speeds vs altitude (0–3000 m)."""
    import matplotlib.pyplot as plt

    pts = altitude_speed_sweep(polar, mass, area, g,
                               alt_min=0.0, alt_max=3000.0, alt_step=50.0)
    alt = np.array([p.altitude_m for p in pts])
    v_stall = np.array([p.V_stall for p in pts])
    v_range = np.array([p.V_best_range for p in pts])
    v_endur = np.array([p.V_best_endurance for p in pts])
    rho_arr = np.array([p.rho for p in pts])

    fig, ax1 = plt.subplots(figsize=(7, 5))

    ax1.plot(alt, v_stall, "o-", color="tab:red", markersize=3,
             label=f"$V_{{stall}}$  ({v_stall[0]:.1f} – {v_stall[-1]:.1f} m/s)")
    ax1.plot(alt, v_endur, "s-", color="tab:green", markersize=3,
             label=f"$V_{{best\\ endurance}}$  ({v_endur[0]:.1f} – {v_endur[-1]:.1f} m/s)")
    ax1.plot(alt, v_range, "^-", color="tab:blue", markersize=3,
             label=f"$V_{{best\\ range}}$  ({v_range[0]:.1f} – {v_range[-1]:.1f} m/s)")

    ax1.fill_between(alt, v_stall, v_endur, alpha=0.08, color="tab:orange")

    ax1.set_xlabel("Altitude [m]")
    ax1.set_ylabel("Airspeed [m/s]")
    ax1.set_title(f"Characteristic speeds vs altitude  (m={mass:.2f} kg, S={area:.4f} m²)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left", fontsize=9)

    # Secondary axis: air density
    ax2 = ax1.twinx()
    ax2.plot(alt, rho_arr, "--", color="grey", alpha=0.5, linewidth=1)
    ax2.set_ylabel("Air density [kg/m³]", color="grey")
    ax2.tick_params(axis="y", labelcolor="grey")

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved altitude-speed plot to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_motor_rpm_vs_airspeed(motor: Motor, battery: Battery,
                               hover: HoverPoint | None,
                               cruise: list[CruisePoint | None],
                               climb: list[ClimbPoint | None],
                               save_path: str | None = None,
                               show: bool = False,
                               ax=None) -> None:
    """Plot full-throttle (max) and cruise-trim motor RPM vs airspeed.

    Overlays:
      * full-throttle RPM curve (from climb sweep — the max at each V)
      * cruise-trim RPM curve (from cruise sweep — RPM needed to hold level flight)
      * hover RPM as a horizontal marker (V = 0)
      * theoretical no-load ceiling Kv · V_nominal as a dashed reference
    """
    import matplotlib.pyplot as plt

    valid_k = [c for c in climb if c is not None]
    valid_c = [c for c in cruise if c is not None]

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure

    if valid_k:
        V_k = np.array([c.V for c in valid_k])
        rpm_max = np.array([c.rpm_max for c in valid_k])
        ax.plot(V_k, rpm_max, "o-", color="tab:red",
                label="Max RPM (full throttle)")

    if valid_c:
        V_c = np.array([c.V for c in valid_c])
        rpm_trim = np.array([c.rpm for c in valid_c])
        ax.plot(V_c, rpm_trim, "s-", color="tab:blue",
                label="Trim RPM (cruise)")

    if hover is not None:
        ax.axhline(hover.rpm, color="tab:green", ls=":", lw=1.2,
                   label=f"Hover RPM = {hover.rpm:.0f}")

    rpm_noload = motor.Kv * battery.V_nominal
    ax.axhline(rpm_noload, color="k", ls="--", lw=1, alpha=0.6,
               label=f"Kv·V_nom = {rpm_noload:.0f} (no-load ceiling)")

    ax.set_xlabel("Airspeed [m/s]")
    ax.set_ylabel("Motor / prop RPM")
    ax.set_title("Motor RPM vs airspeed")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    if standalone:
        fig.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"Saved motor-RPM plot to {save_path}")
        if show:
            plt.show()
        else:
            plt.close(fig)


def plot_torque_thrust_vs_airspeed(n_motors: int,
                                   hover: HoverPoint | None,
                                   cruise: list[CruisePoint | None],
                                   climb: list[ClimbPoint | None],
                                   save_path: str | None = None,
                                   show: bool = False) -> None:
    """Plot per-motor thrust and torque vs airspeed — trim vs available.

    Two panels side-by-side:
      * Thrust per motor: trim (cruise, from cruise sweep) and available
        (full-throttle, from climb sweep).
      * Torque per motor: same two curves.

    Hover values (V = 0, trim = hover) are shown as markers where known.
    """
    import matplotlib.pyplot as plt

    valid_c = [c for c in cruise if c is not None]
    valid_k = [c for c in climb if c is not None]

    fig, (ax_t, ax_q) = plt.subplots(1, 2, figsize=(13, 5))

    # --- Thrust per motor ---
    if valid_k:
        V_k = np.array([c.V for c in valid_k])
        T_avail = np.array([c.T_max_total_N for c in valid_k]) / n_motors
        ax_t.plot(V_k, T_avail, "o-", color="tab:red",
                  label="Available (full throttle)")
    if valid_c:
        V_c = np.array([c.V for c in valid_c])
        T_trim = np.array([c.thrust_per_motor_N for c in valid_c])
        ax_t.plot(V_c, T_trim, "s-", color="tab:blue", label="Trim (cruise)")
    if hover is not None:
        ax_t.plot(0.0, hover.thrust_per_motor_N, "o", color="tab:green",
                  ms=8, label=f"Hover trim = {hover.thrust_per_motor_N:.2f} N")
    ax_t.axhline(0, color="k", lw=0.8)
    ax_t.set_xlabel("Airspeed [m/s]")
    ax_t.set_ylabel("Thrust per motor [N]")
    ax_t.set_title(f"Thrust per motor vs airspeed (N = {n_motors})")
    ax_t.grid(True, alpha=0.3)
    ax_t.legend(fontsize=9)

    # --- Torque per motor ---
    if valid_k:
        Q_avail = np.array([c.torque_per_motor_Nm for c in valid_k])
        ax_q.plot(V_k, Q_avail, "o-", color="tab:red",
                  label="Available (full throttle)")
    if valid_c:
        Q_trim = np.array([c.torque_per_motor_Nm for c in valid_c])
        ax_q.plot(V_c, Q_trim, "s-", color="tab:blue", label="Trim (cruise)")
    ax_q.axhline(0, color="k", lw=0.8)
    ax_q.set_xlabel("Airspeed [m/s]")
    ax_q.set_ylabel("Torque per motor [N·m]")
    ax_q.set_title(f"Torque per motor vs airspeed (N = {n_motors})")
    ax_q.grid(True, alpha=0.3)
    ax_q.legend(fontsize=9)

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved torque/thrust plot to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _save_individual(fig, path):
    """Save a single-plot figure and close it."""
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _individual_dir(save_path: str) -> str:
    """Return a sibling 'individual/' directory next to *save_path*."""
    base = os.path.splitext(save_path)[0]
    return base + "__individual"


def plot_performance(plane: dict, polar: dict, motor: Motor, prop: Propeller,
                     battery: Battery, n_motors: int,
                     hover: HoverPoint | None,
                     cruise: list[CruisePoint | None],
                     climb: list[ClimbPoint | None],
                     save_path: str,
                     show: bool = False,
                     ctrl: list[ControlAuthorityPoint | None] | None = None) -> None:
    import matplotlib.pyplot as plt

    mass = plane["mass"]; area = plane["area"]
    rho = plane.get("rho", 1.225); g = plane.get("gravity", 9.81)
    W = mass * g

    valid_c = [c for c in cruise if c is not None]
    valid_k = [c for c in climb if c is not None]

    if not valid_c:
        print("No cruise data to plot — skipping.")
        return

    V_c = np.array([c.V for c in valid_c])
    drag = np.array([c.drag_N for c in valid_c])
    thr = np.array([c.throttle for c in valid_c])
    P = np.array([c.P_elec_total_W for c in valid_c])
    endur = np.array([c.endurance_h for c in valid_c])
    rng = np.array([c.range_km for c in valid_c])
    I = np.array([c.current_per_motor_A for c in valid_c])

    V_k = np.array([c.V for c in valid_k]) if valid_k else np.array([])
    T_max = np.array([c.T_max_total_N for c in valid_k]) if valid_k else np.array([])
    roc = np.array([c.roc_ms for c in valid_k]) if valid_k else np.array([])
    TW_max = np.array([c.TW_max for c in valid_k]) if valid_k else np.array([])
    TW_trim = np.array([c.TW_cruise for c in valid_k]) if valid_k else np.array([])

    i_end = int(np.argmin(P))
    i_rng = int(np.argmax(rng))

    valid_ctrl = [c for c in ctrl if c is not None] if ctrl else []
    n_rows = 3 if valid_ctrl else 2

    # --- Individual plots directory ---
    ind = _individual_dir(save_path)
    os.makedirs(ind, exist_ok=True)
    n_ind = 0

    # ---- Combined figure ----
    fig, axes = plt.subplots(n_rows, 4, figsize=(18, 4.25 * n_rows))
    hover_str = ("hover: INFEASIBLE" if hover is None else
                 f"hover: {hover.throttle*100:.0f}% thr, "
                 f"{hover.P_elec_total_W:.0f} W, "
                 f"{hover.endurance_min:.1f} min  "
                 f"(T/W_max = {hover.thrust_to_weight_max:.2f})")
    fig.suptitle(
        f"{prop.name} + {motor.name} + {battery.series}S{battery.parallel}P  "
        f"|  m={mass:.2f} kg  S={area:.3f} m²  N={n_motors}\n{hover_str}",
        fontsize=11,
    )

    # 1) Thrust-to-weight vs airspeed
    ax = axes[0, 0]
    if V_k.size:
        # Prepend hover point (V=0) if available so curve starts at static T/W
        if hover is not None:
            V_tw = np.concatenate([[0.0], V_k])
            TWm = np.concatenate([[hover.thrust_to_weight_max], TW_max])
            TWt = np.concatenate([[1.0], TW_trim])
        else:
            V_tw, TWm, TWt = V_k, TW_max, TW_trim
        ax.plot(V_tw, TWm, "o-", color="tab:red",
                label=f"T/W max (full throttle x {n_motors})")
        ax.plot(V_tw, TWt, "s-", color="tab:blue",
                label="T/W trim (= D/W)")
    else:
        ax.plot(V_c, drag / W, "s-", color="tab:blue", label="T/W trim")
    ax.axhline(1.0, color="k", ls=":", lw=1, label="T/W = 1 (hover)")
    ax.set_xlabel("Airspeed [m/s]")
    ax.set_ylabel("Thrust / Weight")
    ax.set_title(f"Thrust-to-weight vs airspeed  (W = {W:.1f} N)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Individual: T/W
    fi, ai = plt.subplots(figsize=(7, 5))
    if V_k.size:
        if hover is not None:
            ai.plot(V_tw, TWm, "o-", color="tab:red",
                    label=f"T/W max (full throttle x {n_motors})")
            ai.plot(V_tw, TWt, "s-", color="tab:blue", label="T/W trim (= D/W)")
        else:
            ai.plot(V_k, TW_max, "o-", color="tab:red",
                    label=f"T/W max (full throttle x {n_motors})")
            ai.plot(V_k, TW_trim, "s-", color="tab:blue", label="T/W trim (= D/W)")
    else:
        ai.plot(V_c, drag / W, "s-", color="tab:blue", label="T/W trim")
    ai.axhline(1.0, color="k", ls=":", lw=1, label="T/W = 1 (hover)")
    ai.set_xlabel("Airspeed [m/s]"); ai.set_ylabel("Thrust / Weight")
    ai.set_title(f"Thrust-to-weight vs airspeed  (W = {W:.1f} N)")
    ai.grid(True, alpha=0.3); ai.legend(fontsize=9)
    _save_individual(fi, os.path.join(ind, "01_thrust_to_weight.png")); n_ind += 1

    # 2) Electrical power required for cruise trim
    ax = axes[0, 1]
    ax.plot(V_c, P, "o-", color="tab:purple")
    ax.plot(V_c[i_end], P[i_end], "*", ms=14, color="tab:orange",
            label=f"min P @ {V_c[i_end]:.1f} m/s = {P[i_end]:.0f} W")
    ax.set_xlabel("Airspeed [m/s]")
    ax.set_ylabel("P_elec total [W]")
    ax.set_title("Cruise trim power")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Individual: cruise power
    fi, ai = plt.subplots(figsize=(7, 5))
    ai.plot(V_c, P, "o-", color="tab:purple")
    ai.plot(V_c[i_end], P[i_end], "*", ms=14, color="tab:orange",
            label=f"min P @ {V_c[i_end]:.1f} m/s = {P[i_end]:.0f} W")
    ai.set_xlabel("Airspeed [m/s]"); ai.set_ylabel("P_elec total [W]")
    ai.set_title("Cruise trim power"); ai.grid(True, alpha=0.3); ai.legend(fontsize=9)
    _save_individual(fi, os.path.join(ind, "02_cruise_power.png")); n_ind += 1

    # 3) Endurance & range
    ax = axes[0, 2]
    ax.plot(V_c, endur * 60.0, "o-", color="tab:green", label="endurance [min]")
    ax.set_xlabel("Airspeed [m/s]")
    ax.set_ylabel("Endurance [min]", color="tab:green")
    ax.tick_params(axis="y", labelcolor="tab:green")
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(V_c, rng, "s-", color="tab:brown", label="range [km]")
    ax2.set_ylabel("Range [km]", color="tab:brown")
    ax2.tick_params(axis="y", labelcolor="tab:brown")
    ax.axvline(V_c[i_end], color="tab:green", ls="--", lw=1, alpha=0.6)
    ax.axvline(V_c[i_rng], color="tab:brown", ls="--", lw=1, alpha=0.6)
    ax.set_title(
        f"Endurance / Range  (best R = {rng[i_rng]:.1f} km @ "
        f"{V_c[i_rng]:.1f} m/s)"
    )

    # Individual: endurance & range
    fi, ai = plt.subplots(figsize=(7, 5))
    ai.plot(V_c, endur * 60.0, "o-", color="tab:green", label="endurance [min]")
    ai.set_xlabel("Airspeed [m/s]")
    ai.set_ylabel("Endurance [min]", color="tab:green")
    ai.tick_params(axis="y", labelcolor="tab:green")
    ai.grid(True, alpha=0.3)
    ai2 = ai.twinx()
    ai2.plot(V_c, rng, "s-", color="tab:brown", label="range [km]")
    ai2.set_ylabel("Range [km]", color="tab:brown")
    ai2.tick_params(axis="y", labelcolor="tab:brown")
    ai.axvline(V_c[i_end], color="tab:green", ls="--", lw=1, alpha=0.6)
    ai.axvline(V_c[i_rng], color="tab:brown", ls="--", lw=1, alpha=0.6)
    ai.set_title(f"Endurance / Range  (best R = {rng[i_rng]:.1f} km @ "
                 f"{V_c[i_rng]:.1f} m/s)")
    _save_individual(fi, os.path.join(ind, "03_endurance_range.png")); n_ind += 1

    # 4) Trim throttle
    ax = axes[1, 0]
    ax.plot(V_c, thr * 100.0, "o-", color="tab:orange")
    if hover is not None:
        ax.axhline(hover.throttle * 100.0, color="tab:red", ls="--", lw=1.2)
        ax.annotate(f"hover {hover.throttle*100:.0f}%",
                    xy=(V_c[0], hover.throttle * 100.0),
                    xytext=(V_c[0] + (V_c[-1] - V_c[0]) * 0.35,
                            hover.throttle * 100.0 + 3),
                    fontsize=8, color="tab:red",
                    arrowprops=dict(arrowstyle="->", color="tab:red", lw=0.8))
    ax.set_xlabel("Airspeed [m/s]")
    ax.set_ylabel("Trim throttle [%]")
    ax.set_title("Throttle needed to cruise")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    # Individual: trim throttle
    fi, ai = plt.subplots(figsize=(7, 5))
    ai.plot(V_c, thr * 100.0, "o-", color="tab:orange")
    if hover is not None:
        ai.axhline(hover.throttle * 100.0, color="tab:red", ls="--", lw=1.2)
        ai.annotate(f"hover {hover.throttle*100:.0f}%",
                    xy=(V_c[0], hover.throttle * 100.0),
                    xytext=(V_c[0] + (V_c[-1] - V_c[0]) * 0.35,
                            hover.throttle * 100.0 + 3),
                    fontsize=8, color="tab:red",
                    arrowprops=dict(arrowstyle="->", color="tab:red", lw=0.8))
    ai.set_xlabel("Airspeed [m/s]"); ai.set_ylabel("Trim throttle [%]")
    ai.set_title("Throttle needed to cruise"); ai.set_ylim(0, 105)
    ai.grid(True, alpha=0.3)
    _save_individual(fi, os.path.join(ind, "04_trim_throttle.png")); n_ind += 1

    # 5) Battery current per motor
    ax = axes[1, 1]
    ax.plot(V_c, I, "o-", color="tab:red")
    if motor.Imax is not None:
        ax.axhline(motor.Imax, color="k", ls="--", lw=1,
                   label=f"motor Imax = {motor.Imax:g} A")
    if hover is not None:
        ax.axhline(hover.current_per_motor_A, color="tab:blue", ls=":", lw=1,
                   label=f"hover = {hover.current_per_motor_A:.1f} A")
    ax.set_xlabel("Airspeed [m/s]")
    ax.set_ylabel("Current per motor [A]")
    ax.set_title("Cruise current draw")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Individual: current draw
    fi, ai = plt.subplots(figsize=(7, 5))
    ai.plot(V_c, I, "o-", color="tab:red")
    if motor.Imax is not None:
        ai.axhline(motor.Imax, color="k", ls="--", lw=1,
                   label=f"motor Imax = {motor.Imax:g} A")
    if hover is not None:
        ai.axhline(hover.current_per_motor_A, color="tab:blue", ls=":", lw=1,
                   label=f"hover = {hover.current_per_motor_A:.1f} A")
    ai.set_xlabel("Airspeed [m/s]"); ai.set_ylabel("Current per motor [A]")
    ai.set_title("Cruise current draw"); ai.grid(True, alpha=0.3); ai.legend(fontsize=9)
    _save_individual(fi, os.path.join(ind, "05_cruise_current.png")); n_ind += 1

    # 6) Angle of attack for level flight
    ax = axes[0, 3]
    a_lvl = alpha_level_flight(polar, mass, area, rho, g, V_c)
    ax.plot(V_c, a_lvl, "o-", color="tab:olive")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("Airspeed [m/s]")
    ax.set_ylabel("Alpha [deg]")
    ax.set_title("Angle of attack (L = W)")
    ax.grid(True, alpha=0.3)

    # Individual: angle of attack
    fi, ai = plt.subplots(figsize=(7, 5))
    ai.plot(V_c, a_lvl, "o-", color="tab:olive")
    ai.axhline(0, color="k", lw=0.8)
    ai.set_xlabel("Airspeed [m/s]"); ai.set_ylabel("Alpha [deg]")
    ai.set_title("Angle of attack (L = W)"); ai.grid(True, alpha=0.3)
    _save_individual(fi, os.path.join(ind, "06_angle_of_attack.png")); n_ind += 1

    # 7) Rate of climb
    ax = axes[1, 2]
    if V_k.size:
        ax.plot(V_k, roc, "o-", color="tab:cyan")
        i_roc = int(np.argmax(roc))
        ax.plot(V_k[i_roc], roc[i_roc], "*", ms=14, color="tab:orange",
                label=f"max RoC = {roc[i_roc]:.2f} m/s @ {V_k[i_roc]:.1f} m/s")
        ax.legend(fontsize=8)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("Airspeed [m/s]")
    ax.set_ylabel("Rate of climb [m/s]")
    ax.set_title("Climb (full throttle)")
    ax.grid(True, alpha=0.3)

    # Individual: rate of climb
    if V_k.size:
        fi, ai = plt.subplots(figsize=(7, 5))
        ai.plot(V_k, roc, "o-", color="tab:cyan")
        i_roc = int(np.argmax(roc))
        ai.plot(V_k[i_roc], roc[i_roc], "*", ms=14, color="tab:orange",
                label=f"max RoC = {roc[i_roc]:.2f} m/s @ {V_k[i_roc]:.1f} m/s")
        ai.axhline(0, color="k", lw=0.8)
        ai.set_xlabel("Airspeed [m/s]"); ai.set_ylabel("Rate of climb [m/s]")
        ai.set_title("Climb (full throttle)"); ai.grid(True, alpha=0.3)
        ai.legend(fontsize=9)
        _save_individual(fi, os.path.join(ind, "07_rate_of_climb.png")); n_ind += 1

    # 8) Thrust vs Drag — V_max (powertrain)
    ax = axes[1, 3]
    if V_k.size:
        ax.plot(V_k, T_max, "o-", color="tab:red",
                label="Full-throttle thrust")
        ax.plot(V_k, [c.drag_N for c in valid_k], "s-", color="tab:blue",
                label="Drag (L = W)")
        vmax_pt = solve_vmax(polar, plane, motor, prop, battery,
                             n_motors=n_motors, soc=1.0)
        if vmax_pt is not None:
            ax.axvline(vmax_pt.V, color="tab:green", ls="--", lw=1.5,
                       label=f"V_max = {vmax_pt.V:.1f} m/s")
            ax.plot(vmax_pt.V, vmax_pt.drag_N, "*", ms=14,
                    color="tab:green", zorder=5)
        ax.set_xlabel("Airspeed [m/s]")
        ax.set_ylabel("Force [N]")
        ax.set_title("Thrust vs Drag → V_max")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        # Individual: thrust vs drag
        fi, ai = plt.subplots(figsize=(7, 5))
        ai.plot(V_k, T_max, "o-", color="tab:red", label="Full-throttle thrust")
        ai.plot(V_k, [c.drag_N for c in valid_k], "s-", color="tab:blue",
                label="Drag (L = W)")
        if vmax_pt is not None:
            ai.axvline(vmax_pt.V, color="tab:green", ls="--", lw=1.5,
                       label=f"V_max = {vmax_pt.V:.1f} m/s")
            ai.plot(vmax_pt.V, vmax_pt.drag_N, "*", ms=14,
                    color="tab:green", zorder=5)
        ai.set_xlabel("Airspeed [m/s]"); ai.set_ylabel("Force [N]")
        ai.set_title("Thrust vs Drag → V_max"); ai.grid(True, alpha=0.3)
        ai.legend(fontsize=9)
        _save_individual(fi, os.path.join(ind, "08_thrust_vs_drag.png")); n_ind += 1
    else:
        ax.axis("off")

    # 9-12) Control authority panels (row 3)
    if n_rows == 3:
        V_ca = np.array([c.V for c in valid_ctrl])
        dT = np.array([c.dT_max_N for c in valid_ctrl])
        M_roll = np.array([c.M_roll_Nm for c in valid_ctrl])
        M_pitch = np.array([c.M_pitch_Nm for c in valid_ctrl])
        a_roll = np.array([c.alpha_roll_rads2 for c in valid_ctrl])
        a_pitch = np.array([c.alpha_pitch_rads2 for c in valid_ctrl])
        cp_roll = np.array([c.cp_roll for c in valid_ctrl])
        cp_pitch = np.array([c.cp_pitch for c in valid_ctrl])

        M_aero = np.array([c.M_aero_pitch_Nm for c in valid_ctrl])
        M_net = np.array([c.M_pitch_net_Nm for c in valid_ctrl])

        # 9) Available moment vs airspeed (with aero pitching moment)
        ax = axes[2, 0]
        ax.plot(V_ca, M_roll, "o-", color="tab:blue", label="Roll moment")
        ax.plot(V_ca, M_pitch, "s-", color="tab:red", label="Pitch moment (gross)")
        ax.plot(V_ca, M_net, "^-", color="tab:green", label="Pitch moment (net)")
        if np.any(np.isfinite(M_aero)) and np.any(M_aero != 0):
            ax.axhline(abs(M_aero[0]), color="tab:orange", ls="--", lw=1,
                       label=f"|M_aero| = {abs(M_aero[0]):.3f} N·m")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel("Airspeed [m/s]")
        ax.set_ylabel("Max moment [N·m]")
        ax.set_title("Control moment (diff. thrust + CG/AC)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        # Individual: control moment
        fi, ai = plt.subplots(figsize=(7, 5))
        ai.plot(V_ca, M_roll, "o-", color="tab:blue", label="Roll moment")
        ai.plot(V_ca, M_pitch, "s-", color="tab:red", label="Pitch moment (gross)")
        ai.plot(V_ca, M_net, "^-", color="tab:green", label="Pitch moment (net)")
        if np.any(np.isfinite(M_aero)) and np.any(M_aero != 0):
            ai.axhline(abs(M_aero[0]), color="tab:orange", ls="--", lw=1,
                       label=f"|M_aero| = {abs(M_aero[0]):.3f} N·m")
        ai.axhline(0, color="k", lw=0.8)
        ai.set_xlabel("Airspeed [m/s]"); ai.set_ylabel("Max moment [N·m]")
        ai.set_title("Control moment (diff. thrust + CG/AC)")
        ai.grid(True, alpha=0.3); ai.legend(fontsize=9)
        _save_individual(fi, os.path.join(ind, "09_control_moment.png")); n_ind += 1

        # 10) Angular acceleration authority
        ax = axes[2, 1]
        ax.plot(V_ca, np.degrees(a_roll), "o-", color="tab:blue",
                label=f"Roll (Ixx={plane.get('inertia',{}).get('Ixx','?')})")
        ax.plot(V_ca, np.degrees(a_pitch), "s-", color="tab:red",
                label=f"Pitch (Iyy={plane.get('inertia',{}).get('Iyy','?')})")
        ax.set_xlabel("Airspeed [m/s]")
        ax.set_ylabel("Max angular accel [deg/s²]")
        ax.set_title("Angular acceleration authority")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        # Individual: angular acceleration
        fi, ai = plt.subplots(figsize=(7, 5))
        ai.plot(V_ca, np.degrees(a_roll), "o-", color="tab:blue",
                label=f"Roll (Ixx={plane.get('inertia',{}).get('Ixx','?')})")
        ai.plot(V_ca, np.degrees(a_pitch), "s-", color="tab:red",
                label=f"Pitch (Iyy={plane.get('inertia',{}).get('Iyy','?')})")
        ai.set_xlabel("Airspeed [m/s]"); ai.set_ylabel("Max angular accel [deg/s²]")
        ai.set_title("Angular acceleration authority")
        ai.grid(True, alpha=0.3); ai.legend(fontsize=9)
        _save_individual(fi, os.path.join(ind, "10_angular_acceleration.png")); n_ind += 1

        # 11) Control power ratio
        ax = axes[2, 2]
        ax.plot(V_ca, cp_roll, "o-", color="tab:blue", label="Roll CP")
        ax.plot(V_ca, cp_pitch, "s-", color="tab:red", label="Pitch CP")
        ax.set_xlabel("Airspeed [m/s]")
        ax.set_ylabel("M / (q·S·c)")
        ax.set_title("Control power ratio")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        # Individual: control power ratio
        fi, ai = plt.subplots(figsize=(7, 5))
        ai.plot(V_ca, cp_roll, "o-", color="tab:blue", label="Roll CP")
        ai.plot(V_ca, cp_pitch, "s-", color="tab:red", label="Pitch CP")
        ai.set_xlabel("Airspeed [m/s]"); ai.set_ylabel("M / (q·S·c)")
        ai.set_title("Control power ratio"); ai.grid(True, alpha=0.3); ai.legend(fontsize=9)
        _save_individual(fi, os.path.join(ind, "11_control_power_ratio.png")); n_ind += 1

        # 12) ΔT budget per motor
        ax = axes[2, 3]
        T_trim = np.array([c.T_trim_per_motor_N for c in valid_ctrl])
        T_max_m = np.array([c.T_max_per_motor_N for c in valid_ctrl])
        ax.fill_between(V_ca, T_trim - dT, T_trim + dT,
                        alpha=0.25, color="tab:green", label="±ΔT envelope")
        ax.plot(V_ca, T_trim, "o-", color="tab:blue", label="T_trim/motor")
        ax.plot(V_ca, T_max_m, "s--", color="tab:red", label="T_max/motor")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel("Airspeed [m/s]")
        ax.set_ylabel("Thrust per motor [N]")
        ax.set_title("Thrust budget per motor")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        # Individual: thrust budget
        fi, ai = plt.subplots(figsize=(7, 5))
        ai.fill_between(V_ca, T_trim - dT, T_trim + dT,
                        alpha=0.25, color="tab:green", label="±ΔT envelope")
        ai.plot(V_ca, T_trim, "o-", color="tab:blue", label="T_trim/motor")
        ai.plot(V_ca, T_max_m, "s--", color="tab:red", label="T_max/motor")
        ai.axhline(0, color="k", lw=0.8)
        ai.set_xlabel("Airspeed [m/s]"); ai.set_ylabel("Thrust per motor [N]")
        ai.set_title("Thrust budget per motor"); ai.grid(True, alpha=0.3)
        ai.legend(fontsize=9)
        _save_individual(fi, os.path.join(ind, "12_thrust_budget.png")); n_ind += 1

    # Individual: characteristic speeds vs altitude
    plot_altitude_speeds(polar, mass, area, g,
                         save_path=os.path.join(ind, "13_altitude_speeds.png"))
    n_ind += 1

    # Individual: motor RPM vs airspeed
    plot_motor_rpm_vs_airspeed(
        motor, battery, hover, cruise, climb,
        save_path=os.path.join(ind, "14_motor_rpm_vs_airspeed.png"),
    )
    n_ind += 1

    # Individual: torque & thrust (trim vs available) vs airspeed
    plot_torque_thrust_vs_airspeed(
        n_motors, hover, cruise, climb,
        save_path=os.path.join(ind, "15_torque_thrust_vs_airspeed.png"),
    )
    n_ind += 1

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150)
    print(f"Saved performance plot to {save_path}")
    print(f"Saved {n_ind} individual plots to {ind}/")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _auto_plot_path(prop_file: str, motor_xml: str, battery_xml: str,
                    polar_file: str, out_dir: str) -> str:
    def stem(p: str) -> str:
        return Path(p).stem
    name = (f"perf__{stem(prop_file)}__{stem(motor_xml)}__"
            f"{stem(battery_xml)}__{stem(polar_file)}.png")
    # Filesystem-friendly
    name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)
    return os.path.join(out_dir, name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("prop_file", help="APC-format propeller data file")
    ap.add_argument("motor_xml", help="Motor XML")
    ap.add_argument("battery_xml", help="Battery XML")
    ap.add_argument("polar_file", help="Flow5 polar .txt (alpha sweep)")
    ap.add_argument("--plane", default="plane.xml",
                    help="Plane parameter XML (default: plane.xml)")
    ap.add_argument("--n-motors", type=int, default=None,
                    help="Override number of motors/props (default: <count> "
                         "from motor XML)")
    ap.add_argument("--soc", type=float, default=1.0,
                    help="Battery state of charge 0..1 (default 1.0)")
    ap.add_argument("--usable", type=float, default=0.8,
                    help="Usable battery fraction for endurance (default 0.8)")
    ap.add_argument("--vmin", type=float, default=5.0, help="Min cruise V [m/s]")
    ap.add_argument("--vmax", type=float, default=40.0, help="Max cruise V [m/s]")
    ap.add_argument("--vstep", type=float, default=1.0, help="V step [m/s]")
    ap.add_argument("--out", default="plots",
                    help="Output directory for the performance plot")
    ap.add_argument("--no-plot", action="store_true", help="Skip plot")
    ap.add_argument("--show", action="store_true",
                    help="Show plot interactively")
    args = ap.parse_args()

    prop = load_propeller(args.prop_file)
    motor = load_motor(args.motor_xml)
    battery = load_battery(args.battery_xml)
    polar = load_polar(Path(args.polar_file))
    plane = {"rho": 1.225, "gravity": 9.81}
    plane.update(load_plane_xml(Path(args.plane)))
    if "mass" not in plane or "area" not in plane:
        raise SystemExit(
            f"{args.plane} must define <mass> and <area> for coupled analysis."
        )

    n_motors = args.n_motors if args.n_motors is not None else motor.count
    if n_motors < 1:
        raise SystemExit(f"n_motors must be >= 1 (got {n_motors})")

    V_array = np.arange(args.vmin, args.vmax + 1e-9, args.vstep)

    hover = solve_hover(
        motor, prop, battery,
        mass=plane["mass"], g=plane.get("gravity", 9.81),
        n_motors=n_motors, soc=args.soc,
        usable_fraction=args.usable,
    )
    cruise = cruise_sweep(
        polar, plane, motor, prop, battery, V_array,
        n_motors=n_motors, soc=args.soc,
        usable_fraction=args.usable,
    )
    climb = climb_sweep(
        polar, plane, motor, prop, battery, V_array,
        n_motors=n_motors, soc=args.soc,
    )
    ctrl = control_authority_sweep(
        polar, plane, motor, prop, battery, V_array,
        n_motors=n_motors, soc=args.soc,
    )

    print_report(plane, polar, motor, prop, battery, n_motors,
                 hover, cruise, climb, usable_fraction=args.usable,
                 ctrl=ctrl)

    # Altitude speed analysis
    alt_pts = altitude_speed_sweep(
        polar, plane["mass"], plane["area"],
        g=plane.get("gravity", 9.81),
    )
    print("\n--- CHARACTERISTIC SPEEDS vs ALTITUDE ---")
    print("   Alt [m]   ρ [kg/m³]  V_stall  V_endurance  V_best_range")
    print("  " + "-" * 58)
    for p in alt_pts[::5]:  # every 500 m
        print(f"  {p.altitude_m:6.0f}    {p.rho:7.4f}    "
              f"{p.V_stall:6.2f}     {p.V_best_endurance:6.2f}       "
              f"{p.V_best_range:6.2f}")
    print("=" * 70)

    if not args.no_plot:
        save_path = _auto_plot_path(
            args.prop_file, args.motor_xml, args.battery_xml,
            args.polar_file, args.out,
        )
        plot_performance(plane, polar, motor, prop, battery, n_motors,
                         hover, cruise, climb,
                         save_path=save_path, show=args.show,
                         ctrl=ctrl)
        # Standalone altitude-speed plot
        alt_plot = os.path.splitext(save_path)[0] + "__altitude_speeds.png"
        plot_altitude_speeds(
            polar, plane["mass"], plane["area"],
            g=plane.get("gravity", 9.81),
            save_path=alt_plot, show=args.show,
        )
        # Standalone motor-RPM plot
        rpm_plot = os.path.splitext(save_path)[0] + "__motor_rpm.png"
        plot_motor_rpm_vs_airspeed(
            motor, battery, hover, cruise, climb,
            save_path=rpm_plot, show=args.show,
        )
        # Standalone torque & thrust plot
        tq_plot = os.path.splitext(save_path)[0] + "__torque_thrust.png"
        plot_torque_thrust_vs_airspeed(
            n_motors, hover, cruise, climb,
            save_path=tq_plot, show=args.show,
        )


if __name__ == "__main__":
    main()
