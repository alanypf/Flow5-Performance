"""Motor + propeller combined performance calculator.

Propeller data uses APC-style PERFILES format (e.g. PER3_7x15E.txt), which
contains repeating "PROP RPM = N" blocks followed by a table of V, J, Ct, Cp,
thrust, torque, power, ... for that RPM.

Battery spec is an XML file with the following schema:

    <battery>
        <name>Tattu 4S 5000mAh 25C</name>
        <chemistry>LiPo</chemistry>
        <configuration>
            <series>4</series>       <!-- cells in series -->
            <parallel>1</parallel>   <!-- parallel strings -->
        </configuration>
        <cell>
            <V_nominal unit="V">3.7</V_nominal>
            <V_full    unit="V">4.2</V_full>
            <V_empty   unit="V">3.3</V_empty>
            <capacity  unit="Ah">5.0</capacity>
            <R_internal unit="ohm">0.005</R_internal>
        </cell>
        <C_rating unit="1/h">25</C_rating>
    </battery>

Motor spec is an XML file with the following schema:

    <motor>
        <name>SunnySky X2216</name>
        <Kv unit="rpm/V">880</Kv>          <!-- velocity constant -->
        <R  unit="ohm">0.09</R>             <!-- winding resistance -->
        <I0 unit="A">0.7</I0>               <!-- no-load current at I0_V -->
        <I0_V unit="V">10</I0_V>            <!-- voltage at which I0 measured (optional) -->
        <Imax unit="A">30</Imax>            <!-- max continuous current (optional) -->
        <Pmax unit="W">450</Pmax>           <!-- max continuous power (optional) -->
    </motor>

Usage (as a script):

    python motor_prop_performance.py PER3_7x15E.txt motor.xml battery.xml \
        --throttle 1.0 --vmin 0 --vmax 30 --vstep 2

Outputs a table of airspeed vs operating RPM, thrust, torque, current,
electrical power, mechanical power, motor/prop/overall efficiency.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np


MPH_TO_MS = 0.44704


# ---------------------------------------------------------------------------
# Propeller
# ---------------------------------------------------------------------------


@dataclass
class PropPoint:
    rpm: float
    v_ms: float
    J: float
    Ct: float
    Cp: float
    thrust_N: float
    torque_Nm: float
    power_W: float


class Propeller:
    def __init__(self, name: str, diameter_in: float, pitch_in: float,
                 points: list[PropPoint]):
        self.name = name
        self.diameter_in = diameter_in
        self.pitch_in = pitch_in
        self.diameter_m = diameter_in * 0.0254
        self.points = points
        # Unique sorted RPM list for interpolation
        self.rpms = sorted({p.rpm for p in points})

    def _points_at_rpm(self, rpm: float) -> list[PropPoint]:
        return [p for p in self.points if p.rpm == rpm]

    def _interp_at_rpm(self, rpm: float, v_ms: float) -> PropPoint:
        """Interpolate thrust/torque/power at a given airspeed within one RPM block."""
        pts = self._points_at_rpm(rpm)
        vs = np.array([p.v_ms for p in pts])
        order = np.argsort(vs)
        vs = vs[order]
        pts = [pts[i] for i in order]

        v_ms = max(vs[0], min(vs[-1], v_ms))
        thrust = np.interp(v_ms, vs, [p.thrust_N for p in pts])
        torque = np.interp(v_ms, vs, [p.torque_Nm for p in pts])
        power = np.interp(v_ms, vs, [p.power_W for p in pts])
        Ct = np.interp(v_ms, vs, [p.Ct for p in pts])
        Cp = np.interp(v_ms, vs, [p.Cp for p in pts])
        J = np.interp(v_ms, vs, [p.J for p in pts])
        return PropPoint(rpm=rpm, v_ms=float(v_ms), J=float(J),
                         Ct=float(Ct), Cp=float(Cp),
                         thrust_N=float(thrust), torque_Nm=float(torque),
                         power_W=float(power))

    def at(self, rpm: float, v_ms: float) -> PropPoint:
        """Bilinear-style interpolation in RPM and airspeed."""
        rpms = self.rpms
        if rpm <= rpms[0]:
            return self._interp_at_rpm(rpms[0], v_ms)
        if rpm >= rpms[-1]:
            return self._interp_at_rpm(rpms[-1], v_ms)

        # Find bracketing RPMs
        for i in range(len(rpms) - 1):
            if rpms[i] <= rpm <= rpms[i + 1]:
                r_lo, r_hi = rpms[i], rpms[i + 1]
                break
        p_lo = self._interp_at_rpm(r_lo, v_ms)
        p_hi = self._interp_at_rpm(r_hi, v_ms)
        t = (rpm - r_lo) / (r_hi - r_lo)
        mix = lambda a, b: a + (b - a) * t
        return PropPoint(
            rpm=rpm, v_ms=v_ms,
            J=mix(p_lo.J, p_hi.J),
            Ct=mix(p_lo.Ct, p_hi.Ct),
            Cp=mix(p_lo.Cp, p_hi.Cp),
            thrust_N=mix(p_lo.thrust_N, p_hi.thrust_N),
            torque_Nm=mix(p_lo.torque_Nm, p_hi.torque_Nm),
            power_W=mix(p_lo.power_W, p_hi.power_W),
        )


def load_propeller(path: str) -> Propeller:
    with open(path, "r", encoding="latin-1") as f:
        lines = f.readlines()

    # Name / dimensions from first non-empty line, e.g. "7x15E  (7x15E.dat)"
    name = ""
    for ln in lines:
        if ln.strip():
            name = ln.strip().split()[0]
            break
    m = re.match(r"(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)", name)
    diameter_in = float(m.group(1)) if m else 0.0
    pitch_in = float(m.group(2)) if m else 0.0

    points: list[PropPoint] = []
    current_rpm: float | None = None
    for ln in lines:
        m_rpm = re.search(r"PROP RPM\s*=\s*(\d+)", ln)
        if m_rpm:
            current_rpm = float(m_rpm.group(1))
            continue
        if current_rpm is None:
            continue
        parts = ln.split()
        # A data row starts with a float (V in mph) and has at least 15 columns.
        if len(parts) < 15:
            continue
        try:
            nums = [float(x) for x in parts[:15]]
        except ValueError:
            continue
        v_mph, J, Pe, Ct, Cp = nums[0:5]
        pwr_W, torque_Nm, thrust_N = nums[8], nums[9], nums[10]
        points.append(PropPoint(
            rpm=current_rpm,
            v_ms=v_mph * MPH_TO_MS,
            J=J, Ct=Ct, Cp=Cp,
            thrust_N=thrust_N,
            torque_Nm=torque_Nm,
            power_W=pwr_W,
        ))

    if not points:
        raise ValueError(f"No propeller data parsed from {path}")
    return Propeller(name=name, diameter_in=diameter_in,
                     pitch_in=pitch_in, points=points)


# ---------------------------------------------------------------------------
# Motor
# ---------------------------------------------------------------------------


@dataclass
class Motor:
    name: str
    Kv: float         # rpm / V
    R: float          # ohm
    I0: float         # A at I0_V
    I0_V: float       # V (reference voltage for I0)
    Imax: float | None = None
    Pmax: float | None = None
    count: int = 1    # number of identical motors on the airframe

    @property
    def Kt(self) -> float:
        """Torque constant [N·m / A]."""
        return 60.0 / (2.0 * math.pi * self.Kv)

    def current(self, V_eff: float, rpm: float) -> float:
        back_emf = rpm / self.Kv
        return (V_eff - back_emf) / self.R

    def torque(self, V_eff: float, rpm: float) -> float:
        """Shaft torque [N·m] at effective terminal voltage V_eff and rpm."""
        I = self.current(V_eff, rpm)
        # Scale no-load current with speed (iron + bearing losses grow ~ rpm)
        no_load_rpm = self.Kv * self.I0_V
        I0_eff = self.I0 * (rpm / no_load_rpm) if no_load_rpm > 0 else self.I0
        return (I - I0_eff) * self.Kt


def load_motor(path: str) -> Motor:
    tree = ET.parse(path)
    root = tree.getroot()

    def get(tag, default=None, required=False, cast=float):
        el = root.find(tag)
        if el is None or el.text is None or not el.text.strip():
            if required:
                raise ValueError(f"Motor XML missing required <{tag}>")
            return default
        return cast(el.text.strip())

    name = get("name", default="motor", cast=str) or "motor"
    Kv = get("Kv", required=True)
    R = get("R", required=True)
    I0 = get("I0", required=True)
    I0_V = get("I0_V", default=10.0)
    Imax = get("Imax", default=None)
    Pmax = get("Pmax", default=None)
    count = int(get("count", default=1.0) or 1)
    return Motor(name=name, Kv=Kv, R=R, I0=I0, I0_V=I0_V,
                 Imax=Imax, Pmax=Pmax, count=count)


# ---------------------------------------------------------------------------
# Battery
# ---------------------------------------------------------------------------


@dataclass
class Battery:
    name: str
    chemistry: str
    series: int
    parallel: int
    V_nominal_cell: float
    V_full_cell: float
    V_empty_cell: float
    capacity_cell_Ah: float
    R_cell: float
    C_rating: float | None = None

    @property
    def V_nominal(self) -> float:
        return self.series * self.V_nominal_cell

    @property
    def V_full(self) -> float:
        return self.series * self.V_full_cell

    @property
    def V_empty(self) -> float:
        return self.series * self.V_empty_cell

    @property
    def capacity_Ah(self) -> float:
        return self.parallel * self.capacity_cell_Ah

    @property
    def R_internal(self) -> float:
        return self.series * self.R_cell / self.parallel

    @property
    def I_max(self) -> float | None:
        if self.C_rating is None:
            return None
        return self.C_rating * self.capacity_Ah

    def terminal_voltage(self, V_oc: float, current_A: float) -> float:
        """Loaded terminal voltage given open-circuit voltage and draw."""
        return V_oc - current_A * self.R_internal


def load_battery(path: str) -> Battery:
    tree = ET.parse(path)
    root = tree.getroot()

    def req(tag_path: str, cast=float):
        el = root.find(tag_path)
        if el is None or el.text is None or not el.text.strip():
            raise ValueError(f"Battery XML missing required <{tag_path}>")
        return cast(el.text.strip())

    def opt(tag_path: str, default=None, cast=float):
        el = root.find(tag_path)
        if el is None or el.text is None or not el.text.strip():
            return default
        return cast(el.text.strip())

    name = opt("name", default="battery", cast=str) or "battery"
    chemistry = opt("chemistry", default="LiPo", cast=str) or "LiPo"
    return Battery(
        name=name,
        chemistry=chemistry,
        series=int(req("configuration/series", cast=float)),
        parallel=int(req("configuration/parallel", cast=float)),
        V_nominal_cell=req("cell/V_nominal"),
        V_full_cell=opt("cell/V_full", default=req("cell/V_nominal")),
        V_empty_cell=opt("cell/V_empty", default=req("cell/V_nominal")),
        capacity_cell_Ah=req("cell/capacity"),
        R_cell=req("cell/R_internal"),
        C_rating=opt("C_rating", default=None),
    )


# ---------------------------------------------------------------------------
# Combined operating point
# ---------------------------------------------------------------------------


@dataclass
class OperatingPoint:
    v_ms: float
    throttle: float
    rpm: float
    V_oc: float
    V_terminal: float
    V_eff: float
    back_emf: float
    current_A: float
    thrust_N: float
    torque_Nm: float
    P_shaft_W: float
    P_elec_W: float
    eta_motor: float
    eta_prop: float
    eta_total: float


def solve_operating_point(motor: Motor, prop: Propeller,
                          battery: Battery, throttle: float,
                          v_ms: float, soc: float = 1.0) -> OperatingPoint | None:
    """Find rpm where motor torque = prop torque at given airspeed.

    Battery voltage follows a linear SoC model between V_empty and V_full; the
    loaded terminal voltage is V_oc - I * R_internal. Throttle is modeled as
    PWM duty-cycle so V_eff = throttle * V_terminal.
    """
    soc = max(0.0, min(1.0, soc))
    V_oc = battery.V_empty + soc * (battery.V_full - battery.V_empty)

    def V_eff_for(rpm: float) -> float:
        # Self-consistent: current depends on V_eff, V_eff depends on current.
        V_eff = throttle * V_oc
        for _ in range(8):
            I = max(0.0, motor.current(V_eff, rpm))
            V_term = V_oc - I * battery.R_internal
            V_eff = throttle * V_term
        return V_eff

    def residual(rpm: float) -> float:
        V_eff = V_eff_for(rpm)
        return motor.torque(V_eff, rpm) - prop.at(rpm, v_ms).torque_Nm

    V_eff_noload = throttle * V_oc
    lo, hi = prop.rpms[0], min(prop.rpms[-1], motor.Kv * V_eff_noload)
    if hi <= lo:
        return None
    f_lo, f_hi = residual(lo), residual(hi)
    if f_lo * f_hi > 0:
        # No sign change — motor can't reach prop-drag equilibrium in range.
        return None

    for _ in range(60):
        mid = 0.5 * (lo + hi)
        f_mid = residual(mid)
        if abs(f_mid) < 1e-6 or (hi - lo) < 1e-3:
            break
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid

    rpm = 0.5 * (lo + hi)
    pp = prop.at(rpm, v_ms)
    omega = rpm * 2 * math.pi / 60.0
    P_shaft = pp.torque_Nm * omega
    V_eff = V_eff_for(rpm)
    I = motor.current(V_eff, rpm)
    V_term = V_oc - I * battery.R_internal
    back_emf = rpm / motor.Kv
    P_elec = V_term * I  # battery-side power (throttle assumed lossless PWM)
    eta_motor = P_shaft / P_elec if P_elec > 1e-9 else 0.0
    P_useful = pp.thrust_N * v_ms
    eta_prop = P_useful / P_shaft if P_shaft > 1e-9 else 0.0
    eta_total = P_useful / P_elec if P_elec > 1e-9 else 0.0

    return OperatingPoint(
        v_ms=v_ms, throttle=throttle, rpm=rpm,
        V_oc=V_oc, V_terminal=V_term, V_eff=V_eff, back_emf=back_emf,
        current_A=I,
        thrust_N=pp.thrust_N, torque_Nm=pp.torque_Nm,
        P_shaft_W=P_shaft, P_elec_W=P_elec,
        eta_motor=eta_motor, eta_prop=eta_prop, eta_total=eta_total,
    )


def sweep(motor: Motor, prop: Propeller, battery: Battery, throttle: float,
          v_min: float, v_max: float, v_step: float,
          soc: float = 1.0) -> list[OperatingPoint]:
    out = []
    v = v_min
    while v <= v_max + 1e-9:
        op = solve_operating_point(motor, prop, battery, throttle, v, soc=soc)
        if op is not None:
            out.append(op)
        v += v_step
    return out


def sweep_throttle(motor: Motor, prop: Propeller, battery: Battery,
                   v_ms: float, n_steps: int = 21,
                   soc: float = 1.0) -> tuple[np.ndarray, list[OperatingPoint]]:
    """Sweep throttle 0..1 at a fixed airspeed. Returns (throttles, ops)."""
    throttles = np.linspace(0.0, 1.0, n_steps)
    ts: list[float] = []
    ops: list[OperatingPoint] = []
    for t in throttles:
        op = solve_operating_point(motor, prop, battery, float(t), v_ms, soc=soc)
        if op is not None:
            ts.append(float(t))
            ops.append(op)
    return np.array(ts), ops


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _save_individual_mp(fig, path):
    """Save a single-plot figure and close it."""
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _individual_dir_mp(save_path: str) -> str:
    """Return a sibling 'individual/' directory next to *save_path*."""
    base = os.path.splitext(save_path)[0]
    return base + "__individual"


def plot_results(ops: list[OperatingPoint], prop: Propeller, motor: Motor,
                 battery: Battery, throttle: float, soc: float,
                 save_path: str | None = None, show: bool = True) -> None:
    """Plot the important performance curves vs airspeed."""
    import matplotlib.pyplot as plt

    v = np.array([o.v_ms for o in ops])
    rpm = np.array([o.rpm for o in ops])
    thrust = np.array([o.thrust_N for o in ops])
    torque = np.array([o.torque_Nm for o in ops])
    P_shaft = np.array([o.P_shaft_W for o in ops])
    P_elec = np.array([o.P_elec_W for o in ops])
    I = np.array([o.current_A for o in ops])
    eta_m = np.array([o.eta_motor for o in ops])
    eta_p = np.array([o.eta_prop for o in ops])
    eta_t = np.array([o.eta_total for o in ops])

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True)
    n = motor.count
    title = (f"{prop.name}  +  {motor.name} x{n}  +  "
             f"{battery.series}S{battery.parallel}P {battery.chemistry}  "
             f"(throttle={throttle:.2f}, SoC={soc:.2f})")
    fig.suptitle(title, fontsize=12)

    # Per-motor arrays (unscaled originals)
    thrust_1 = thrust
    P_shaft_1 = P_shaft
    P_elec_1 = P_elec
    I_1 = I

    # Total arrays (all motors combined)
    thrust_tot = thrust_1 * n
    P_shaft_tot = P_shaft_1 * n
    P_elec_tot = P_elec_1 * n
    I_tot = I_1 * n

    lbl_1 = " (per motor)" if n > 1 else ""
    lbl_t = " (total)" if n > 1 else ""

    # --- Individual plots directory ---
    ind = _individual_dir_mp(save_path) if save_path else None
    if ind:
        os.makedirs(ind, exist_ok=True)
    n_ind = 0

    ax = axes[0, 0]
    if n > 1:
        ax.plot(v, thrust_1, "x--", color="tab:cyan", alpha=0.6,
                label=f"per motor")
    ax.plot(v, thrust_tot, "o-", color="tab:blue",
            label=f"total ({n} motors)" if n > 1 else None)
    ax.set_ylabel("Thrust [N]")
    ax.grid(True, alpha=0.3)
    ax.set_title("Thrust vs airspeed")
    if n > 1:
        ax.legend(fontsize=8)

    # Individual: thrust
    if ind:
        fi, ai = plt.subplots(figsize=(7, 5))
        if n > 1:
            ai.plot(v, thrust_1, "x--", color="tab:cyan", alpha=0.6, label="per motor")
        ai.plot(v, thrust_tot, "o-", color="tab:blue",
                label=f"total ({n} motors)" if n > 1 else None)
        ai.set_xlabel("Airspeed [m/s]"); ai.set_ylabel("Thrust [N]")
        ai.set_title("Thrust vs airspeed"); ai.grid(True, alpha=0.3)
        if n > 1: ai.legend(fontsize=9)
        _save_individual_mp(fi, os.path.join(ind, "01_thrust.png")); n_ind += 1

    ax = axes[0, 1]
    ax.plot(v, P_elec_tot, "o-", color="tab:red",
            label=f"P_elec{lbl_t}")
    ax.plot(v, P_shaft_tot, "s-", color="tab:orange",
            label=f"P_shaft{lbl_t}")
    ax.plot(v, thrust_tot * v, "^-", color="tab:green",
            label="P_useful = T·V")
    if motor.Pmax is not None:
        ax.axhline(motor.Pmax, color="grey", ls=":", lw=1,
                   label=f"motor Pmax={motor.Pmax:g} W")
        ax.axhline(motor.Pmax * n, color="k", ls="--", lw=1,
                   label=f"total Pmax={motor.Pmax * n:g} W")
    ax.set_ylabel("Power [W]")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title("Power flow")

    # Individual: power flow
    if ind:
        fi, ai = plt.subplots(figsize=(7, 5))
        ai.plot(v, P_elec_tot, "o-", color="tab:red", label=f"P_elec{lbl_t}")
        ai.plot(v, P_shaft_tot, "s-", color="tab:orange", label=f"P_shaft{lbl_t}")
        ai.plot(v, thrust_tot * v, "^-", color="tab:green", label="P_useful = T·V")
        if motor.Pmax is not None:
            ai.axhline(motor.Pmax, color="grey", ls=":", lw=1,
                       label=f"motor Pmax={motor.Pmax:g} W")
            ai.axhline(motor.Pmax * n, color="k", ls="--", lw=1,
                       label=f"total Pmax={motor.Pmax * n:g} W")
        ai.set_xlabel("Airspeed [m/s]"); ai.set_ylabel("Power [W]")
        ai.set_title("Power flow"); ai.grid(True, alpha=0.3); ai.legend(fontsize=9)
        _save_individual_mp(fi, os.path.join(ind, "02_power_flow.png")); n_ind += 1

    ax = axes[0, 2]
    ax.plot(v, eta_m, "o-", label="η motor")
    ax.plot(v, eta_p, "s-", label="η prop")
    ax.plot(v, eta_t, "^-", label="η total")
    ax.set_ylabel("Efficiency")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title("Efficiencies")

    # Individual: efficiencies
    if ind:
        fi, ai = plt.subplots(figsize=(7, 5))
        ai.plot(v, eta_m, "o-", label="η motor")
        ai.plot(v, eta_p, "s-", label="η prop")
        ai.plot(v, eta_t, "^-", label="η total")
        ai.set_xlabel("Airspeed [m/s]"); ai.set_ylabel("Efficiency")
        ai.set_ylim(0, 1); ai.set_title("Efficiencies")
        ai.grid(True, alpha=0.3); ai.legend(fontsize=9)
        _save_individual_mp(fi, os.path.join(ind, "03_efficiencies.png")); n_ind += 1

    ax = axes[1, 0]
    ax.plot(v, rpm, "o-", color="tab:purple")
    ax.set_ylabel("RPM")
    ax.set_xlabel("Airspeed [m/s]")
    ax.grid(True, alpha=0.3)
    ax.set_title("Operating RPM")

    # Individual: RPM
    if ind:
        fi, ai = plt.subplots(figsize=(7, 5))
        ai.plot(v, rpm, "o-", color="tab:purple")
        ai.set_xlabel("Airspeed [m/s]"); ai.set_ylabel("RPM")
        ai.set_title("Operating RPM"); ai.grid(True, alpha=0.3)
        _save_individual_mp(fi, os.path.join(ind, "04_operating_rpm.png")); n_ind += 1

    ax = axes[1, 1]
    if n > 1:
        ax.plot(v, I_1, "x--", color="lightsalmon", alpha=0.6,
                label="per motor")
    ax.plot(v, I_tot, "o-", color="tab:red",
            label=f"total ({n} motors)" if n > 1 else None)
    if motor.Imax is not None:
        ax.axhline(motor.Imax, color="grey", ls=":", lw=1,
                   label=f"motor Imax={motor.Imax:g} A")
        ax.axhline(motor.Imax * n, color="k", ls="--", lw=1,
                   label=f"total Imax={motor.Imax * n:g} A")
    ax.set_ylabel("Battery current [A]")
    ax.set_xlabel("Airspeed [m/s]")
    ax.grid(True, alpha=0.3)
    ax.set_title("Current draw")
    if n > 1 or motor.Imax is not None:
        ax.legend(fontsize=8)

    # Individual: current draw
    if ind:
        fi, ai = plt.subplots(figsize=(7, 5))
        if n > 1:
            ai.plot(v, I_1, "x--", color="lightsalmon", alpha=0.6, label="per motor")
        ai.plot(v, I_tot, "o-", color="tab:red",
                label=f"total ({n} motors)" if n > 1 else None)
        if motor.Imax is not None:
            ai.axhline(motor.Imax, color="grey", ls=":", lw=1,
                       label=f"motor Imax={motor.Imax:g} A")
            ai.axhline(motor.Imax * n, color="k", ls="--", lw=1,
                       label=f"total Imax={motor.Imax * n:g} A")
        ai.set_xlabel("Airspeed [m/s]"); ai.set_ylabel("Battery current [A]")
        ai.set_title("Current draw"); ai.grid(True, alpha=0.3)
        if n > 1 or motor.Imax is not None: ai.legend(fontsize=9)
        _save_individual_mp(fi, os.path.join(ind, "05_current_draw.png")); n_ind += 1

    ax = axes[1, 2]
    # Thrust-to-power (g/W) — common figure of merit
    g_per_W = np.where(P_elec_tot > 0,
                       thrust_tot / 9.81 * 1000.0 / P_elec_tot, 0.0)
    ax.plot(v, g_per_W, "o-", color="tab:brown")
    ax.set_ylabel("Thrust / Power [g/W]")
    ax.set_xlabel("Airspeed [m/s]")
    ax.grid(True, alpha=0.3)
    ax.set_title("Propulsive economy")

    # Individual: propulsive economy
    if ind:
        fi, ai = plt.subplots(figsize=(7, 5))
        ai.plot(v, g_per_W, "o-", color="tab:brown")
        ai.set_xlabel("Airspeed [m/s]"); ai.set_ylabel("Thrust / Power [g/W]")
        ai.set_title("Propulsive economy"); ai.grid(True, alpha=0.3)
        _save_individual_mp(fi, os.path.join(ind, "06_propulsive_economy.png")); n_ind += 1

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
        if n_ind:
            print(f"Saved {n_ind} individual plots to {ind}/")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_throttle_sweep(throttles: np.ndarray, ops: list[OperatingPoint],
                        prop: Propeller, motor: Motor, battery: Battery,
                        v_ms: float, soc: float,
                        save_path: str | None = None,
                        show: bool = True) -> None:
    """Plot prop thrust, torque, and shaft power vs throttle at fixed airspeed."""
    import matplotlib.pyplot as plt

    n = motor.count
    thrust_1 = np.array([o.thrust_N for o in ops])
    torque_1 = np.array([o.torque_Nm for o in ops])
    P_shaft_1 = np.array([o.P_shaft_W for o in ops])
    P_elec_1 = np.array([o.P_elec_W for o in ops])
    I_1 = np.array([o.current_A for o in ops])

    thrust_tot = thrust_1 * n
    torque_tot = torque_1 * n
    P_shaft_tot = P_shaft_1 * n
    P_elec_tot = P_elec_1 * n
    I_tot = I_1 * n

    ncols = 3 if n == 1 else 5
    fig, axes = plt.subplots(1, ncols, figsize=(4.3 * ncols, 4))
    title = (f"{prop.name}  +  {motor.name} x{n}  +  "
             f"{battery.series}S{battery.parallel}P {battery.chemistry}  "
             f"(V={v_ms:.1f} m/s, SoC={soc:.2f})")
    fig.suptitle(title, fontsize=12)

    # --- Individual plots directory ---
    ind = _individual_dir_mp(save_path) if save_path else None
    if ind:
        os.makedirs(ind, exist_ok=True)
    n_ind = 0

    # --- Thrust ---
    ax = axes[0]
    if n > 1:
        ax.plot(throttles, thrust_1, "x--", color="tab:cyan", alpha=0.6,
                label="per motor")
    ax.plot(throttles, thrust_tot, "o-", color="tab:blue",
            label=f"total ({n} motors)" if n > 1 else None)
    ax.set_xlabel("Throttle")
    ax.set_ylabel("Thrust [N]")
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.set_title("Thrust vs throttle")
    if n > 1:
        ax.legend(fontsize=8)

    if ind:
        fi, ai = plt.subplots(figsize=(7, 5))
        if n > 1:
            ai.plot(throttles, thrust_1, "x--", color="tab:cyan", alpha=0.6, label="per motor")
        ai.plot(throttles, thrust_tot, "o-", color="tab:blue",
                label=f"total ({n} motors)" if n > 1 else None)
        ai.set_xlabel("Throttle"); ai.set_ylabel("Thrust [N]")
        ai.set_xlim(0, 1); ai.set_title("Thrust vs throttle"); ai.grid(True, alpha=0.3)
        if n > 1: ai.legend(fontsize=9)
        _save_individual_mp(fi, os.path.join(ind, "01_thrust.png")); n_ind += 1

    # --- Torque ---
    ax = axes[1]
    if n > 1:
        ax.plot(throttles, torque_1, "x--", color="moccasin", alpha=0.6,
                label="per motor")
    ax.plot(throttles, torque_tot, "o-", color="tab:orange",
            label=f"total ({n} motors)" if n > 1 else None)
    ax.set_xlabel("Throttle")
    ax.set_ylabel("Torque [N·m]")
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.set_title("Torque vs throttle")
    if n > 1:
        ax.legend(fontsize=8)

    if ind:
        fi, ai = plt.subplots(figsize=(7, 5))
        if n > 1:
            ai.plot(throttles, torque_1, "x--", color="moccasin", alpha=0.6, label="per motor")
        ai.plot(throttles, torque_tot, "o-", color="tab:orange",
                label=f"total ({n} motors)" if n > 1 else None)
        ai.set_xlabel("Throttle"); ai.set_ylabel("Torque [N·m]")
        ai.set_xlim(0, 1); ai.set_title("Torque vs throttle"); ai.grid(True, alpha=0.3)
        if n > 1: ai.legend(fontsize=9)
        _save_individual_mp(fi, os.path.join(ind, "02_torque.png")); n_ind += 1

    # --- Shaft power ---
    ax = axes[2]
    if n > 1:
        ax.plot(throttles, P_shaft_1, "x--", color="lightsalmon", alpha=0.6,
                label="per motor")
    ax.plot(throttles, P_shaft_tot, "o-", color="tab:red",
            label=f"total ({n} motors)" if n > 1 else None)
    if motor.Pmax is not None:
        ax.axhline(motor.Pmax, color="grey", ls=":", lw=1,
                   label=f"motor Pmax={motor.Pmax:g} W")
        ax.axhline(motor.Pmax * n, color="k", ls="--", lw=1,
                   label=f"total Pmax={motor.Pmax * n:g} W")
    ax.set_xlabel("Throttle")
    ax.set_ylabel("Shaft power [W]")
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.set_title("Power vs throttle")
    if n > 1 or motor.Pmax is not None:
        ax.legend(fontsize=8)

    if ind:
        fi, ai = plt.subplots(figsize=(7, 5))
        if n > 1:
            ai.plot(throttles, P_shaft_1, "x--", color="lightsalmon", alpha=0.6, label="per motor")
        ai.plot(throttles, P_shaft_tot, "o-", color="tab:red",
                label=f"total ({n} motors)" if n > 1 else None)
        if motor.Pmax is not None:
            ai.axhline(motor.Pmax, color="grey", ls=":", lw=1,
                       label=f"motor Pmax={motor.Pmax:g} W")
            ai.axhline(motor.Pmax * n, color="k", ls="--", lw=1,
                       label=f"total Pmax={motor.Pmax * n:g} W")
        ai.set_xlabel("Throttle"); ai.set_ylabel("Shaft power [W]")
        ai.set_xlim(0, 1); ai.set_title("Power vs throttle"); ai.grid(True, alpha=0.3)
        if n > 1 or motor.Pmax is not None: ai.legend(fontsize=9)
        _save_individual_mp(fi, os.path.join(ind, "03_shaft_power.png")); n_ind += 1

    if n > 1:
        # --- Total electrical power ---
        ax = axes[3]
        ax.plot(throttles, P_elec_1, "x--", color="lightsalmon", alpha=0.6,
                label="per motor")
        ax.plot(throttles, P_elec_tot, "o-", color="tab:red",
                label=f"total ({n} motors)")
        if motor.Pmax is not None:
            ax.axhline(motor.Pmax, color="grey", ls=":", lw=1,
                       label=f"motor Pmax={motor.Pmax:g} W")
            ax.axhline(motor.Pmax * n, color="k", ls="--", lw=1,
                       label=f"total Pmax={motor.Pmax * n:g} W")
        ax.set_xlabel("Throttle")
        ax.set_ylabel("Elec power [W]")
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.set_title("Elec power vs throttle")
        ax.legend(fontsize=8)

        if ind:
            fi, ai = plt.subplots(figsize=(7, 5))
            ai.plot(throttles, P_elec_1, "x--", color="lightsalmon", alpha=0.6, label="per motor")
            ai.plot(throttles, P_elec_tot, "o-", color="tab:red", label=f"total ({n} motors)")
            if motor.Pmax is not None:
                ai.axhline(motor.Pmax, color="grey", ls=":", lw=1,
                           label=f"motor Pmax={motor.Pmax:g} W")
                ai.axhline(motor.Pmax * n, color="k", ls="--", lw=1,
                           label=f"total Pmax={motor.Pmax * n:g} W")
            ai.set_xlabel("Throttle"); ai.set_ylabel("Elec power [W]")
            ai.set_xlim(0, 1); ai.set_title("Elec power vs throttle")
            ai.grid(True, alpha=0.3); ai.legend(fontsize=9)
            _save_individual_mp(fi, os.path.join(ind, "04_elec_power.png")); n_ind += 1

        # --- Total current ---
        ax = axes[4]
        ax.plot(throttles, I_1, "x--", color="lightsalmon", alpha=0.6,
                label="per motor")
        ax.plot(throttles, I_tot, "o-", color="tab:red",
                label=f"total ({n} motors)")
        if motor.Imax is not None:
            ax.axhline(motor.Imax, color="grey", ls=":", lw=1,
                       label=f"motor Imax={motor.Imax:g} A")
            ax.axhline(motor.Imax * n, color="k", ls="--", lw=1,
                       label=f"total Imax={motor.Imax * n:g} A")
        ax.set_xlabel("Throttle")
        ax.set_ylabel("Current [A]")
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.set_title("Current vs throttle")
        ax.legend(fontsize=8)

        if ind:
            fi, ai = plt.subplots(figsize=(7, 5))
            ai.plot(throttles, I_1, "x--", color="lightsalmon", alpha=0.6, label="per motor")
            ai.plot(throttles, I_tot, "o-", color="tab:red", label=f"total ({n} motors)")
            if motor.Imax is not None:
                ai.axhline(motor.Imax, color="grey", ls=":", lw=1,
                           label=f"motor Imax={motor.Imax:g} A")
                ai.axhline(motor.Imax * n, color="k", ls="--", lw=1,
                           label=f"total Imax={motor.Imax * n:g} A")
            ai.set_xlabel("Throttle"); ai.set_ylabel("Current [A]")
            ai.set_xlim(0, 1); ai.set_title("Current vs throttle")
            ai.grid(True, alpha=0.3); ai.legend(fontsize=9)
            _save_individual_mp(fi, os.path.join(ind, "05_current.png")); n_ind += 1

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
        if n_ind:
            print(f"Saved {n_ind} individual plots to {ind}/")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _auto_plot_path(prop_file: str, motor_xml: str, battery_xml: str,
                    throttle: float, soc: float, plot_dir: str) -> str:
    """Build a deterministic PNG path from the inputs so each combo is unique."""
    def stem(p: str) -> str:
        name = os.path.splitext(os.path.basename(p))[0]
        return re.sub(r"[^A-Za-z0-9._-]+", "_", name)

    fname = (f"{stem(prop_file)}__{stem(motor_xml)}__{stem(battery_xml)}"
             f"__thr{throttle:.2f}_soc{soc:.2f}.png")
    os.makedirs(plot_dir, exist_ok=True)
    return os.path.join(plot_dir, fname)


def _print_table(ops: list[OperatingPoint], n_motors: int = 1) -> None:
    hdr = ("V[m/s]   RPM   Thrust[N] Torque[Nm] Pshaft[W] Pelec[W]  I[A]  "
           "eta_m  eta_p  eta_tot")
    if n_motors > 1:
        hdr += "  | Tot_I[A] Tot_T[N] Tot_Ps[W] Tot_Pe[W]"
    print(hdr)
    print("-" * len(hdr))
    for o in ops:
        line = (f"{o.v_ms:6.2f} {o.rpm:7.0f} {o.thrust_N:9.3f} "
                f"{o.torque_Nm:10.4f} {o.P_shaft_W:9.1f} {o.P_elec_W:8.1f} "
                f"{o.current_A:5.2f} {o.eta_motor:6.3f} {o.eta_prop:6.3f} "
                f"{o.eta_total:7.3f}")
        if n_motors > 1:
            line += (f"  | {o.current_A * n_motors:8.2f} "
                     f"{o.thrust_N * n_motors:8.3f} "
                     f"{o.P_shaft_W * n_motors:9.1f} "
                     f"{o.P_elec_W * n_motors:9.1f}")
        print(line)


def _print_throttle_table(throttles: np.ndarray,
                          ops: list[OperatingPoint],
                          n_motors: int = 1) -> None:
    hdr = ("Thr    V[m/s]   RPM    Voc[V]  Vterm[V] Veff[V] BEMF[V]  "
           "I[A]  Thrust[N] Torque[Nm] Pshaft[W] Pelec[W]  "
           "eta_m  eta_p  eta_tot")
    if n_motors > 1:
        hdr += "  | Tot_I[A] Tot_T[N] Tot_Ps[W] Tot_Pe[W]"
    print(hdr)
    print("-" * len(hdr))
    for t, o in zip(throttles, ops):
        line = (f"{t:5.3f} {o.v_ms:6.2f} {o.rpm:7.0f} "
                f"{o.V_oc:7.2f} {o.V_terminal:8.2f} {o.V_eff:7.2f} "
                f"{o.back_emf:7.2f} {o.current_A:6.2f} "
                f"{o.thrust_N:9.3f} {o.torque_Nm:10.4f} "
                f"{o.P_shaft_W:9.1f} {o.P_elec_W:8.1f} "
                f"{o.eta_motor:6.3f} {o.eta_prop:6.3f} {o.eta_total:7.3f}")
        if n_motors > 1:
            line += (f"  | {o.current_A * n_motors:8.2f} "
                     f"{o.thrust_N * n_motors:8.3f} "
                     f"{o.P_shaft_W * n_motors:9.1f} "
                     f"{o.P_elec_W * n_motors:9.1f}")
        print(line)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("prop_file", help="APC-format propeller data file")
    ap.add_argument("motor_xml", help="Motor spec XML file")
    ap.add_argument("battery_xml", help="Battery spec XML file")
    ap.add_argument("--throttle", type=float, default=None,
                    help="(Ignored — throttle is swept automatically)")
    ap.add_argument("--soc", type=float, default=1.0,
                    help="Battery state of charge 0..1 (default 1.0 = full)")
    ap.add_argument("--vmin", type=float, default=0.0, help="Min airspeed m/s")
    ap.add_argument("--vmax", type=float, default=30.0, help="Max airspeed m/s")
    ap.add_argument("--vstep", type=float, default=2.0, help="Airspeed step m/s")
    ap.add_argument("--plot", action="store_true",
                    help="Also show the plot interactively")
    ap.add_argument("--save-plot", default=None,
                    help="Override the auto-generated plot filename")
    ap.add_argument("--no-plot", action="store_true",
                    help="Skip plot generation entirely")
    ap.add_argument("--plot-dir", default="plots",
                    help="Directory for auto-named plots (default: plots)")
    ap.add_argument("--sweep-v", type=float, default=None,
                    help="Airspeed [m/s] for the throttle sweep plot "
                         "(default: --vmin)")
    ap.add_argument("--sweep-steps", type=int, default=21,
                    help="Number of throttle samples in 0..1 sweep")
    ap.add_argument("--motors", type=int, default=None,
                    help="Number of motors (overrides XML <count>; default 1)")
    args = ap.parse_args()

    prop = load_propeller(args.prop_file)
    motor = load_motor(args.motor_xml)
    if args.motors is not None:
        motor.count = args.motors
    battery = load_battery(args.battery_xml)

    V_oc = battery.V_empty + args.soc * (battery.V_full - battery.V_empty)

    print(f"Propeller: {prop.name}  (D={prop.diameter_in}in, "
          f"P={prop.pitch_in}in)")
    print(f"Motor:     {motor.name}  Kv={motor.Kv} rpm/V  R={motor.R} ohm  "
          f"I0={motor.I0} A  (x{motor.count})")
    print(f"Battery:   {battery.name}  {battery.chemistry} "
          f"{battery.series}S{battery.parallel}P  "
          f"{battery.capacity_Ah:.2f} Ah  "
          f"V_nom={battery.V_nominal:.1f}V  V_oc={V_oc:.2f}V  "
          f"R_int={battery.R_internal*1000:.1f} mohm")
    if battery.I_max is not None:
        print(f"           I_max (C-rating): {battery.I_max:.0f} A")
    print(f"SoC: {args.soc:.2f}\n")

    # Sweep throttle at each airspeed
    v = args.vmin
    all_ops: list[OperatingPoint] = []
    all_throttles: list[float] = []
    all_velocities: list[float] = []
    while v <= args.vmax + 1e-9:
        throttles, ops = sweep_throttle(
            motor, prop, battery, v,
            n_steps=args.sweep_steps, soc=args.soc,
        )
        if ops:
            print(f"\n--- V = {v:.2f} m/s ---")
            _print_throttle_table(throttles, ops, n_motors=motor.count)
            all_ops.extend(ops)
            all_throttles.extend(throttles.tolist())
            all_velocities.extend([v] * len(ops))
        else:
            print(f"\nV = {v:.2f} m/s — no operating points at any throttle.")
        v += args.vstep

    if not all_ops:
        print("\nNo operating points found at any throttle/airspeed "
              "combination.")
        return

    if not args.no_plot:
        for vi in sorted({v for v in all_velocities}):
            idx = [i for i, vv in enumerate(all_velocities) if vv == vi]
            thr = np.array([all_throttles[i] for i in idx])
            ops_v = [all_ops[i] for i in idx]
            if len(ops_v) < 2:
                continue
            save_path = args.save_plot or _auto_plot_path(
                args.prop_file, args.motor_xml, args.battery_xml,
                0.0, args.soc, args.plot_dir,
            )
            base, ext = os.path.splitext(save_path)
            sweep_path = f"{base}__V{vi:.1f}ms__throttle{ext}"
            plot_throttle_sweep(thr, ops_v, prop, motor, battery,
                                v_ms=vi, soc=args.soc,
                                save_path=sweep_path, show=args.plot)


if __name__ == "__main__":
    main()
