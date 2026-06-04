"""Hover-only performance analysis for tailsitters.

Solves the V=0 trim point: throttle, rpm, per-motor current, electrical
power, and endurance from usable battery energy. Forward-flight (cruise,
climb, V_max, control authority) lives in forward_flight_performance.py.

The thrust-balance root-finder ``_find_throttle_for_thrust`` is imported
from forward_flight_performance because it is the general-V form;
hover is just the V=0 case.

Usage
-----
This module is a library; there is no CLI here. Drive it from
``performance.py`` or from your own script:

    from motor_prop_performance import (
        load_propeller, load_motor, load_battery,
    )
    from config_paths import find_config
    from hover_performance import solve_hover, print_hover_section

    prop    = load_propeller(find_config("PER3_7x13E.txt", "propellers"))
    motor   = load_motor(find_config("motor.xml", "motors"))
    battery = load_battery(find_config("battery.xml", "batteries"))

    hover = solve_hover(
        motor, prop, battery,
        mass=4.0,          # kg, total airframe mass
        g=9.81,            # m/s^2
        n_motors=4,        # number of motor+prop pairs
        soc=1.0,           # battery state of charge 0..1
        usable_fraction=0.8,  # fraction of nominal Wh you'll actually use
    )

    if hover is None:
        # Full-throttle static thrust < weight (or solver couldn't converge).
        # Pick a higher-Kv motor, larger prop, or lighter airframe.
        ...
    else:
        # HoverPoint fields: throttle (0..1), rpm, thrust_per_motor_N,
        # current_per_motor_A, P_elec_per_motor_W, P_elec_total_W,
        # endurance_min, thrust_to_weight_max
        print(f"hover throttle = {hover.throttle*100:.1f} %")
        print(f"endurance      = {hover.endurance_min:.1f} min")

    # Optional: format the same block performance.py prints
    plane = {"mass": 4.0, "gravity": 9.81}
    print_hover_section(plane, motor, prop, battery,
                        n_motors=4, hover=hover)

Returns ``None`` (instead of raising) when hover is infeasible -- callers
should branch on that to print a clear failure message.
"""

from __future__ import annotations

from dataclasses import dataclass

from motor_prop_performance import (
    Battery,
    Motor,
    Propeller,
    solve_operating_point,
)
from forward_flight_performance import _find_throttle_for_thrust


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


def print_hover_section(plane: dict, motor: Motor, prop: Propeller,
                        battery: Battery, n_motors: int,
                        hover: HoverPoint | None) -> None:
    """Print the HOVER block (everything between the top header and CRUISE)."""
    mass = plane["mass"]
    g = plane.get("gravity", 9.81)
    W = mass * g

    print("\n--- HOVER ---")
    if hover is None:
        op_full = solve_operating_point(motor, prop, battery, 1.0, 0.0)
        T_full = (op_full.thrust_N * n_motors) if op_full else 0.0
        print(f"  INFEASIBLE - full-throttle static thrust {T_full:.2f} N "
              f"< weight {W:.2f} N")
        print(f"  Thrust-to-weight at full throttle: {T_full/W:.3f}")
        return

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
