"""Coupled airframe + propulsion performance estimator for tailsitters.

Thin CLI orchestrator. Hover-only analysis lives in
``hover_performance.py``; cruise / climb / V_max / control-authority /
altitude analysis lives in ``forward_flight_performance.py``.

Both modules share the airframe + propulsion stack loaded here once.

Usage:
    py -3 performance.py PER3_7x15E.txt motor.xml battery.xml polar.txt \\
        --plane plane.xml --n-motors 1 --vmin 5 --vmax 40 --vstep 1
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from motor_prop_performance import (
    Battery,
    Motor,
    Propeller,
    load_battery,
    load_motor,
    load_propeller,
)
from plot_flow5 import load_plane_xml, load_polar
from config_paths import find_config

from hover_performance import (
    print_hover_section,
    solve_hover,
)
from forward_flight_performance import (
    climb_sweep,
    control_authority_sweep,
    cruise_sweep,
    plot_altitude_speeds,
    plot_forward_performance,
    plot_motor_rpm_vs_airspeed,
    plot_torque_thrust_vs_airspeed,
    print_altitude_speeds,
    print_forward_section,
    stall_speed,
)


# ---------------------------------------------------------------------------
# Top-level report header (airframe + propulsion summary)
# ---------------------------------------------------------------------------


def _print_header(plane: dict, polar: dict, motor: Motor, prop: Propeller,
                  battery: Battery, n_motors: int,
                  usable_fraction: float) -> None:
    mass = plane["mass"]; area = plane["area"]
    rho = plane.get("rho", 1.225); g = plane.get("gravity", 9.81)
    W = mass * g
    E_Wh = battery.V_nominal * battery.capacity_Ah * usable_fraction
    chord = plane.get("chord", 1.0)
    x_cg = plane.get("cg")
    x_ac = plane.get("ac")

    print("=" * 70)
    print(f"Propeller : {prop.name}  ({prop.diameter_in}\" x {prop.pitch_in}\")")
    print(f"Motor     : {motor.name}  Kv={motor.Kv:.0f} rpm/V  R={motor.R*1000:.1f} mohm")
    print(f"Battery   : {battery.name}  {battery.series}S{battery.parallel}P  "
          f"{battery.capacity_Ah:.2f} Ah  V_nom={battery.V_nominal:.1f} V")
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


def _auto_plot_path(prop_file: str, motor_xml: str, battery_xml: str,
                    polar_file: str, out_dir: str) -> str:
    def stem(p: str) -> str:
        return Path(p).stem
    name = (f"perf__{stem(prop_file)}__{stem(motor_xml)}__"
            f"{stem(battery_xml)}__{stem(polar_file)}.png")
    name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)
    return os.path.join(out_dir, name)


def main() -> None:
    """CLI: full flight-envelope performance from a Flow5 polar + powertrain.

    Inputs : Flow5 polar .txt + prop + motor + battery + plane.xml.
    Output : printed hover / cruise / climb / V_max / control-authority tables
             (plus optional plots) - nothing is written back to config/.

    Versus the sibling CLIs:
      * motor_prop_performance.py - propulsion ONLY (no airframe/aero); this
        script adds the airframe and couples lift & drag to the powertrain.
      * sdf_aero_performance.py   - the same envelope but reads aero from the
        Gazebo SDF model instead of a Flow5 polar. Use that to check the sim;
        use THIS to work straight from Flow5 data.
      * plot_flow5.py             - only visualises the polar, no propulsion.
    """
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

    prop = load_propeller(find_config(args.prop_file, "propellers"))
    motor = load_motor(find_config(args.motor_xml, "motors"))
    battery = load_battery(find_config(args.battery_xml, "batteries"))
    polar = load_polar(Path(find_config(args.polar_file, "aero")))
    plane = {"rho": 1.225, "gravity": 9.81}
    plane.update(load_plane_xml(Path(find_config(args.plane, "planes"))))
    if "mass" not in plane or "area" not in plane:
        raise SystemExit(
            f"{args.plane} must define <mass> and <area> for coupled analysis."
        )

    n_motors = args.n_motors if args.n_motors is not None else motor.count
    if n_motors < 1:
        raise SystemExit(f"n_motors must be >= 1 (got {n_motors})")

    V_array = np.arange(args.vmin, args.vmax + 1e-9, args.vstep)

    # --- Compute ---
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

    # --- Report ---
    _print_header(plane, polar, motor, prop, battery, n_motors, args.usable)
    print_hover_section(plane, motor, prop, battery, n_motors, hover)
    print_forward_section(plane, polar, motor, prop, battery, n_motors,
                          cruise, climb, ctrl=ctrl, hover=hover)
    print_altitude_speeds(polar, plane["mass"], plane["area"],
                          g=plane.get("gravity", 9.81))
    print("=" * 70)

    # --- Plot ---
    if not args.no_plot:
        save_path = _auto_plot_path(
            args.prop_file, args.motor_xml, args.battery_xml,
            args.polar_file, args.out,
        )
        plot_forward_performance(plane, polar, motor, prop, battery, n_motors,
                                 cruise, climb,
                                 save_path=save_path, show=args.show,
                                 ctrl=ctrl, hover=hover)
        alt_plot = os.path.splitext(save_path)[0] + "__altitude_speeds.png"
        plot_altitude_speeds(
            polar, plane["mass"], plane["area"],
            g=plane.get("gravity", 9.81),
            save_path=alt_plot, show=args.show,
        )
        rpm_plot = os.path.splitext(save_path)[0] + "__motor_rpm.png"
        plot_motor_rpm_vs_airspeed(
            motor, battery, hover, cruise, climb,
            save_path=rpm_plot, show=args.show,
        )
        tq_plot = os.path.splitext(save_path)[0] + "__torque_thrust.png"
        plot_torque_thrust_vs_airspeed(
            n_motors, hover, cruise, climb,
            save_path=tq_plot, show=args.show,
        )


if __name__ == "__main__":
    main()
