#!/usr/bin/env python3
"""Cross-compare a real-flight log against a SITL log to tune the VITERNA aero model.

Consumes two directories of per-message CSVs produced by Ardu_Log/chop_log.py
(one from a real flight, one from a SITL run that used model-aero-VITERNA-m.sdf)
and reports where the simulated aircraft diverges from the real one, mapped to the
SDF aero coefficient to adjust.

The two flights are never assumed to share a clock:
  * cruise   -> time-independent operating-point signatures (binned by airspeed)
  * transition -> matched on each log's OWN "Transition ... done" event, then aligned
                  by cross-correlation of pitch rate (lag recovered, not assumed)
  * time-to-target-airspeed -> a per-log scalar, compared directly (no alignment)

  * transition tracks -> forward (hover->cruise) and back (cruise->hover) compared
                  separately: XY ground track, position-controller XY overshoot of
                  the commanded stop point, and pitch, real vs SITL. The back
                  transition's overshoot is the post-stall-drag (CDa_stall) probe.

The MODEL UPDATE is fitted on the battery POWER channel (measured pack power
inverted through the motor/prop/battery model to thrust, then
CD = CD0 + CL^2/(pi*AR*eff) regressed over every level-cruise sample);
thrust-from-RPM is kept only as a cross-check. Coefficients the flight cannot
constrain are held at the Flow5 prior.

Outputs (into --outdir): drag-signature plot, lift/trim plot, power/current
plot, transition overlay + tracks plots, report.txt, metrics.json (every
scalar, machine-readable) and a row in the run ledger CSV (score + coefficient
history across runs, default <outdir>/../compare_runs.csv). --write-sdf also
emits a copy of the SDF with the fitted revisions applied.

--real / --sitl accept a chop_log CSV directory OR a raw .bin log (auto-chopped
into <outdir>/<logname>_chopped, reused unless --rechop). --sitl also accepts
'latest' / 'latest-N': the newest (N-back) SITL log via LASTLOG.TXT.
Defaults for the recurring arguments come from config/compare.ini ([compare]
section, CLI wins), so the routine run is just:

    python compare_logs.py                 # real + latest SITL log, all defaults
    python compare_logs.py --sitl latest-1 --write-sdf

Fully explicit:
    python compare_logs.py \
        --real /home/alan/Ardu_Log/2026-5-22-16-07-32-2nd-aircraft-fw.bin \
        --sitl /home/alan/ardupilot/logs/00000103.BIN \
        --sdf  model-aero-VITERNA-m.sdf \
        --prop /home/alan/uav_sim_workspace/ardupilot_gazebo/models/waterdrop/propellers/PER3_7x11E.csv \
        --outdir plots/compare --no-show
"""
from __future__ import annotations

import argparse
import bisect
import configparser
import csv
import datetime
import importlib.util
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field

import numpy as np

from sdf_aero_performance import (
    AeroModel,
    CD_of_alpha,
    CL_of_alpha,
    alpha_trim,
    cruise_sweep,
    load_propeller_csv,
    load_sdf_model,
)
from hover_performance import solve_hover
from motor_prop_performance import load_battery, load_motor, solve_operating_point
from config_paths import find_config, CONFIG_DIR

G = 9.80665


# ---------------------------------------------------------------------------
# Log loading -- a directory of <TYPE>.csv files from chop_log.py
# ---------------------------------------------------------------------------
@dataclass
class LogData:
    name: str
    path: str
    msgs: dict = field(default_factory=dict)      # type -> {col: np.ndarray}
    events: list = field(default_factory=list)     # [(rel_time, text)]
    params: dict = field(default_factory=dict)     # PARM name -> float
    t0: float | None = None                        # absolute timestamp of log zero

    def has(self, mtype: str, *cols: str) -> bool:
        if mtype not in self.msgs:
            return False
        return all(c in self.msgs[mtype] for c in cols)

    def col(self, mtype: str, name: str) -> np.ndarray:
        return self.msgs[mtype][name]

    def rel(self, mtype: str) -> np.ndarray:
        return self.msgs[mtype]["rel_time"]

    def event_rel(self, *substrings: str) -> float | None:
        """rel_time of the first message containing ALL given substrings (ci)."""
        for t, text in self.events:
            low = text.lower()
            if all(s.lower() in low for s in substrings):
                return t
        return None


def _load_chop_log():
    """Import Ardu_Log/chop_log.py from a sibling/home location."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(os.path.dirname(here), "Ardu_Log"),   # workspace/Ardu_Log
        os.path.expanduser("~/Ardu_Log"),
    ]
    for d in candidates:
        p = os.path.join(d, "chop_log.py")
        if os.path.exists(p):
            spec = importlib.util.spec_from_file_location("chop_log", p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise SystemExit(f"could not find chop_log.py (looked in {candidates})")


def ensure_chopped(path: str, name: str, outdir: str, tier: int, hz: float,
                   rechop: bool) -> str:
    """Accept either a chop_log CSV directory or a raw .bin log.

    A directory is used as-is. A .bin file is chopped (full data: tier 3 +
    all sensors by default) into <outdir>/<logname>_chopped and reused on
    later runs unless --rechop is given.
    """
    path = os.path.abspath(os.path.expanduser(path))
    if os.path.isdir(path):
        return path
    if not os.path.isfile(path):
        raise SystemExit(f"[{name}] not a file or directory: {path}")
    stem = os.path.splitext(os.path.basename(path))[0]
    out = os.path.join(os.path.abspath(outdir), f"{stem}_chopped")
    if os.path.isdir(out) and os.listdir(out) and not rechop:
        print(f"[{name}] reusing chopped dir {out}  (--rechop to regenerate)")
        return out
    print(f"[{name}] chopping {path}\n        -> {out}  (tier {tier}, all sensors, full time)")
    cl = _load_chop_log()
    cl.chop_log(path, out, 0.0, 1e12, downsample_hz=hz, all_sensors=True,
                tier=tier, group_pids=False, group_psc=False)
    return out


def load_log(path: str, name: str) -> LogData:
    if not os.path.isdir(path):
        raise SystemExit(f"[{name}] not a directory: {path}")
    log = LogData(name=name, path=path)

    for fn in sorted(os.listdir(path)):
        if not fn.endswith(".csv"):
            continue
        mtype = fn[:-4]
        if mtype == "params":
            continue
        cols: dict[str, list] = {}
        with open(os.path.join(path, fn), newline="") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                continue
            for h in header:
                cols[h] = []
            for row in reader:
                for h, v in zip(header, row):
                    cols[h].append(v)
        if not cols.get("rel_time"):
            continue
        arrs: dict[str, np.ndarray] = {}
        for h, vals in cols.items():
            try:
                arrs[h] = np.array([float(x) for x in vals], dtype=float)
            except ValueError:
                arrs[h] = np.array(vals, dtype=object)
        log.msgs[mtype] = arrs

    # Establish absolute->relative offset from any message that has both.
    for arrs in log.msgs.values():
        if "timestamp" in arrs and "rel_time" in arrs and len(arrs["rel_time"]):
            log.t0 = float(arrs["timestamp"][0]) - float(arrs["rel_time"][0])
            break

    # messages.txt: "<abs_ts>\t<text>" -> (rel_time, text)
    msg_path = os.path.join(path, "messages.txt")
    if os.path.exists(msg_path) and log.t0 is not None:
        with open(msg_path) as f:
            for line in f:
                parts = line.rstrip("\n").split("\t", 1)
                if len(parts) != 2:
                    continue
                try:
                    abs_ts = float(parts[0])
                except ValueError:
                    continue
                log.events.append((abs_ts - log.t0, parts[1]))
        log.events.sort(key=lambda e: e[0])

    # params.parm: "name,value" lines
    parm_path = os.path.join(path, "params.parm")
    if os.path.exists(parm_path):
        with open(parm_path) as f:
            for line in f:
                if "," not in line:
                    continue
                k, v = line.rstrip("\n").split(",", 1)
                try:
                    log.params[k.strip()] = float(v)
                except ValueError:
                    pass
    return log


# ---------------------------------------------------------------------------
# Propeller thrust table: (rpm, airspeed) -> thrust per prop (N)
# (CSV format produced for PropellerPerformancePlugin: rpm,v_ms,thrust_N,torque_Nm)
# ---------------------------------------------------------------------------
class PropTable:
    def __init__(self, path: str):
        rows = []
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                try:
                    rows.append((float(r["rpm"]), float(r["v_ms"]),
                                 float(r["thrust_N"])))
                except (KeyError, ValueError):
                    continue
        if not rows:
            raise SystemExit(f"no prop data in {path}")
        rows.sort(key=lambda x: (x[0], x[1]))
        self.max_rpm = rows[-1][0]
        self.rpms: list[float] = []
        self.speeds: list[list[float]] = []
        self.thrust: list[list[float]] = []
        cur = None
        for rpm, v, t in rows:
            if rpm != cur:
                self.rpms.append(rpm)
                self.speeds.append([])
                self.thrust.append([])
                cur = rpm
            self.speeds[-1].append(v)
            self.thrust[-1].append(t)

    @staticmethod
    def _interp(xs, ys, x):
        if not xs:
            return 0.0
        if x <= xs[0]:
            return ys[0]
        if x >= xs[-1]:
            return ys[-1]
        hi = bisect.bisect_right(xs, x)
        lo = hi - 1
        f = (x - xs[lo]) / (xs[hi] - xs[lo])
        return ys[lo] + f * (ys[hi] - ys[lo])

    def thrust_at(self, rpm: float, v: float) -> float:
        if not self.rpms:
            return 0.0
        if rpm <= self.rpms[0]:
            lo = hi = 0
            f = 0.0
        elif rpm >= self.rpms[-1]:
            lo = hi = len(self.rpms) - 1
            f = 0.0
        else:
            hi = bisect.bisect_right(self.rpms, rpm)
            lo = hi - 1
            f = (rpm - self.rpms[lo]) / (self.rpms[hi] - self.rpms[lo])
        t_lo = self._interp(self.speeds[lo], self.thrust[lo], v)
        if lo == hi:
            return t_lo
        t_hi = self._interp(self.speeds[hi], self.thrust[hi], v)
        return t_lo + f * (t_hi - t_lo)


# ---------------------------------------------------------------------------
# Phase boundaries from each log's own transition events / state
# ---------------------------------------------------------------------------
@dataclass
class Phases:
    t_fw: float | None       # rel_time hover->cruise transition completed
    t_vtol: float | None     # rel_time cruise->hover transition completed
    cruise_lo: float | None  # cruise window [lo, hi]
    cruise_hi: float | None
    note: str = ""


def detect_phases(log: LogData) -> Phases:
    t_fw = log.event_rel("transition", "fw")
    if t_fw is None:
        t_fw = log.event_rel("transition", "done")  # generic fallback
    t_vtol = log.event_rel("transition", "vtol")
    note = ""
    if t_fw is None:
        note = "no 'Transition FW done' event found; cruise window inferred from airspeed"
    return Phases(t_fw=t_fw, t_vtol=t_vtol,
                  cruise_lo=t_fw, cruise_hi=t_vtol, note=note)


# ---------------------------------------------------------------------------
# Airspeed / climb-rate helpers (prefer ARSP, fall back to EKF/GPS)
# ---------------------------------------------------------------------------
def airspeed_series(log: LogData):
    if log.has("ARSP", "Airspeed"):
        return log.rel("ARSP"), log.col("ARSP", "Airspeed"), "ARSP.Airspeed"
    if log.has("CTUN", "As"):
        return log.rel("CTUN"), log.col("CTUN", "As"), "CTUN.As"
    if log.has("XKF1", "VN", "VE"):
        spd = np.hypot(log.col("XKF1", "VN"), log.col("XKF1", "VE"))
        return log.rel("XKF1"), spd, "XKF1 groundspeed (no airspeed!)"
    if log.has("GPS", "Spd"):
        return log.rel("GPS"), log.col("GPS", "Spd"), "GPS.Spd groundspeed (no airspeed!)"
    return None, None, None


def climbrate_series(log: LogData):
    if log.has("TECS", "dh"):
        return log.rel("TECS"), log.col("TECS", "dh")
    if log.has("XKF1", "VD"):
        return log.rel("XKF1"), -log.col("XKF1", "VD")
    return None, None


_EARTH_R = 6378137.0  # WGS84 equatorial radius (m), good enough for a local plane


def position_ned_series(log: LogData):
    """Local NED position (north, east, down) in metres vs time.

    Prefer the EKF estimate XKF1.PN/PE/PD (already in a local NED frame relative
    to the EKF origin, primary core only). Fall back to POS lat/lng/alt projected
    onto a tangent plane at the first sample, then AHR2. Returns
    (rel_time, north, east, down, source) or five Nones.
    """
    if log.has("XKF1", "PN", "PE", "PD"):
        rel = log.rel("XKF1")
        N, E, D = (log.col("XKF1", c) for c in ("PN", "PE", "PD"))
        if "C" in log.msgs["XKF1"]:                      # one row per EKF core
            core = log.col("XKF1", "C")
            vals, counts = np.unique(core, return_counts=True)
            sel = core == vals[int(np.argmax(counts))]   # busiest (primary) core
            rel, N, E, D = rel[sel], N[sel], E[sel], D[sel]
        return rel, N, E, D, "XKF1.PN/PE/PD"
    for mtype, alt in (("POS", "Alt"), ("AHR2", "Alt")):
        if log.has(mtype, "Lat", "Lng", alt):
            rel = log.rel(mtype)
            lat, lng = log.col(mtype, "Lat"), log.col(mtype, "Lng")
            alt_m = log.col(mtype, alt)
            lat0 = float(lat[0])
            north = np.radians(lat - lat0) * _EARTH_R
            east = np.radians(lng - float(lng[0])) * _EARTH_R * math.cos(math.radians(lat0))
            return rel, north, east, -(alt_m - float(alt_m[0])), f"{mtype} lat/lng/alt"
    return None, None, None, None, None


def pos_track_error_series(log: LogData):
    """Horizontal position-controller tracking error |actual - target| (m) vs time.

    From PSCN/PSCE (the QuadPlane/VTOL position controller, active through the back
    transition): PN/PE are the achieved NE position, TPN/TPE the commanded target.
    Their horizontal distance is exactly the XY overshoot of the commanded stop
    point. Returns (rel_time, err_m, source) or three Nones (fixed-wing-only logs
    won't carry PSC).
    """
    if log.has("PSCN", "PN", "TPN") and log.has("PSCE", "PE", "TPE"):
        rel = log.rel("PSCN")
        eN = log.col("PSCN", "PN") - log.col("PSCN", "TPN")
        eE = interp_at(log.rel("PSCE"),
                       log.col("PSCE", "PE") - log.col("PSCE", "TPE"), rel)
        return rel, np.hypot(eN, eE), "PSCN/PSCE actual-target"
    return None, None, None


def interp_at(t_src, y_src, t_query):
    """np.interp guarding against unsorted / empty inputs."""
    if t_src is None or len(t_src) == 0:
        return np.full(len(t_query), np.nan)
    order = np.argsort(t_src)
    return np.interp(t_query, np.asarray(t_src)[order], np.asarray(y_src)[order])


def battery_series(log: LogData):
    """Propulsion-pack telemetry (rel_time, current A, volt V, source).

    With multiple BAT instances the propulsion pack is taken as the instance
    drawing the most current (avoids picking an avionics/secondary pack that
    reads ~0 A). Returns four Nones when BAT.Curr/Volt is not logged.
    """
    if not log.has("BAT", "Curr", "Volt"):
        return None, None, None, None
    bt = log.rel("BAT")
    curr, volt = log.col("BAT", "Curr"), log.col("BAT", "Volt")
    src = "BAT"
    if "Inst" in log.msgs["BAT"]:
        inst = log.col("BAT", "Inst")
        best, binst = -1.0, -1
        for i in sorted(set(int(x) for x in inst)):
            med = float(np.nanmedian(np.abs(curr[inst == i])))
            if med > best:
                best, binst = med, i
        sel = inst == binst
        bt, curr, volt = bt[sel], curr[sel], volt[sel]
        src = f"BAT[{binst}]"
    return bt, curr, volt, src


# ---------------------------------------------------------------------------
# Measured total thrust at cruise sample times, from ESC RPM (fallback RCOU PWM)
# ---------------------------------------------------------------------------
def total_thrust(log: LogData, t_query: np.ndarray, v_query: np.ndarray,
                 prop: PropTable, sdf_prop_info: dict):
    """Total thrust over all rotors at each query time. Returns (thrust, source).

    Real logs may telemeter only one ESC instance, so we average per-rotor thrust
    over the instances that DO have clean data and scale by the SDF rotor count,
    rather than summing whatever happens to be logged.
    """
    n_motors = int(sdf_prop_info.get("n_motors", 4)) or 4
    ceiling = prop.max_rpm * 1.1  # reject corrupt RPM spikes (e.g. 7.8e5)

    # Preferred: ESC.RPM, split by Instance.
    if log.has("ESC", "RPM"):
        inst = log.col("ESC", "Instance") if "Instance" in log.msgs["ESC"] else None
        rpm = log.col("ESC", "RPM")
        rel = log.rel("ESC")
        per_rotor = []  # one thrust(t_query) array per usable instance
        instances = (sorted(set(int(x) for x in inst)) if inst is not None else [None])
        for i in instances:
            sel = np.ones(len(rpm), bool) if i is None else (inst == i)
            sel &= (rpm > 0) & (rpm <= ceiling)
            if sel.sum() < 5:
                continue
            rpm_i = interp_at(rel[sel], rpm[sel], t_query)
            thr_i = np.array([prop.thrust_at(rpm_i[k], max(v_query[k], 0.0))
                              for k in range(len(t_query))])
            per_rotor.append(thr_i)
        if per_rotor:
            mean_rotor = np.mean(per_rotor, axis=0)
            return mean_rotor * n_motors, f"ESC.RPM mean of {len(per_rotor)} x {n_motors} motors"

    # Fallback: RCOU PWM -> commanded rad/s via ArduPilotPlugin multiplier -> rpm.
    if log.has("RCOU", "C1"):
        mult = sdf_prop_info.get("max_rad_per_s", 2100.0)
        smin, smax = 1000.0, 2000.0
        rel = log.rel("RCOU")
        total = np.zeros(len(t_query))
        n = 0
        for ch in ("C1", "C2", "C3", "C4"):
            if ch not in log.msgs["RCOU"]:
                continue
            pwm = log.col("RCOU", ch)
            rads = np.clip((pwm - smin) / (smax - smin), 0, 1) * mult
            rpm_ch = rads * 60.0 / (2.0 * math.pi)
            rpm_i = interp_at(rel, rpm_ch, t_query)
            for k in range(len(t_query)):
                total[k] += prop.thrust_at(rpm_i[k], max(v_query[k], 0.0))
            n += 1
        return total, f"RCOU PWM x{n} (no ESC RPM)"

    return None, None


# ---------------------------------------------------------------------------
# Cruise operating-point signature: medians binned by airspeed
# ---------------------------------------------------------------------------
@dataclass
class CruiseSig:
    v: np.ndarray
    thrust: np.ndarray
    aoa: np.ndarray
    pitch: np.ndarray
    throttle: np.ndarray
    thrust_src: str
    aoa_src: str
    n_samples: int
    note: str = ""
    current: np.ndarray = None      # pack current (A), per airspeed bin
    power: np.ndarray = None        # electrical pack power (W), per bin
    elec_src: str = "n/a"
    # per-sample (unbinned) level-cruise arrays, for regression fits:
    # keys t, v, thrust, aoa, pitch, current, power, volt (aligned, may hold NaN)
    raw: dict = None


_FIT_ROLL_MAX = 45.0   # deg; beyond this the coordinated-turn CL correction is shaky


def cruise_signature(log: LogData, prop: PropTable, sdf_prop_info: dict,
                     v_lo: float, v_hi: float, v_step: float,
                     max_climb: float, max_roll: float = 10.0,
                     max_accel: float = 0.5) -> CruiseSig | None:
    ph = detect_phases(log)
    t_arsp, v_arsp, vsrc = airspeed_series(log)
    if t_arsp is None:
        print(f"  [{log.name}] no airspeed/velocity signal -> skip cruise signature")
        return None

    # cruise time mask on the airspeed grid
    mask = np.ones(len(t_arsp), dtype=bool)
    note = ph.note
    if ph.cruise_lo is not None:
        mask &= t_arsp >= ph.cruise_lo
    if ph.cruise_hi is not None:
        mask &= t_arsp <= ph.cruise_hi
    mask &= v_arsp >= v_lo

    # level-flight filter via climb rate
    tcr, cr = climbrate_series(log)
    if tcr is not None:
        cr_q = interp_at(tcr, cr, t_arsp)
        mask &= np.abs(cr_q) <= max_climb

    # quasi-steady filter: T = D only holds while V is steady, so drop
    # acceleration transients (~1 s smoothed dV/dt)
    if len(t_arsp) > 5:
        dt_med = float(np.median(np.diff(t_arsp))) or 0.02
        w = max(1, int(round(1.0 / dt_med)))
        v_smooth = np.convolve(v_arsp, np.ones(w) / w, mode="same")
        mask &= np.abs(np.gradient(v_smooth, t_arsp)) <= max_accel

    # Banked turns: CL = n*W/(q*S) with n = 1/cos(roll) (coordinated, level).
    # The binned signature keeps only wings-level samples (|roll|<=max_roll, so
    # real and SITL bins are the same operating point); the regression keeps
    # turning samples up to _FIT_ROLL_MAX and corrects CL by the load factor --
    # a constant-V turn is a legitimate CL sweep.
    roll_q = None
    if log.has("ATT", "Roll"):
        roll_q = interp_at(log.rel("ATT"), log.col("ATT", "Roll"), t_arsp)
        mask &= np.abs(roll_q) <= _FIT_ROLL_MAX

    t_q = t_arsp[mask]
    v_q = v_arsp[mask]
    if roll_q is not None:
        roll_k = roll_q[mask]
        load = 1.0 / np.cos(np.radians(np.clip(roll_k, -60.0, 60.0)))
        level_sel = np.abs(roll_k) <= max_roll
    else:
        load = np.ones(len(t_q))
        level_sel = np.ones(len(t_q), dtype=bool)
    if len(t_q) < 10:
        print(f"  [{log.name}] only {len(t_q)} cruise samples -> skip")
        return None

    thr, thr_src = total_thrust(log, t_q, v_q, prop, sdf_prop_info)

    aoa_src = "n/a"
    if log.has("AOA", "AOA"):
        aoa_q = interp_at(log.rel("AOA"), log.col("AOA", "AOA"), t_q)
        aoa_src = "AOA.AOA"
    elif log.has("ATT", "Pitch"):
        aoa_q = interp_at(log.rel("ATT"), log.col("ATT", "Pitch"), t_q)
        aoa_src = "ATT.Pitch (AoA proxy)"
        note = (note + "; " if note else "") + "no AOA log -> pitch used as AoA proxy"
    else:
        aoa_q = np.full(len(t_q), np.nan)

    pitch_q = (interp_at(log.rel("ATT"), log.col("ATT", "Pitch"), t_q)
               if log.has("ATT", "Pitch") else np.full(len(t_q), np.nan))

    if log.has("CTUN", "ThO"):
        thr_pct = interp_at(log.rel("CTUN"), log.col("CTUN", "ThO"), t_q)
    elif log.has("TECS", "th"):
        thr_pct = interp_at(log.rel("TECS"), log.col("TECS", "th"), t_q) * 100.0
    else:
        thr_pct = np.full(len(t_q), np.nan)

    # battery pack current (A) and electrical power (W = V*I)
    bt, curr_raw, volt_raw, elec_src = battery_series(log)
    if bt is not None:
        cur_q = interp_at(bt, curr_raw, t_q)
        volt_q = interp_at(bt, volt_raw, t_q)
        pow_q = cur_q * volt_q
    else:
        elec_src = "n/a"
        cur_q = np.full(len(t_q), np.nan)
        volt_q = np.full(len(t_q), np.nan)
        pow_q = np.full(len(t_q), np.nan)

    # bin by airspeed, take medians (wings-level samples only, so the real and
    # SITL bins describe the same operating point)
    edges = np.arange(v_lo, v_hi + v_step, v_step)
    centers, t_b, a_b, p_b, q_b, cu_b, pw_b = [], [], [], [], [], [], []
    for i in range(len(edges) - 1):
        b = level_sel & (v_q >= edges[i]) & (v_q < edges[i + 1])
        if b.sum() < 3:
            continue
        centers.append(0.5 * (edges[i] + edges[i + 1]))
        t_b.append(np.nanmedian(thr[b]) if thr is not None else np.nan)
        a_b.append(np.nanmedian(aoa_q[b]))
        p_b.append(np.nanmedian(pitch_q[b]))
        q_b.append(np.nanmedian(thr_pct[b]))
        cu_b.append(np.nanmedian(cur_q[b]))
        pw_b.append(np.nanmedian(pow_q[b]))
    if not centers:
        print(f"  [{log.name}] no populated airspeed bins -> skip")
        return None

    print(f"  [{log.name}] cruise: {int(level_sel.sum())} level + "
          f"{int((~level_sel).sum())} banked samples, {len(centers)} bins "
          f"(airspeed {vsrc}, thrust {thr_src}, elec {elec_src})")
    raw = {"t": t_q, "v": v_q, "load": load,
           "thrust": thr if thr is not None else np.full(len(t_q), np.nan),
           "aoa": aoa_q, "pitch": pitch_q,
           "current": cur_q, "power": pow_q, "volt": volt_q}
    return CruiseSig(np.array(centers), np.array(t_b), np.array(a_b),
                     np.array(p_b), np.array(q_b), thr_src, aoa_src,
                     len(t_q), note, current=np.array(cu_b),
                     power=np.array(pw_b), elec_src=elec_src, raw=raw)


# ---------------------------------------------------------------------------
# Hover electrical signature: median battery draw while hovering (V ~ 0)
# ---------------------------------------------------------------------------
@dataclass
class HoverSig:
    current: float        # median pack current (A)
    power: float          # median electrical power (W = V*I)
    volt: float           # median pack voltage (V)
    n_samples: int
    elec_src: str
    note: str = ""


def hover_signature(log: LogData, max_climb: float, v_hover_max: float = 4.0,
                    settle: float = 3.0, min_alt: float = 2.0) -> HoverSig | None:
    """Median battery draw in hover: in-air, near-zero airspeed, level.

    Hover windows come from the log's own transition events: everything
    before 'Transition FW done' (takeoff hover) plus everything after
    'Transition VTOL done' (landing hover), each trimmed by `settle` seconds.
    In-air is gated on altitude above the first position sample; without a
    position log it falls back to current above 25% of the window peak
    (drops disarmed/ground-idle samples that would drag the median down).
    """
    bt, curr, volt, src = battery_series(log)
    if bt is None:
        print(f"  [{log.name}] no BAT.Curr/Volt -> skip hover signature")
        return None
    ph = detect_phases(log)
    mask = np.zeros(len(bt), bool)
    note = ""
    if ph.t_fw is not None:
        mask |= bt <= ph.t_fw - settle
    if ph.t_vtol is not None:
        mask |= bt >= ph.t_vtol + settle
    if ph.t_fw is None and ph.t_vtol is None:
        mask[:] = True
        note = "no transition events; hover = low-airspeed samples of whole log"

    t_a, v_a, _ = airspeed_series(log)
    if t_a is not None:
        mask &= interp_at(t_a, v_a, bt) <= v_hover_max
    tcr, cr = climbrate_series(log)
    if tcr is not None:
        mask &= np.abs(interp_at(tcr, cr, bt)) <= max_climb

    t_pos, _, _, down, _ = position_ned_series(log)
    if t_pos is not None:
        alt = -(interp_at(t_pos, down, bt) - float(down[0]))
        mask &= alt >= min_alt
    elif mask.any():
        mask &= curr >= 0.25 * float(np.nanmax(curr[mask]))
        note = (note + "; " if note else "") + "no position log; in-air gated on current"

    if mask.sum() < 5:
        print(f"  [{log.name}] only {int(mask.sum())} hover samples -> skip hover signature")
        return None
    cur_med = float(np.nanmedian(curr[mask]))
    pow_med = float(np.nanmedian(curr[mask] * volt[mask]))
    print(f"  [{log.name}] hover: {int(mask.sum())} samples, "
          f"{cur_med:.1f} A / {pow_med:.0f} W ({src})")
    return HoverSig(current=cur_med, power=pow_med,
                    volt=float(np.nanmedian(volt[mask])),
                    n_samples=int(mask.sum()), elec_src=src, note=note)


def predicted_hover(prop_csv: str, motor_xml: str, battery_xml: str,
                    n_motors: int, mass: float):
    """Model hover point (pack current A, power W, throttle 0..1) from solve_hover.

    Same powertrain solve as sdf_aero_performance.py's hover block; None when
    inputs are missing or full-throttle static thrust < weight.
    """
    if not (motor_xml and battery_xml):
        return None
    h = solve_hover(load_motor(motor_xml), load_propeller_csv(prop_csv),
                    load_battery(battery_xml), mass, n_motors=n_motors)
    if h is None:
        return None
    return h.current_per_motor_A * n_motors, h.P_elec_total_W, h.throttle


@dataclass
class Revision:
    tag: str
    old: float
    new: float            # NaN when held at prior
    kind: str             # 'identified' | 'combination' | 'held'
    note: str
    sigma: float = float("nan")
    channel: str = ""     # measurement the number rests on


# Identifiability thresholds: what variation the data must show to separate
# coupled parameters (equation-error sense). Below these, the coupled pair is
# only known as a combination, so we move the offset onto the leading term and
# hold its partner at the Flow5 prior.
_CL_SPREAD_MIN = 0.25     # (CLmax-CLmin)/CLmean to split CD0 from induced (eff)
_AOA_SPAN_MIN = 3.0       # deg of AoA range to split CL0 from CLa
_MIN_FIT_SAMPLES = 20     # per-sample regression minimum


class PowerInverter:
    """Powertrain model inverted: (V, P_elec per motor) -> thrust per motor (N).

    solve_operating_point is tabulated once over a (V, throttle) grid; lookups
    then interpolate measured electrical power onto thrust along the throttle
    axis at the two bracketing airspeeds. Power outside the tabulated envelope
    (below idle / above full throttle) returns NaN rather than extrapolating.
    """

    def __init__(self, motor, prop, battery, v_lo: float, v_hi: float,
                 soc: float = 1.0, nv: int = 21, nthr: int = 33):
        self.soc = soc
        self.Vs = np.linspace(max(v_lo, 0.5), max(v_hi, v_lo + 1.0), nv)
        self.tabs = []
        for V in self.Vs:
            P, T = [], []
            for thr in np.linspace(0.02, 1.0, nthr):
                op = solve_operating_point(motor, prop, battery, float(thr),
                                           float(V), soc=soc)
                if op is None:
                    continue
                P.append(op.P_elec_W)
                T.append(op.thrust_N)
            if len(P) >= 2:
                P, T = np.array(P), np.array(T)
                order = np.argsort(P)
                self.tabs.append((P[order], T[order]))
            else:
                self.tabs.append(None)

    def _row(self, i: int, P: float) -> float:
        tab = self.tabs[i]
        if tab is None or P < tab[0][0] or P > tab[0][-1]:
            return float("nan")
        return float(np.interp(P, tab[0], tab[1]))

    def thrust_at(self, V: float, P_per_motor: float) -> float:
        if V <= self.Vs[0]:
            return self._row(0, P_per_motor)
        if V >= self.Vs[-1]:
            return self._row(len(self.Vs) - 1, P_per_motor)
        hi = int(np.searchsorted(self.Vs, V))
        lo = hi - 1
        f = (V - self.Vs[lo]) / (self.Vs[hi] - self.Vs[lo])
        t_lo, t_hi = self._row(lo, P_per_motor), self._row(hi, P_per_motor)
        return t_lo + f * (t_hi - t_lo)


def _soc_from_volt(battery, volt_med: float, current_med: float,
                   n_motors: int) -> float:
    """Model SoC whose open-circuit voltage matches the flight's measured pack
    voltage (terminal volt + per-motor current * R_internal sag)."""
    if not (np.isfinite(volt_med) and np.isfinite(current_med)):
        return 1.0
    v_oc = volt_med + (current_med / max(n_motors, 1)) * battery.R_internal
    rng = battery.V_full - battery.V_empty
    if rng <= 0:
        return 1.0
    return min(1.0, max(0.05, (v_oc - battery.V_empty) / rng))


def _ols_line(x: np.ndarray, y: np.ndarray):
    """Least-squares y = b0 + b1*x -> (b0, b1, sigma_b0, sigma_b1)."""
    n = len(x)
    A = np.column_stack([np.ones(n), x])
    coef, _, rank, _ = np.linalg.lstsq(A, y, rcond=None)
    b0, b1 = float(coef[0]), float(coef[1])
    dof = n - 2
    if dof <= 0 or rank < 2:
        return b0, b1, float("nan"), float("nan")
    r = y - A @ coef
    s2 = float(r @ r) / dof
    cov = s2 * np.linalg.inv(A.T @ A)
    return b0, b1, math.sqrt(cov[0, 0]), math.sqrt(cov[1, 1])


def recommend_revisions(real_sig: "CruiseSig | None", aero: AeroModel,
                        mass: float, powertrain: dict | None = None,
                        fit_dt: float = 0.5, v_step: float = 2.0):
    """Identifiability-aware model update, fitted on the per-sample time series.

    Drag is estimated on the BATTERY POWER channel when a powertrain model
    (dict with motor/prop/battery/n_motors) is supplied: measured pack power is
    inverted through the powertrain to thrust, hence CD, and
    CD = CD0 + CL^2/(pi*AR*eff) is regressed over all level-cruise samples.
    Thrust-from-RPM serves only as a cross-check (single ESC instance + RPM
    spikes make it the noisy channel); it becomes the primary only when no
    powertrain model or battery telemetry exists.

    Samples are decimated to ~1/fit_dt Hz so the OLS sigmas aren't shrunk by
    serial correlation. Revisions are tagged 'identified' (the data span
    resolves the term), 'combination' (only a coupled sum is constrained ->
    offset on the leading term, partner held) or 'held' (keep Flow5 prior).
    """
    revs: list[Revision] = []
    diag = {"n_fit": 0, "eff_identifiable": False, "cla_identifiable": False,
            "drag_channel": None}
    if real_sig is None or real_sig.raw is None or len(real_sig.raw["t"]) < 2:
        return revs, diag
    raw = real_sig.raw
    t = raw["t"]
    step = max(1, int(round(fit_dt / max(float(np.median(np.diff(t))), 1e-3))))
    sl = slice(None, None, step)
    v = raw["v"][sl]
    power, volt, current = raw["power"][sl], raw["volt"][sl], raw["current"][sl]
    aoa, thrust_rpm = raw["aoa"][sl], raw["thrust"][sl]
    load = raw.get("load")
    load = load[sl] if load is not None else np.ones(len(v))

    W, S = mass * G, aero.area
    ok_v = v > 1.0
    q = 0.5 * aero.rho * v * v
    # lift = n*W in a coordinated level turn (n from roll) -- banked samples
    # are kept on purpose: a constant-V turn sweeps CL, which is what separates
    # induced from parasitic drag
    CL = np.where(ok_v, load * W / np.where(ok_v, q * S, 1.0), np.nan)
    x_ind = CL * CL / (math.pi * aero.AR)          # CD = CD0 + x_ind/eff

    base = ok_v & np.isfinite(CL)
    if base.any():
        CLb = CL[base]
        diag["V"] = (float(v[base].min()), float(v[base].max()))
        diag["CL"] = (float(CLb.min()), float(CLb.max()))
        diag["cl_spread"] = (float((CLb.max() - CLb.min()) / CLb.mean())
                             if CLb.mean() else 0.0)

    # ---- per-sample thrust on the power channel (primary) ----
    T_pow = np.full(len(v), np.nan)
    n_mot = int(powertrain["n_motors"]) if powertrain else 0
    if powertrain is not None and np.isfinite(power).sum() >= _MIN_FIT_SAMPLES:
        soc = _soc_from_volt(powertrain["battery"], float(np.nanmedian(volt)),
                             float(np.nanmedian(current)), n_mot)
        vf = v[base] if base.any() else v
        inv = PowerInverter(powertrain["motor"], powertrain["prop"],
                            powertrain["battery"], float(np.nanmin(vf)),
                            float(np.nanmax(vf)), soc=soc)
        for i in range(len(v)):
            if ok_v[i] and np.isfinite(power[i]):
                T_pow[i] = inv.thrust_at(float(v[i]), float(power[i]) / n_mot)
        T_pow *= n_mot
        diag["soc"] = soc
        # binned power-derived thrust curve for the drag-signature plot
        edges = np.arange(math.floor(np.nanmin(vf)),
                          math.ceil(np.nanmax(vf)) + v_step, v_step)
        pc_v, pc_t = [], []
        for k in range(len(edges) - 1):
            b = (v >= edges[k]) & (v < edges[k + 1]) & np.isfinite(T_pow)
            if b.sum() >= 3:
                pc_v.append(0.5 * (edges[k] + edges[k + 1]))
                pc_t.append(float(np.nanmedian(T_pow[b])))
        if pc_v:
            diag["power_thrust_curve"] = (np.array(pc_v), np.array(pc_t))

    def drag_fit(T_tot: np.ndarray, channel: str):
        """Fit CD(CL^2) on one thrust channel -> (revisions, mean dCD, n)."""
        sel = np.isfinite(T_tot) & base
        n = int(sel.sum())
        if n < _MIN_FIT_SAMPLES:
            return [], float("nan"), n
        CD_meas = T_tot[sel] / (q[sel] * S)
        x, CLs = x_ind[sel], CL[sel]
        dCD_mean = float(np.mean(CD_meas - (aero.CD0 + x / aero.eff)))
        spread = float((CLs.max() - CLs.min()) / CLs.mean()) if CLs.mean() else 0.0
        out = []
        b1 = float("nan")
        if spread >= _CL_SPREAD_MIN:
            b0, b1, s0, s1 = _ols_line(x, CD_meas)
        # joint fit only when it lands on physical values; a wild eff usually
        # means the high-CL samples aren't really steady level flight
        plausible = (np.isfinite(b1) and b1 > 0 and b0 > 0
                     and 0.1 <= 1.0 / b1 <= 2.0)
        if plausible:
            out.append(Revision("CD0", aero.CD0, b0, "identified",
                                f"CL spans {CLs.min():.2f}-{CLs.max():.2f} "
                                f"({spread:.0%} spread, {n} samples) -> parasitic "
                                "and induced separable", s0, channel))
            out.append(Revision("eff", aero.eff, 1.0 / b1, "identified",
                                "induced slope 1/(pi*AR*eff), fit jointly with CD0",
                                s1 / (b1 * b1), channel))
        else:
            if spread < _CL_SPREAD_MIN:
                reason = f"CL spread {spread:.0%} < {_CL_SPREAD_MIN:.0%}"
            else:
                reason = (f"joint CD0/eff fit unphysical (CD0={b0:.4f}, "
                          f"eff={1.0 / b1 if b1 else float('nan'):.2f})")
            resid = CD_meas - (aero.CD0 + x / aero.eff)
            out.append(Revision("CD0", aero.CD0, aero.CD0 + float(np.mean(resid)),
                                "combination",
                                f"{reason}: induced held at eff={aero.eff}, whole "
                                f"dCD={float(np.mean(resid)):+.4f} folded into CD0 "
                                f"({n} samples)",
                                float(np.std(resid) / math.sqrt(n)), channel))
        return out, dCD_mean, n

    pow_revs, dCD_pow, n_pow = drag_fit(T_pow, "battery power")
    rpm_revs, dCD_rpm, n_rpm = drag_fit(thrust_rpm,
                                        f"thrust-from-RPM ({real_sig.thrust_src})")
    diag["dCD0_power"], diag["dCD0_rpm"] = dCD_pow, dCD_rpm
    if pow_revs:
        revs += pow_revs
        diag["drag_channel"] = "battery power"
        diag["n_fit"] = n_pow
    elif rpm_revs:
        revs += rpm_revs
        diag["drag_channel"] = "thrust-from-RPM (no power data/model -> fallback)"
        diag["n_fit"] = n_rpm
    diag["eff_identifiable"] = any(r.tag == "eff" and r.kind == "identified"
                                   for r in revs)

    # ---- lift: CL = CL0 + CLa*alpha on real AoA-sensor samples ----
    if real_sig.aoa_src.startswith("AOA"):
        sel = np.isfinite(aoa) & base
        n = int(sel.sum())
        if n >= _MIN_FIT_SAMPLES:
            a_rad = np.radians(aoa[sel])
            span = float(np.degrees(a_rad.max() - a_rad.min()))
            diag["aoa"] = (float(aoa[sel].min()), float(aoa[sel].max()))
            b1 = float("nan")
            if span >= _AOA_SPAN_MIN:
                b0, b1, s0, s1 = _ols_line(a_rad, CL[sel])
            # a fitted lift slope far off any plausible wing value means the
            # AoA estimate and the load-corrected CL don't co-vary as physics
            # demands (EKF-derived AoA is noisy, esp. in turns) -> combination
            if np.isfinite(b1) and 1.5 <= b1 <= 7.0:
                diag["cla_identifiable"] = True
                revs.append(Revision("CL0", aero.CL0, b0, "identified",
                                     f"AoA spans {span:.1f} deg ({n} samples) -> "
                                     "CL0/CLa separable", s0, "AOA + level CL"))
                revs.append(Revision("CLa", aero.CLa, b1, "identified",
                                     "lift slope per rad, fit jointly with CL0",
                                     s1, "AOA + level CL"))
            else:
                if span < _AOA_SPAN_MIN:
                    reason = f"AoA span {span:.1f} deg < {_AOA_SPAN_MIN} deg"
                else:
                    reason = (f"joint CL0/CLa fit unphysical "
                              f"(CLa_fit={b1:.2f}/rad outside 1.5-7)")
                resid = CL[sel] - (aero.CL0 + aero.CLa * a_rad)
                revs.append(Revision("CL0", aero.CL0,
                                     aero.CL0 + float(np.mean(resid)), "combination",
                                     f"{reason}: fixes only CL0+CLa*a, offset "
                                     f"folded into CL0 ({n} samples)",
                                     float(np.std(resid) / math.sqrt(n)),
                                     "AOA + level CL"))
    return revs, diag


def predicted_elec(aero: AeroModel, mass: float, prop_csv: str, motor_xml: str,
                   battery_xml: str, n_motors: int, V: np.ndarray):
    """Predicted (pack current A, pack power W) vs airspeed from the SDF aero coupled
    to the supplied motor/prop/battery -- the powertrain reference to compare measured
    battery draw against. Returns (V, current, power) or None if inputs are missing."""
    if not (motor_xml and battery_xml):
        return None
    prop = load_propeller_csv(prop_csv)
    motor = load_motor(motor_xml)
    battery = load_battery(battery_xml)
    pts = cruise_sweep(aero, mass, V, motor, prop, battery, n_motors)
    Vv = np.array([p.V for p in pts])
    cur = np.array([p.current_per_motor_A * n_motors for p in pts])
    pw = np.array([p.P_elec_total_W for p in pts])
    return Vv, cur, pw


def sdf_predicted_cruise(aero: AeroModel, mass: float, V: np.ndarray):
    """Predicted (alpha_trim deg, thrust_req N) for level cruise from the SDF aero."""
    W = mass * G
    S, rho = aero.area, aero.rho
    alpha = np.full(len(V), np.nan)
    thrust = np.full(len(V), np.nan)
    for i, v in enumerate(V):
        if v <= 0:
            continue
        CL_req = 2.0 * W / (rho * S * v * v)
        a = alpha_trim(aero, CL_req)
        if math.isnan(a):
            continue
        CD = float(CD_of_alpha(aero, np.array([a]))[0])
        alpha[i] = math.degrees(a)
        thrust[i] = 0.5 * rho * v * v * S * CD
    return alpha, thrust


# ---------------------------------------------------------------------------
# Transition dynamics (time-mismatch robust)
# ---------------------------------------------------------------------------
def resample_window(t, y, t_lo, t_hi, dt):
    if t is None or len(t) == 0:
        return None, None
    grid = np.arange(0.0, t_hi - t_lo, dt)
    vals = interp_at(t, y, grid + t_lo)
    return grid, vals


def best_lag(a: np.ndarray, b: np.ndarray, dt: float):
    """Lag (s) and normalized peak of cross-correlation. +lag => a lags b."""
    a = a - np.nanmean(a)
    b = b - np.nanmean(b)
    a = np.nan_to_num(a)
    b = np.nan_to_num(b)
    denom = math.sqrt(np.sum(a * a) * np.sum(b * b))
    if denom == 0:
        return 0.0, 0.0
    corr = np.correlate(a, b, mode="full") / denom
    lags = np.arange(-len(b) + 1, len(a)) * dt
    k = int(np.argmax(corr))
    return float(lags[k]), float(corr[k])


@dataclass
class TransitionMetrics:
    have: bool = False
    lag_s: float = float("nan")
    xcorr_peak: float = float("nan")
    rms_pitch: float = float("nan")
    rms_roll: float = float("nan")
    rms_yaw: float = float("nan")
    t2v_real: float = float("nan")
    t2v_sitl: float = float("nan")
    target_v: float = float("nan")
    note: str = ""


def time_to_airspeed(log: LogData, t_start: float, target_v: float, pre: float):
    t_arsp, v_arsp, _ = airspeed_series(log)
    if t_arsp is None:
        return float("nan")
    m = (t_arsp >= t_start - pre) & (v_arsp >= target_v)
    if not m.any():
        return float("nan")
    return float(t_arsp[m][0] - (t_start - pre))


def transition_metrics(real: LogData, sitl: LogData, pre: float, post: float,
                        dt: float) -> TransitionMetrics:
    tm = TransitionMetrics()
    pr, ps = detect_phases(real), detect_phases(sitl)
    if pr.t_fw is None or ps.t_fw is None:
        tm.note = "missing 'Transition FW done' in real or SITL -> skipped"
        return tm
    if not (real.has("RATE", "P") and sitl.has("RATE", "P")):
        tm.note = "RATE.P missing -> skipped"
        return tm

    gr, pr_rate = resample_window(real.rel("RATE"), real.col("RATE", "P"),
                                  pr.t_fw - pre, pr.t_fw + post, dt)
    gs, ps_rate = resample_window(sitl.rel("RATE"), sitl.col("RATE", "P"),
                                  ps.t_fw - pre, ps.t_fw + post, dt)
    if gr is None or gs is None:
        tm.note = "could not resample RATE.P window"
        return tm

    lag, peak = best_lag(pr_rate, ps_rate, dt)
    tm.lag_s, tm.xcorr_peak = lag, peak

    # Per-axis RMS attitude error after shifting SITL by the recovered lag.
    # Pitch is the headline: it is the tailsitter transition axis that the
    # longitudinal aero drives. Roll/yaw are dominated by heading choice and
    # Euler gimbal-lock near vertical pitch, so they are informational only.
    if real.has("ATT", "Roll", "Pitch", "Yaw") and sitl.has("ATT", "Roll", "Pitch", "Yaw"):
        grid = np.arange(0.0, (pre + post), dt)
        rms = {}
        for axis in ("Roll", "Pitch", "Yaw"):
            r = interp_at(real.rel("ATT"), real.col("ATT", axis), grid + (pr.t_fw - pre))
            s = interp_at(sitl.rel("ATT"), sitl.col("ATT", axis),
                          grid + (ps.t_fw - pre) + lag)
            d = (r - s + 180.0) % 360.0 - 180.0   # wrap to [-180, 180]
            rms[axis] = float(np.sqrt(np.nanmean(d * d)))
        tm.rms_pitch, tm.rms_roll, tm.rms_yaw = rms["Pitch"], rms["Roll"], rms["Yaw"]

    # time-to-target-airspeed (target from params, else cruise default)
    target = real.params.get("ARSPD_FBW_MIN")
    if target is None or target <= 0:
        target = real.params.get("TRIM_ARSPD_CM", 0) / 100.0 or 15.0
    tm.target_v = target
    tm.t2v_real = time_to_airspeed(real, pr.t_fw, target, pre)
    tm.t2v_sitl = time_to_airspeed(sitl, ps.t_fw, target, pre)
    tm.have = True
    return tm


# ---------------------------------------------------------------------------
# Transition ground-track + attitude (position-overshoot focus)
#
# The forward (hover->cruise) and back (cruise->hover) transitions are compared
# separately because the back transition is the one that exposes post-stall drag:
# the aircraft pitches up into deep stall to decelerate, and if SITL under-models
# the high-AoA drag it sails past the hover point -> XY overshoot. We quantify the
# horizontal excursion from the settle point and the attitude track, real vs SITL,
# lag-aligned on pitch rate (the longitudinal transition axis) like the forward
# metrics above.
# ---------------------------------------------------------------------------
@dataclass
class TransitionTrack:
    kind: str                       # "forward" | "back"
    have: bool = False
    lag_s: float = float("nan")
    xcorr_peak: float = float("nan")
    grid: np.ndarray = None         # time relative to the transition event (s)
    # per-log horizontal ground track (m) referenced to the position AT the event,
    # plus horizontal distance r=hypot(dN,dE) and pitch (deg)
    real_n: np.ndarray = None
    real_e: np.ndarray = None
    real_r: np.ndarray = None
    real_pitch: np.ndarray = None
    sitl_n: np.ndarray = None
    sitl_e: np.ndarray = None
    sitl_r: np.ndarray = None
    sitl_pitch: np.ndarray = None
    pos_src: str = ""
    # position-controller horizontal tracking error |actual-target| (m), the XY
    # overshoot of the commanded stop point (None if PSC not logged)
    real_perr: np.ndarray = None
    sitl_perr: np.ndarray = None
    perr_src: str = ""
    overshoot_real: float = float("nan")   # peak tracking error over the window
    overshoot_sitl: float = float("nan")
    overshoot_ratio: float = float("nan")  # SITL / real
    rms_horiz: float = float("nan")        # RMS |r_sitl - r_real| trajectory diff
    rms_pitch: float = float("nan")        # RMS pitch difference (lag-aligned)
    note: str = ""


def _lag_on_pitchrate(real: LogData, sitl: LogData, t_real: float, t_sitl: float,
                      pre: float, post: float, dt: float):
    """Pitch-rate cross-correlation lag (s) and peak between the two event windows."""
    if not (real.has("RATE", "P") and sitl.has("RATE", "P")):
        return 0.0, float("nan")
    _, r = resample_window(real.rel("RATE"), real.col("RATE", "P"),
                           t_real - pre, t_real + post, dt)
    _, s = resample_window(sitl.rel("RATE"), sitl.col("RATE", "P"),
                           t_sitl - pre, t_sitl + post, dt)
    if r is None or s is None:
        return 0.0, float("nan")
    return best_lag(r, s, dt)


def _track_on_grid(log: LogData, t_event: float, grid: np.ndarray, pre: float,
                   shift: float = 0.0):
    """Ground track + pitch for one log on `grid` (time relative to t_event).

    Positions are referenced to the aircraft position AT the event (grid==pre),
    so both logs start their excursion from a common origin regardless of where
    the EKF origin sits. Returns (n, e, r, pitch, src) or None.
    """
    t_pos, N, E, _, src = position_ned_series(log)
    if t_pos is None:
        return None
    base = grid + (t_event - pre) + shift
    n = interp_at(t_pos, N, base)
    e = interp_at(t_pos, E, base)
    i0 = int(np.argmin(np.abs(grid - pre)))      # sample at the event
    n = n - n[i0]
    e = e - e[i0]
    r = np.hypot(n, e)
    pitch = (interp_at(log.rel("ATT"), log.col("ATT", "Pitch"), base)
             if log.has("ATT", "Pitch") else np.full(len(grid), np.nan))
    return n, e, r, pitch, src


def _transition_track(real: LogData, sitl: LogData, kind: str,
                      t_real: float | None, t_sitl: float | None,
                      pre: float, post: float, dt: float) -> TransitionTrack:
    tt = TransitionTrack(kind=kind)
    if t_real is None or t_sitl is None:
        tt.note = f"missing {'Transition VTOL done' if kind == 'back' else 'Transition FW done'} event in real or SITL"
        return tt
    lag, peak = _lag_on_pitchrate(real, sitl, t_real, t_sitl, pre, post, dt)
    tt.lag_s, tt.xcorr_peak = lag, peak
    grid = np.arange(0.0, pre + post, dt)
    tt.grid = grid
    rt = _track_on_grid(real, t_real, grid, pre, shift=0.0)
    st = _track_on_grid(sitl, t_sitl, grid, pre, shift=lag)  # shift SITL onto real
    if rt is None or st is None:
        tt.note = "no position signal (XKF1/POS/AHR2) in real or SITL"
        return tt
    tt.real_n, tt.real_e, tt.real_r, tt.real_pitch, src_r = rt
    tt.sitl_n, tt.sitl_e, tt.sitl_r, tt.sitl_pitch, src_s = st
    tt.pos_src = src_r if src_r == src_s else f"{src_r} vs {src_s}"

    # Trajectory difference (heading-invariant, since both start at the event point)
    dr = tt.sitl_r - tt.real_r
    tt.rms_horiz = float(np.sqrt(np.nanmean(dr * dr)))
    dp = (tt.sitl_pitch - tt.real_pitch + 180.0) % 360.0 - 180.0
    tt.rms_pitch = float(np.sqrt(np.nanmean(dp * dp)))

    # Overshoot = peak position-controller tracking error over the window. This is
    # the bounded XY excursion past the commanded stop point, NOT the cruise travel
    # through the window, so the two logs are compared on the same physical quantity.
    er, perr_r, psrc_r = pos_track_error_series(real)
    es, perr_s, psrc_s = pos_track_error_series(sitl)
    base_r = grid + (t_real - pre)
    base_s = grid + (t_sitl - pre) + lag
    if er is not None:
        tt.real_perr = interp_at(er, perr_r, base_r)
        tt.overshoot_real = float(np.nanmax(tt.real_perr))
    if es is not None:
        tt.sitl_perr = interp_at(es, perr_s, base_s)
        tt.overshoot_sitl = float(np.nanmax(tt.sitl_perr))
    if er is not None and es is not None:
        tt.perr_src = psrc_r if psrc_r == psrc_s else f"{psrc_r} vs {psrc_s}"
        tt.overshoot_ratio = (tt.overshoot_sitl / tt.overshoot_real
                              if tt.overshoot_real else float("nan"))
    elif tt.kind == "back":
        tt.note = "no PSC tracking error (position controller not logged); overshoot from trajectory only"
    tt.have = True
    return tt


def transition_tracks(real: LogData, sitl: LogData, pre: float, post: float,
                      dt: float) -> dict:
    """Forward- and back-transition ground-track/attitude comparison (real vs SITL)."""
    pr, ps = detect_phases(real), detect_phases(sitl)
    return {
        "forward": _transition_track(real, sitl, "forward", pr.t_fw, ps.t_fw,
                                     pre, post, dt),
        "back": _transition_track(real, sitl, "back", pr.t_vtol, ps.t_vtol,
                                  pre, post, dt),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def make_plots(real_sig, sitl_sig, aero, mass, real, sitl, tm,
               outdir, show, pre, post, dt, elec_pred=None,
               hover_real=None, hover_sitl=None, hover_pred=None,
               pow_curve=None):
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)

    # ---- Drag signature: total thrust vs airspeed ----
    fig, ax = plt.subplots(figsize=(8, 6))
    if real_sig is not None and np.isfinite(real_sig.thrust).any():
        ax.plot(real_sig.v, real_sig.thrust, "o-", color="C0", label=f"real ({real_sig.thrust_src})")
    if pow_curve is not None:
        # the trusted drag channel: measured pack power inverted through the
        # powertrain model (what the MODEL UPDATE fit actually uses)
        ax.plot(pow_curve[0], pow_curve[1], "^-", color="C2",
                label="real (from BAT power, inverted powertrain)")
    if sitl_sig is not None and np.isfinite(sitl_sig.thrust).any():
        ax.plot(sitl_sig.v, sitl_sig.thrust, "s-", color="C1", label=f"SITL ({sitl_sig.thrust_src})")
    vmax = max([s.v.max() for s in (real_sig, sitl_sig) if s is not None] + [20.0])
    V = np.linspace(2, vmax * 1.1, 80)
    _, thr_pred = sdf_predicted_cruise(aero, mass, V)
    ax.plot(V, thr_pred, "k--", label="SDF predicted (level, T=D)")
    ax.set_xlabel("airspeed (m/s)")
    ax.set_ylabel("total thrust (N)")
    ax.set_title("Drag signature: thrust required vs airspeed")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "drag_signature.png"), dpi=120)

    # ---- Lift/trim signature: AoA & pitch vs airspeed ----
    fig, (axa, axt) = plt.subplots(1, 2, figsize=(13, 5))
    alpha_pred, _ = sdf_predicted_cruise(aero, mass, V)
    if real_sig is not None:
        axa.plot(real_sig.v, real_sig.aoa, "o-", color="C0", label=f"real ({real_sig.aoa_src})")
        axt.plot(real_sig.v, real_sig.throttle, "o-", color="C0", label="real")
    if sitl_sig is not None:
        axa.plot(sitl_sig.v, sitl_sig.aoa, "s-", color="C1", label=f"SITL ({sitl_sig.aoa_src})")
        axt.plot(sitl_sig.v, sitl_sig.throttle, "s-", color="C1", label="SITL")
    axa.plot(V, alpha_pred, "k--", label="SDF predicted alpha_trim")
    axa.set_xlabel("airspeed (m/s)")
    axa.set_ylabel("AoA / pitch (deg)")
    axa.set_title("Lift/trim signature")
    axa.grid(True)
    axa.legend()
    axt.set_xlabel("airspeed (m/s)")
    axt.set_ylabel("throttle (%)")
    axt.set_title("Throttle vs airspeed")
    axt.grid(True)
    axt.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "lift_trim_signature.png"), dpi=120)

    # ---- Current & power vs airspeed ----
    fig, (axc, axp) = plt.subplots(1, 2, figsize=(13, 5))
    for sig, c, mk, tag in ((real_sig, "C0", "o-", "real"),
                            (sitl_sig, "C1", "s-", "SITL")):
        if sig is None or sig.current is None or not np.isfinite(sig.current).any():
            continue
        axc.plot(sig.v, sig.current, mk, color=c, label=f"{tag} ({sig.elec_src})")
        axp.plot(sig.v, sig.power, mk, color=c, label=f"{tag} ({sig.elec_src})")
    if elec_pred is not None:
        Vp, curp, pwp = elec_pred
        axc.plot(Vp, curp, "k--", label="powertrain model")
        axp.plot(Vp, pwp, "k--", label="powertrain model")
    # hover points at V = 0 (median draw in the hover phases)
    for hov, c, mk, tag in ((hover_real, "C0", "o", "real hover"),
                            (hover_sitl, "C1", "s", "SITL hover")):
        if hov is None:
            continue
        axc.plot(0.0, hov.current, mk, color=c, mfc="none", ms=10,
                 label=f"{tag} ({hov.elec_src})")
        axp.plot(0.0, hov.power, mk, color=c, mfc="none", ms=10,
                 label=f"{tag} ({hov.elec_src})")
    if hover_pred is not None:
        Ih, Ph, th = hover_pred
        axc.plot(0.0, Ih, "k*", ms=12, label=f"model hover (thr {th * 100:.0f}%)")
        axp.plot(0.0, Ph, "k*", ms=12, label=f"model hover (thr {th * 100:.0f}%)")
    axc.set_xlabel("airspeed (m/s)")
    axc.set_ylabel("pack current (A)")
    axc.set_title("Current vs airspeed")
    axc.grid(True)
    axc.legend()
    axp.set_xlabel("airspeed (m/s)")
    axp.set_ylabel("electrical power (W)")
    axp.set_title("Power consumption vs airspeed")
    axp.grid(True)
    axp.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "power_current.png"), dpi=120)

    # ---- Transition overlay (lag-aligned) ----
    if tm.have:
        pr, ps = detect_phases(real), detect_phases(sitl)
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        grid = np.arange(0.0, pre + post, dt)

        def plot_pair(ax, mtype, colmap, label):
            for log, ps_t, sh, style, c, tag in (
                (real, pr.t_fw, 0.0, "-", "C0", "real"),
                (sitl, ps.t_fw, tm.lag_s, "--", "C1", "SITL"),
            ):
                col = colmap.get(log.name)
                if col is None or not log.has(mtype, col):
                    continue
                y = interp_at(log.rel(mtype), log.col(mtype, col),
                              grid + (ps_t - pre) + sh)
                ax.plot(grid - pre, y, style, color=c, label=f"{tag} {label}")
            ax.axvline(0.0, color="k", lw=0.8, ls=":")
            ax.set_ylabel(label)
            ax.grid(True)
            ax.legend(fontsize=8)

        both = {real.name: "P", sitl.name: "P"}
        plot_pair(axes[0], "RATE", both, "pitch rate (deg/s)")
        plot_pair(axes[1], "ATT", {real.name: "Pitch", sitl.name: "Pitch"}, "pitch (deg)")
        # airspeed on axis 3 (per-log airspeed source)
        for log, ps_t, sh, style, c, tag in (
            (real, pr.t_fw, 0.0, "-", "C0", "real"),
            (sitl, ps.t_fw, tm.lag_s, "--", "C1", "SITL"),
        ):
            t_a, v_a, _ = airspeed_series(log)
            if t_a is not None:
                y = interp_at(t_a, v_a, grid + (ps_t - pre) + sh)
                axes[2].plot(grid - pre, y, style, color=c, label=f"{tag} airspeed")
        if np.isfinite(tm.target_v):
            axes[2].axhline(tm.target_v, color="g", lw=0.8, ls=":", label="target V")
        axes[2].axvline(0.0, color="k", lw=0.8, ls=":")
        axes[2].set_ylabel("airspeed (m/s)")
        axes[2].set_xlabel("time relative to 'Transition FW done' (s)")
        axes[2].grid(True)
        axes[2].legend(fontsize=8)
        fig.suptitle(f"Transition overlay (SITL shifted {tm.lag_s:+.2f}s by pitch-rate xcorr, "
                     f"peak={tm.xcorr_peak:.2f})")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "transition_overlay.png"), dpi=120)

    if show:
        plt.show()
    plt.close("all")


def make_track_plots(tracks: dict, outdir, show, pre):
    """Forward/back transition: XY ground track, horizontal excursion, pitch.

    One column per transition; SITL is lag-aligned onto real. The back-transition
    column is the post-stall-drag diagnostic -- a SITL excursion curve that runs
    above real means SITL coasts further before settling (drag under-modelled).
    """
    import matplotlib.pyplot as plt

    have = [k for k in ("forward", "back") if tracks[k].have]
    if not have:
        return
    os.makedirs(outdir, exist_ok=True)
    ncol = len(have)
    fig, axes = plt.subplots(3, ncol, figsize=(6.5 * ncol, 12), squeeze=False)
    for ci, key in enumerate(have):
        tt = tracks[key]
        g = tt.grid - pre  # time relative to the transition event
        title = "Forward (hover->cruise)" if key == "forward" else "Back (cruise->hover)"

        # row 0: XY ground track (East vs North), origin = position at the event
        ax = axes[0][ci]
        ax.plot(tt.real_e, tt.real_n, "-", color="C0", label="real")
        ax.plot(tt.sitl_e, tt.sitl_n, "--", color="C1", label="SITL")
        ax.plot(0, 0, "k*", ms=12, label="event point")
        ax.set_xlabel("East (m)")
        ax.set_ylabel("North (m)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"{title}\nground track ({tt.pos_src})")
        ax.grid(True)
        ax.legend(fontsize=8)

        # row 1: XY overshoot of the commanded stop point (PSC tracking error),
        # falling back to the real-vs-SITL trajectory difference if PSC is absent
        ax = axes[1][ci]
        if tt.real_perr is not None or tt.sitl_perr is not None:
            if tt.real_perr is not None:
                ax.plot(g, tt.real_perr, "-", color="C0", label="real")
            if tt.sitl_perr is not None:
                ax.plot(g, tt.sitl_perr, "--", color="C1", label="SITL")
            ax.set_ylabel("pos-ctrl tracking error |actual-target| (m)")
            ax.set_title(f"XY overshoot  peak real={tt.overshoot_real:.1f} m  "
                         f"SITL={tt.overshoot_sitl:.1f} m  (x{tt.overshoot_ratio:.2f})")
        else:
            ax.plot(g, np.abs(tt.sitl_r - tt.real_r), "-", color="C3",
                    label="|SITL-real| track")
            ax.set_ylabel("real-vs-SITL track diff (m)")
            ax.set_title(f"trajectory difference  RMS={tt.rms_horiz:.1f} m  "
                         "(no PSC log -> no target overshoot)")
        ax.axvline(0.0, color="k", lw=0.8, ls=":")
        ax.set_xlabel("time relative to event (s)")
        ax.grid(True)
        ax.legend(fontsize=8)

        # row 2: pitch vs time (lag-aligned)
        ax = axes[2][ci]
        ax.plot(g, tt.real_pitch, "-", color="C0", label="real")
        ax.plot(g, tt.sitl_pitch, "--", color="C1", label="SITL")
        ax.axvline(0.0, color="k", lw=0.8, ls=":")
        ax.set_xlabel("time relative to event (s)")
        ax.set_ylabel("pitch (deg)")
        ax.set_title(f"pitch  RMS dpitch={tt.rms_pitch:.1f} deg  "
                     f"(SITL shifted {tt.lag_s:+.2f}s, xcorr {tt.xcorr_peak:.2f})")
        ax.grid(True)
        ax.legend(fontsize=8)

    fig.suptitle("Transition ground-track & attitude (SITL lag-aligned onto real)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "transition_tracks.png"), dpi=120)
    if show:
        plt.show()
    plt.close("all")


# ---------------------------------------------------------------------------
# Report (metrics + tuning map)
# ---------------------------------------------------------------------------
def common_bins(a: CruiseSig, b: CruiseSig):
    """Match real/SITL signature bins by airspeed center."""
    out = []
    for i, va in enumerate(a.v):
        j = int(np.argmin(np.abs(b.v - va)))
        if abs(b.v[j] - va) < 1e-6:
            out.append((va, i, j))
    return out


def write_report(real_sig, sitl_sig, tm, real, sitl, aero, mass, outdir,
                 elec_pred=None, tracks=None,
                 hover_real=None, hover_sitl=None, hover_pred=None,
                 revs=None, diag=None):
    """Write report.txt and return the metrics dict (also saved as metrics.json
    by the caller). revs/diag come from recommend_revisions()."""
    revs, diag = revs or [], diag or {}
    lines = []
    P = lines.append
    P("=" * 70)
    P("  real-vs-SITL comparison  (real = ground truth)")
    P("=" * 70)
    P(f"  real: {real.path}")
    P(f"  SITL: {sitl.path}")
    P("")

    thrust_bias = pitch_bias = aoa_bias = float("nan")
    thrust_slope_ratio = float("nan")

    if real_sig is not None and sitl_sig is not None:
        pairs = common_bins(real_sig, sitl_sig)
        P("--- CRUISE operating-point signature (SITL - real, matched airspeed bins) ---")
        if real_sig.note:
            P(f"  note (real): {real_sig.note}")
        if sitl_sig.note:
            P(f"  note (SITL): {sitl_sig.note}")
        if not pairs:
            P("  no overlapping airspeed bins between the two logs:")
            P(f"    real bins at {np.array2string(real_sig.v, precision=0)} m/s, "
              f"SITL at {np.array2string(sitl_sig.v, precision=0)} m/s")
            P("    -> SITL trims/cruises at a different airspeed than the real "
              "aircraft; fix that first (TRIM_ARSPD_CM / drag level).")
        else:
            P(f"  {'V (m/s)':>8} {'dThrust(N)':>11} {'dThrottle%':>11} "
              f"{'dAoA(deg)':>10} {'dPitch(deg)':>11}")
            dthr, daoa, dpit, vs = [], [], [], []
            for v, i, j in pairs:
                dt_ = sitl_sig.thrust[j] - real_sig.thrust[i]
                dq_ = sitl_sig.throttle[j] - real_sig.throttle[i]
                da_ = sitl_sig.aoa[j] - real_sig.aoa[i]
                dp_ = sitl_sig.pitch[j] - real_sig.pitch[i]
                P(f"  {v:8.1f} {dt_:11.2f} {dq_:11.1f} {da_:10.2f} {dp_:11.2f}")
                vs.append(v); dthr.append(dt_); daoa.append(da_); dpit.append(dp_)
            thrust_bias = float(np.nanmean(dthr))
            aoa_bias = float(np.nanmean(daoa))
            pitch_bias = float(np.nanmean(dpit))
            # slope ratio of thrust(V): SITL vs real
            if len(vs) >= 2:
                sr = np.polyfit(real_sig.v, np.nan_to_num(real_sig.thrust), 1)[0]
                ss = np.polyfit(sitl_sig.v, np.nan_to_num(sitl_sig.thrust), 1)[0]
                thrust_slope_ratio = ss / sr if sr != 0 else float("nan")
            P("")
            P(f"  mean thrust bias  : {thrust_bias:+.2f} N   (SITL - real)")
            P(f"  mean AoA bias     : {aoa_bias:+.2f} deg")
            P(f"  mean pitch bias   : {pitch_bias:+.2f} deg")
            if np.isfinite(thrust_slope_ratio):
                P(f"  thrust-slope ratio: {thrust_slope_ratio:.2f}  (SITL/real, vs airspeed)")
            else:
                P("  thrust-slope ratio: n/a  (need >=2 overlapping airspeed bins)")
    else:
        P("--- CRUISE: insufficient data in one or both logs ---")
    P("")

    # ---- electrical: current & power ----
    cur_bias = pow_bias = pow_bias_rel = float("nan")
    elec_ref = None
    P("--- ELECTRICAL: current & power vs airspeed (cruise) ---")
    has_re = (real_sig is not None and real_sig.current is not None
              and np.isfinite(real_sig.current).any())
    has_se = (sitl_sig is not None and sitl_sig.current is not None
              and np.isfinite(sitl_sig.current).any())
    if sitl_sig is not None and not has_se:
        P("  SITL: no battery telemetry -> compared against the powertrain model")
    if not has_re:
        P("  real: no battery current logged -> nothing to compare")
    elif not has_se and elec_pred is None:
        P("  no SITL battery telemetry and no --motor/--battery model -> skipped")
    else:
        ref = elec_ref = "SITL" if has_se else "model"
        P(f"  {'V(m/s)':>7} {'I_real(A)':>10} {f'I_{ref}(A)':>10} "
          f"{'P_real(W)':>10} {f'P_{ref}(W)':>10}")
        dI, dP = [], []
        for i, v in enumerate(real_sig.v):
            if has_se:
                Ic = float(np.interp(v, sitl_sig.v, sitl_sig.current))
                Pc = float(np.interp(v, sitl_sig.v, sitl_sig.power))
            else:
                Vp, curp, pwp = elec_pred
                Ic = float(np.interp(v, Vp, curp))
                Pc = float(np.interp(v, Vp, pwp))
            P(f"  {v:7.1f} {real_sig.current[i]:10.2f} {Ic:10.2f} "
              f"{real_sig.power[i]:10.1f} {Pc:10.1f}")
            dI.append(Ic - real_sig.current[i])
            dP.append(Pc - real_sig.power[i])
        cur_bias, pow_bias = float(np.nanmean(dI)), float(np.nanmean(dP))
        p_real_mean = float(np.nanmean(real_sig.power))
        pow_bias_rel = pow_bias / p_real_mean if p_real_mean else float("nan")
        P("")
        P(f"  mean current bias : {cur_bias:+.2f} A   ({ref} - real)")
        P(f"  mean power bias   : {pow_bias:+.1f} W   ({ref} - real, "
          f"{pow_bias_rel:+.1%} of real)")
        if np.isfinite(pow_bias):
            if pow_bias < 0:
                P("  -> model/SITL draws LESS than real: real has more drag (see CD0) "
                  "and/or the motor-prop-battery efficiency in the model is optimistic.")
            else:
                P("  -> model/SITL draws MORE than real: model drag/prop too heavy, "
                  "or the powertrain efficiency is pessimistic.")
        P("  NOTE: current/power reflect drag AND the motor/prop/battery model together,"
          " not the aero alone.")
    P("")

    # ---- electrical: hover ----
    P("--- ELECTRICAL: hover current draw (V ~ 0, median over hover phases) ---")
    if hover_real is None and hover_sitl is None and hover_pred is None:
        P("  no hover data in either log and no powertrain model -> skipped")
    else:
        rows = []
        for tag, hov in (("real", hover_real), ("SITL", hover_sitl)):
            if hov is None:
                P(f"  {tag:5s}: n/a (no usable hover samples)")
            else:
                rows.append((tag, hov.current, hov.power))
                P(f"  {tag:5s}: {hov.current:6.1f} A  {hov.power:7.0f} W  "
                  f"({hov.elec_src}, {hov.n_samples} samples"
                  f"{', ' + hov.note if hov.note else ''})")
        if hover_pred is not None:
            Ih, Ph, th = hover_pred
            rows.append(("model", Ih, Ph))
            P(f"  model: {Ih:6.1f} A  {Ph:7.0f} W  (solve_hover, throttle {th * 100:.0f}%)")
        if hover_real is not None:
            for tag, I, Pw in rows:
                if tag == "real":
                    continue
                dI = I - hover_real.current
                rel = dI / hover_real.current if hover_real.current else float("nan")
                P(f"  {tag} - real: {dI:+.1f} A ({rel:+.0%}), "
                  f"{Pw - hover_real.power:+.0f} W")
            P("  NOTE: hover draw probes static prop thrust + motor model + mass only;"
              " the wing aero plays no part at V=0.")
    P("")

    P("--- TRANSITION dynamics (time-mismatch robust) ---")
    if tm.have:
        P(f"  pitch-rate xcorr peak : {tm.xcorr_peak:.3f}  (1.0 = identical shape)")
        P(f"  recovered lag         : {tm.lag_s:+.2f} s  (+ => real lags SITL)")
        P(f"  RMS pitch error       : {tm.rms_pitch:.2f} deg  (transition axis, after lag align)")
        P(f"    roll {tm.rms_roll:.1f} / yaw {tm.rms_yaw:.1f} deg  (heading + gimbal-lock near vertical, informational)")
        P(f"  time-to-{tm.target_v:.1f}m/s real : {tm.t2v_real:.2f} s")
        P(f"  time-to-{tm.target_v:.1f}m/s SITL : {tm.t2v_sitl:.2f} s")
        if np.isfinite(tm.t2v_real) and np.isfinite(tm.t2v_sitl):
            P(f"  time-to-target delta  : {tm.t2v_sitl - tm.t2v_real:+.2f} s  (SITL - real)")
    else:
        P(f"  skipped: {tm.note}")
    P("")

    # ---- transition ground-track / attitude (overshoot) ----
    overshoot_flag = None   # (kind, ratio) of a back-transition over-coast, for the tuning map
    P("--- TRANSITION ground-track & attitude (real vs SITL, lag-aligned) ---")
    if not tracks:
        P("  skipped: no track comparison computed")
    else:
        any_done = False
        for key, label in (("forward", "FORWARD (hover->cruise)"),
                           ("back", "BACK (cruise->hover)")):
            tt = tracks.get(key)
            if tt is None or not tt.have:
                P(f"  {label}: skipped ({tt.note if tt else 'n/a'})")
                continue
            any_done = True
            P(f"  {label}   [pos {tt.pos_src}]")
            if np.isfinite(tt.overshoot_real) or np.isfinite(tt.overshoot_sitl):
                P(f"    XY overshoot (peak)  : real {tt.overshoot_real:6.1f} m   "
                  f"SITL {tt.overshoot_sitl:6.1f} m   (SITL/real x{tt.overshoot_ratio:.2f})   "
                  f"[{tt.perr_src}]")
            else:
                P("    XY overshoot (peak)  : n/a (no PSC position-controller log)")
            P(f"    RMS real-vs-SITL track: {tt.rms_horiz:6.1f} m")
            P(f"    RMS pitch diff       : {tt.rms_pitch:6.1f} deg   "
              f"(SITL shifted {tt.lag_s:+.2f}s, pitch-rate xcorr {tt.xcorr_peak:.2f})")
            if key == "back" and np.isfinite(tt.overshoot_ratio):
                overshoot_flag = (key, tt.overshoot_ratio)
        if any_done:
            P("  (XY overshoot = peak position-controller error |actual-target|, i.e. how far")
            P("   the vehicle sails past the commanded stop point. On the back transition a")
            P("   SITL value above real means SITL coasts further -> post-stall drag too low.)")
    P("")

    # ---- model update: the numeric, primary output ----
    P("--- MODEL UPDATE (identifiability-aware, per-sample regression) ---")
    P(f"  basis: quasi-steady cruise, lift = n*W (n=1/cos(roll), banked samples "
      f"kept to {_FIT_ROLL_MAX:.0f} deg), mass={mass:.2f} kg (--mass overrides)")
    if diag.get("n_fit", 0) == 0 and not revs:
        P("  (insufficient real cruise data to constrain any coefficient)")
    else:
        P(f"  drag channel: {diag.get('drag_channel') or 'n/a'}")
        V = diag.get("V", (float("nan"),) * 2)
        CL = diag.get("CL", (float("nan"),) * 2)
        aoaspan = diag.get("aoa")
        aoa_str = (f"AoA {aoaspan[0]:.1f}-{aoaspan[1]:.1f} deg"
                   if aoaspan else "no AoA sensor data")
        P(f"  data span: {diag.get('n_fit', 0)} fit samples, V {V[0]:.0f}-{V[1]:.0f} m/s, "
          f"CL {CL[0]:.2f}-{CL[1]:.2f} (spread {diag.get('cl_spread', 0):.0%}), {aoa_str}")
        if "soc" in diag:
            P(f"  powertrain inversion at SoC {diag['soc']:.0%} (from measured pack voltage)")

        ident = [r for r in revs if r.kind == "identified"]
        combo = [r for r in revs if r.kind == "combination"]
        if ident:
            P("  data-driven (identified):")
        for r in ident:
            s = f"  sigma~{r.sigma:.4f}" if np.isfinite(r.sigma) else ""
            P(f"    <{r.tag}>  {r.old:+.6f} -> {r.new:+.6f}  "
              f"(d {r.new - r.old:+.6f}{s})  [{r.channel}]")
            P(f"        {r.note}")
        if combo:
            P("  combination only (offset put on leading term, partner held):")
        for r in combo:
            s = f"  sigma~{r.sigma:.4f}" if np.isfinite(r.sigma) else ""
            P(f"    <{r.tag}>  {r.old:+.6f} -> {r.new:+.6f}  "
              f"(d {r.new - r.old:+.6f}{s})  [{r.channel}]")
            P(f"        {r.note}")

        # power-vs-RPM cross-check: same dCD estimated on both channels
        dp = diag.get("dCD0_power", float("nan"))
        dr = diag.get("dCD0_rpm", float("nan"))
        if np.isfinite(dp) and np.isfinite(dr):
            P(f"  cross-check: mean dCD {dp:+.4f} (battery power)  vs  "
              f"{dr:+.4f} (thrust-from-RPM)")
            if abs(dr - dp) <= max(0.25 * abs(dp), 0.005):
                P("        channels agree -> the power-based numbers are corroborated.")
            else:
                P("        channels DISAGREE -> trust POWER (RPM-thrust: one ESC "
                  "instance, spikes); the RPM figure is reference only.")

        # Held at the Flow5/VLM prior: coupled partners the span can't split, plus
        # everything steady cruise can't excite (rates ~ 0, fixed AoA, pre-stall).
        held = []
        if not diag.get("eff_identifiable"):
            held.append(("eff", "induced drag -- fly an AIRSPEED SWEEP (range of CL)"))
        if not diag.get("cla_identifiable"):
            held.append(("CLa", "lift slope -- fly an AoA SWEEP / airspeed sweep"))
        held += [
            ("Cem0, Cema", "pitch trim/stiffness -- needs moment balance / pitch input"),
            ("Cmq (pitch damping)", "needs PITCH DOUBLETS"),
            ("Clp, Clr, Clb", "roll derivatives -- needs ROLL DOUBLETS"),
            ("Cnb, Cnp, Cnr, CYb", "yaw/sideforce -- needs YAW DOUBLETS / sideslip"),
            ("alpha_stall, CLa_stall, CDa_stall", "post-stall Viterna -- needs HIGH-AoA/transition data"),
        ]
        P("  held at Flow5 prior (not identifiable from this flight):")
        for tag, why in held:
            P(f"    {tag:34s} {why}")
        P("  -> to unlock the held set, fly: airspeed sweep + AoA sweep + roll/pitch/yaw doublets.")
        P("  apply with --write-sdf, re-run SITL, re-compare; the ledger tracks the score.")
    P("")

    # ---- qualitative flags: gaps the cruise regression cannot fit ----
    P("--- QUALITATIVE FLAGS (transition / trim -- outside the fit's scope) ---")
    flags = []
    if np.isfinite(pitch_bias) and abs(pitch_bias) > 1.0 and (
            not np.isfinite(aoa_bias) or abs(pitch_bias - aoa_bias) > 1.0):
        flags.append(
            f"near-constant pitch offset ({pitch_bias:+.2f} deg) not explained by AoA: "
            "zero-AoA pitching moment / CG -> adjust Cem0 (and CG vs <cp>).")
    if tm.have:
        if np.isfinite(tm.xcorr_peak) and tm.xcorr_peak < 0.6:
            flags.append(
                f"low pitch-rate xcorr ({tm.xcorr_peak:.2f}): transition pitch dynamics differ "
                "-> check post-stall block (alpha_stall, CLa_stall, CDa_stall) and Cema.")
        if np.isfinite(tm.rms_pitch) and tm.rms_pitch > 5.0:
            flags.append(
                f"high RMS pitch error ({tm.rms_pitch:.1f} deg) in transition "
                "-> high-AoA longitudinal aero (Viterna post-stall: alpha_stall, CLa_stall, "
                "CDa_stall) and pitch-damping suspect.")
        if (np.isfinite(tm.t2v_real) and np.isfinite(tm.t2v_sitl)
                and abs(tm.t2v_sitl - tm.t2v_real) > 1.0):
            d = tm.t2v_sitl - tm.t2v_real
            flags.append(
                f"SITL reaches target airspeed {abs(d):.1f} s "
                f"{'SLOWER' if d > 0 else 'FASTER'} than real: check overall drag level "
                "(CD0 above), prop CSV and SDF mass vs real.")
    if overshoot_flag is not None:
        _, ratio = overshoot_flag
        if ratio > 1.15:
            flags.append(
                f"SITL over-coasts the back transition (peak excursion x{ratio:.2f} vs real): "
                "the aircraft pitches into deep stall to decelerate, so this is the post-stall "
                "DRAG -> RAISE CDa_stall (and check alpha_stall earlier / CLa_stall) so SITL "
                "sheds speed and stops at the hover point instead of overshooting in XY.")
        elif ratio < 0.85:
            flags.append(
                f"SITL under-coasts the back transition (peak excursion x{ratio:.2f} vs real): "
                "post-stall drag too high -> LOWER CDa_stall.")
    if not flags:
        flags.append("none -- transition/trim metrics within thresholds.")
    for s in flags:
        P(f"  * {s}")
    P("")

    # ---- assemble metrics + composite score ----
    t2v_delta = (tm.t2v_sitl - tm.t2v_real
                 if np.isfinite(tm.t2v_real) and np.isfinite(tm.t2v_sitl)
                 else float("nan"))
    trk = {}
    for key in ("forward", "back"):
        tt = tracks.get(key) if tracks else None
        if tt is None or not tt.have:
            trk[key] = {"have": False}
        else:
            trk[key] = {"have": True,
                        "overshoot_real_m": tt.overshoot_real,
                        "overshoot_sitl_m": tt.overshoot_sitl,
                        "overshoot_ratio": tt.overshoot_ratio,
                        "rms_horiz_m": tt.rms_horiz,
                        "rms_pitch_deg": tt.rms_pitch,
                        "lag_s": tt.lag_s, "xcorr_peak": tt.xcorr_peak}
    metrics = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "real": real.path, "sitl": sitl.path, "mass_kg": mass,
        "aero": {k: getattr(aero, k) for k in
                 ("area", "AR", "eff", "CL0", "CLa", "CD0", "Cem0", "Cema",
                  "alpha_stall", "CLa_stall", "CDa_stall")},
        "cruise": {"thrust_bias_N": thrust_bias, "aoa_bias_deg": aoa_bias,
                   "pitch_bias_deg": pitch_bias,
                   "thrust_slope_ratio": thrust_slope_ratio},
        "electrical": {"ref": elec_ref, "current_bias_A": cur_bias,
                       "power_bias_W": pow_bias, "power_bias_rel": pow_bias_rel},
        "hover": {"real_A": hover_real.current if hover_real else None,
                  "real_W": hover_real.power if hover_real else None,
                  "sitl_A": hover_sitl.current if hover_sitl else None,
                  "sitl_W": hover_sitl.power if hover_sitl else None,
                  "model_A": hover_pred[0] if hover_pred else None,
                  "model_W": hover_pred[1] if hover_pred else None},
        "transition": {"have": tm.have, "xcorr_peak": tm.xcorr_peak,
                       "lag_s": tm.lag_s, "rms_pitch_deg": tm.rms_pitch,
                       "target_v_ms": tm.target_v, "t2v_real_s": tm.t2v_real,
                       "t2v_sitl_s": tm.t2v_sitl, "t2v_delta_s": t2v_delta},
        "tracks": trk,
        "fit": {"diag": {k: v for k, v in diag.items()
                         if k != "power_thrust_curve"},
                "revisions": [{"tag": r.tag, "old": r.old, "new": r.new,
                               "kind": r.kind, "sigma": r.sigma,
                               "channel": r.channel, "note": r.note}
                              for r in revs]},
    }
    score, parts = composite_score(metrics)
    metrics["score"] = score
    metrics["score_parts"] = parts
    P("--- FIDELITY SCORE (lower = SITL closer to real; same scales every run) ---")
    if np.isfinite(score):
        P(f"  score = {score:.3f}   "
          f"({', '.join(f'{k}={p:.2f}' for k, p in sorted(parts.items()))})")
        P("  channel scales: power 10% | cruise pitch 2 deg | transition pitch RMS")
        P("  10 deg | time-to-airspeed 2 s | back overshoot x1.25 -> each = 1.0")
    else:
        P("  n/a (no comparable channel produced a finite gap)")

    report = "\n".join(lines)
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "report.txt"), "w") as f:
        f.write(report + "\n")
    print("\n" + report)
    print(f"\n[written] {os.path.join(outdir, 'report.txt')}")
    return metrics


# ---------------------------------------------------------------------------
# Machine-readable outputs: composite score, metrics.json, run ledger
# ---------------------------------------------------------------------------
def composite_score(metrics: dict):
    """Single lower-is-better fidelity number: RMS of normalized channel gaps.

    Each channel is divided by the gap that should count as "1.0 off":
    power 10% of real draw, cruise pitch bias 2 deg, transition pitch RMS
    10 deg, time-to-airspeed 2 s, back-transition overshoot ratio x1.25.
    Only channels with finite data participate, so compare scores between runs
    with the same channels available (the parts are recorded alongside).
    """
    parts = {}

    def add(name, val, scale):
        if val is not None and np.isfinite(val):
            parts[name] = abs(val) / scale

    add("power", metrics["electrical"].get("power_bias_rel"), 0.10)
    add("cruise_pitch", metrics["cruise"].get("pitch_bias_deg"), 2.0)
    add("trans_pitch_rms", metrics["transition"].get("rms_pitch_deg"), 10.0)
    add("t2v", metrics["transition"].get("t2v_delta_s"), 2.0)
    ratio = metrics["tracks"].get("back", {}).get("overshoot_ratio")
    if ratio is not None and np.isfinite(ratio) and ratio > 0:
        add("back_overshoot", math.log(ratio), math.log(1.25))
    if not parts:
        return float("nan"), parts
    return float(np.sqrt(np.mean([p * p for p in parts.values()]))), parts


def _json_sanitize(obj):
    """numpy scalars/arrays -> python; non-finite floats -> None (JSON-safe)."""
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_json_sanitize(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def save_metrics(metrics: dict, outdir: str) -> str:
    path = os.path.join(outdir, "metrics.json")
    with open(path, "w") as f:
        json.dump(_json_sanitize(metrics), f, indent=2)
    return path


# Flat per-run history: one row per compare run, so successive SDF edits can be
# judged by the score trend instead of re-reading reports.
_LEDGER_COLS = [
    "timestamp", "score", "real", "sitl", "sdf", "mass_kg",
    "CD0", "CL0", "CLa", "eff", "alpha_stall", "CLa_stall", "CDa_stall",
    "Cem0", "Cema",
    "power_bias_W", "power_bias_rel", "thrust_bias_N", "aoa_bias_deg",
    "pitch_bias_deg", "trans_xcorr", "trans_rms_pitch_deg", "t2v_delta_s",
    "overshoot_ratio_fwd", "overshoot_ratio_back",
    "fit_CD0", "fit_eff", "fit_CL0", "fit_CLa",
]


def append_ledger(path: str, metrics: dict, sdf_path: str) -> None:
    fit = {r["tag"]: r["new"] for r in metrics["fit"]["revisions"]}
    row = {
        "timestamp": metrics["timestamp"], "score": metrics.get("score"),
        "real": metrics["real"], "sitl": metrics["sitl"], "sdf": sdf_path,
        "mass_kg": metrics["mass_kg"],
        **{k: metrics["aero"].get(k) for k in
           ("CD0", "CL0", "CLa", "eff", "alpha_stall", "CLa_stall",
            "CDa_stall", "Cem0", "Cema")},
        "power_bias_W": metrics["electrical"].get("power_bias_W"),
        "power_bias_rel": metrics["electrical"].get("power_bias_rel"),
        "thrust_bias_N": metrics["cruise"].get("thrust_bias_N"),
        "aoa_bias_deg": metrics["cruise"].get("aoa_bias_deg"),
        "pitch_bias_deg": metrics["cruise"].get("pitch_bias_deg"),
        "trans_xcorr": metrics["transition"].get("xcorr_peak"),
        "trans_rms_pitch_deg": metrics["transition"].get("rms_pitch_deg"),
        "t2v_delta_s": metrics["transition"].get("t2v_delta_s"),
        "overshoot_ratio_fwd": metrics["tracks"].get("forward", {}).get("overshoot_ratio"),
        "overshoot_ratio_back": metrics["tracks"].get("back", {}).get("overshoot_ratio"),
        "fit_CD0": fit.get("CD0"), "fit_eff": fit.get("eff"),
        "fit_CL0": fit.get("CL0"), "fit_CLa": fit.get("CLa"),
    }
    row = {k: ("" if v is None or (isinstance(v, float) and not math.isfinite(v))
               else v) for k, v in _json_sanitize(row).items()}
    new = not os.path.exists(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_LEDGER_COLS)
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"[ledger ] appended to {path}")


# ---------------------------------------------------------------------------
# Apply revisions back into the SDF (--write-sdf)
# ---------------------------------------------------------------------------
def write_revised_sdf(sdf_path: str, revs: list, out_path: str,
                      source_note: str) -> list[str]:
    """Copy the SDF with identified/combination revisions applied to the
    AdvancedLiftDrag block, plus a provenance comment. Pure text substitution
    so formatting and comments survive. Returns the changed-tag summaries.
    """
    with open(sdf_path) as f:
        text = f.read()
    m = re.search(r"<plugin[^>]*advanced[-_]?lift[-_]?drag[^>]*>.*?</plugin>",
                  text, re.S | re.I)
    if not m:
        print(f"[sdf    ] no AdvancedLiftDrag plugin block in {sdf_path} -> skipped")
        return []
    block = m.group(0)
    new_block = block
    changed = []
    for r in revs:
        if r.kind == "held" or not np.isfinite(r.new):
            continue
        # tags are case-sensitive on purpose: <CL0> is lift, <Cl0> roll moment
        pat = re.compile(r"(<{0}>)\s*[-+0-9.eE]+\s*(</{0}>)".format(re.escape(r.tag)))
        if not pat.search(new_block):
            print(f"[sdf    ] tag <{r.tag}> not found in plugin block -> skipped")
            continue
        new_block = pat.sub(
            lambda mm: f"{mm.group(1)}{r.new: .6f}{mm.group(2)}", new_block, count=1)
        changed.append(f"{r.tag} {r.old:.6f}->{r.new:.6f} ({r.kind}, {r.channel})")
    if not changed:
        print("[sdf    ] no applicable revisions -> no SDF written")
        return []
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    comment = ("<!-- compare_logs.py model update {0}\n"
               "         {1}\n"
               "         from {2} -->\n    ".format(
                   stamp, "\n         ".join(changed), source_note))
    text = text[:m.start()] + comment + new_block + text[m.end():]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(text)
    print(f"[sdf    ] wrote revised SDF -> {out_path}")
    for c in changed:
        print(f"          {c}")
    return changed


# ---------------------------------------------------------------------------
# Input resolution: --sitl latest, config-file defaults
# ---------------------------------------------------------------------------
def resolve_latest_sitl(spec: str, logs_dir: str) -> str:
    """'latest' or 'latest-N' -> the (N-back) newest .BIN in the SITL log dir.

    Prefers LASTLOG.TXT (ArduPilot writes the current log number there);
    falls back to the newest *.BIN by mtime.
    """
    m = re.fullmatch(r"latest(?:-(\d+))?", spec)
    if not m:
        return spec
    back = int(m.group(1) or 0)
    d = os.path.expanduser(logs_dir)
    lastlog = os.path.join(d, "LASTLOG.TXT")
    if os.path.exists(lastlog):
        try:
            num = int(open(lastlog).read().strip()) - back
            cand = os.path.join(d, f"{num:08d}.BIN")
            if os.path.exists(cand):
                return cand
        except ValueError:
            pass
    bins = sorted((p for p in os.listdir(d) if p.upper().endswith(".BIN")
                   and p[:8].isdigit()),
                  key=lambda p: os.path.getmtime(os.path.join(d, p)))
    if len(bins) <= back:
        raise SystemExit(f"--sitl {spec}: no matching .BIN under {d}")
    return os.path.join(d, bins[-1 - back])


# config/compare.ini supplies defaults for these (CLI always wins); numeric
# keys are cast. sitl_logs_dir steers 'latest' resolution.
_CFG_STR = ("real", "sitl", "sdf", "prop", "motor", "battery", "outdir",
            "ledger", "sitl_logs_dir")
_CFG_NUM = ("motors", "mass", "vmin", "vmax", "vstep", "max_climb", "max_roll",
            "max_accel")


def load_compare_config(path: str | None) -> dict:
    if path is None:
        path = find_config("compare.ini")
    if not path or not os.path.exists(path):
        return {}
    cp = configparser.ConfigParser()
    cp.read(path)
    if "compare" not in cp:
        return {}
    out = {}
    for k, v in cp["compare"].items():
        if k in _CFG_NUM:
            out[k] = int(v) if k == "motors" else float(v)
        elif k in _CFG_STR:
            out[k] = v
        else:
            print(f"[config ] ignoring unknown key '{k}' in {path}")
    if out:
        print(f"[config ] defaults from {path}: {', '.join(sorted(out))}")
    return out


# ---------------------------------------------------------------------------
def main() -> None:
    """CLI: validate the SITL/VITERNA model against a real flight, log vs log.

    Inputs : a REAL and a SITL flight log (or chop_log CSV dirs) + the SDF
             model + propeller CSV.  Output: comparison plots and report.txt
             mapping each gap to the SDF aero coefficient to adjust.

    This is the validation end of the pipeline; the other CLIs are forward
    predictions from data sheets:
      * gazebo_ald_params.py builds the SDF and sdf_aero_performance.py
        predicts from it; THIS checks that prediction against measured flight
        and points back at the coefficients to retune.
    """
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", default=None,
                    help="REAL flight: a chop_log CSV dir OR a .bin log (auto-chopped)")
    ap.add_argument("--sitl", default=None,
                    help="SITL run: CSV dir, .bin log, or 'latest'/'latest-N' "
                         "(newest SITL log via LASTLOG.TXT)")
    ap.add_argument("--sdf", default=None, help="SDF model file (VITERNA aero)")
    ap.add_argument("--prop", default=None,
                    help="propeller CSV (rpm,v_ms,thrust_N,torque_Nm)")
    ap.add_argument("--motor", default=None,
                    help="motor XML for the current/power prediction (default: ./motor.xml)")
    ap.add_argument("--battery", default=None,
                    help="battery XML for the current/power prediction (default: ./battery.xml)")
    ap.add_argument("--motors", type=int, default=None,
                    help="rotor count for the power prediction (default: SDF n_motors)")
    ap.add_argument("--outdir", default=None,
                    help="output directory (default plots/compare)")
    ap.add_argument("--vmin", type=float, default=None,
                    help="min cruise airspeed (m/s, default 8)")
    ap.add_argument("--vmax", type=float, default=None,
                    help="max airspeed for bins/predict (default 60)")
    ap.add_argument("--vstep", type=float, default=None,
                    help="airspeed bin width (m/s, default 2)")
    ap.add_argument("--max-climb", type=float, default=None,
                    help="|climb rate| ceiling for level-cruise samples (m/s, default 1.5)")
    ap.add_argument("--max-roll", type=float, default=None,
                    help="|roll| ceiling for the binned (wings-level) signature "
                         "(deg, default 10); the regression keeps banked samples "
                         "to 45 deg with load-factor-corrected CL")
    ap.add_argument("--max-accel", type=float, default=None,
                    help="|dV/dt| ceiling for quasi-steady cruise samples "
                         "(m/s^2, default 0.5)")
    ap.add_argument("--mass", type=float, default=None,
                    help="real aircraft mass (kg) for level-cruise CL; default = SDF mass")
    ap.add_argument("--config", default=None,
                    help="defaults INI (default: config/compare.ini, [compare] section)")
    ap.add_argument("--write-sdf", nargs="?", const="auto", default=None,
                    metavar="PATH",
                    help="write a copy of the SDF with the fitted revisions applied "
                         "(default PATH: <outdir>/<sdf-stem>-revised.sdf)")
    ap.add_argument("--ledger", default=None,
                    help="run-ledger CSV (default: <outdir>/../compare_runs.csv)")
    ap.add_argument("--no-ledger", action="store_true",
                    help="do not append this run to the ledger CSV")
    ap.add_argument("--chop-tier", type=int, choices=[1, 2, 3], default=3,
                    help="tier for auto-chopping a .bin input (default 3 = full data)")
    ap.add_argument("--chop-hz", type=float, default=50.0,
                    help="downsample Hz when auto-chopping a .bin input (default 50)")
    ap.add_argument("--rechop", action="store_true",
                    help="re-chop a .bin input even if its *_chopped dir already exists")
    ap.add_argument("--pre", type=float, default=10.0, help="transition window pre-event (s)")
    ap.add_argument("--post", type=float, default=3.0, help="transition window post-event (s)")
    ap.add_argument("--track-pre", type=float, default=15.0,
                    help="ground-track window before the transition event (s); wider than "
                         "--pre to capture the whole deceleration into the 'done' event")
    ap.add_argument("--track-post", type=float, default=5.0,
                    help="ground-track window after the transition event (s)")
    ap.add_argument("--dt", type=float, default=0.02, help="resample dt for xcorr (s)")
    ap.add_argument("--no-show", action="store_true", help="save plots without displaying")
    args = ap.parse_args()

    # config-file defaults fill anything the CLI left unset; CLI always wins
    cfg = load_compare_config(args.config)
    for k in _CFG_STR + _CFG_NUM:
        if k != "sitl_logs_dir" and getattr(args, k, None) is None and k in cfg:
            setattr(args, k, cfg[k])
    args.outdir = args.outdir or "plots/compare"
    args.vmin = 8.0 if args.vmin is None else args.vmin
    args.vmax = 60.0 if args.vmax is None else args.vmax
    args.vstep = 2.0 if args.vstep is None else args.vstep
    args.max_climb = 1.5 if args.max_climb is None else args.max_climb
    args.max_roll = 10.0 if args.max_roll is None else args.max_roll
    args.max_accel = 0.5 if args.max_accel is None else args.max_accel
    missing = [k for k in ("real", "sitl", "sdf", "prop") if not getattr(args, k)]
    if missing:
        ap.error("missing " + ", ".join("--" + m for m in missing)
                 + " (give them on the CLI or in config/compare.ini)")
    sitl_spec = args.sitl
    args.sitl = resolve_latest_sitl(args.sitl,
                                    cfg.get("sitl_logs_dir", "~/ardupilot/logs"))
    if args.sitl != sitl_spec:
        print(f"[sitl   ] '{sitl_spec}' -> {args.sitl}")

    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")

    sdf_path = find_config(args.sdf, "aero")
    aero, sdf_mass, propulsion, info = load_sdf_model(sdf_path)
    mass = args.mass if args.mass else sdf_mass
    sdf_prop_info = {"max_rad_per_s": propulsion.max_rad_per_s or 2100.0,
                     "n_motors": propulsion.n_motors or 4}
    mass_note = "" if args.mass is None else f" (override; SDF sums {sdf_mass:.3f})"
    print(f"mass={mass:.3f} kg{mass_note}  S={aero.area} m^2  AR={aero.AR}  "
          f"CL0={aero.CL0} CLa={aero.CLa} CD0={aero.CD0}")

    prop = PropTable(find_config(args.prop, "propellers"))
    real_dir = ensure_chopped(args.real, "real", args.outdir, args.chop_tier,
                              args.chop_hz, args.rechop)
    sitl_dir = ensure_chopped(args.sitl, "sitl", args.outdir, args.chop_tier,
                              args.chop_hz, args.rechop)
    real = load_log(real_dir, "real")
    sitl = load_log(sitl_dir, "sitl")
    print(f"loaded real: {len(real.msgs)} msg types, {len(real.events)} events")
    print(f"loaded sitl: {len(sitl.msgs)} msg types, {len(sitl.events)} events")

    print("computing cruise signatures...")
    real_sig = cruise_signature(real, prop, sdf_prop_info, args.vmin, args.vmax,
                                args.vstep, args.max_climb, args.max_roll,
                                args.max_accel)
    sitl_sig = cruise_signature(sitl, prop, sdf_prop_info, args.vmin, args.vmax,
                                args.vstep, args.max_climb, args.max_roll,
                                args.max_accel)

    print("computing hover signatures...")
    hover_real = hover_signature(real, args.max_climb)
    hover_sitl = hover_signature(sitl, args.max_climb)

    print("computing transition dynamics...")
    tm = transition_metrics(real, sitl, args.pre, args.post, args.dt)

    print("computing transition ground-tracks (forward & back)...")
    tracks = transition_tracks(real, sitl, args.track_pre, args.track_post, args.dt)
    for key in ("forward", "back"):
        tt = tracks[key]
        if tt.have:
            print(f"  {key}: peak excursion real {tt.overshoot_real:.1f} m / "
                  f"SITL {tt.overshoot_sitl:.1f} m (x{tt.overshoot_ratio:.2f})")
        else:
            print(f"  {key}: skipped ({tt.note})")

    # powertrain current/power prediction (defaults to config/motors + config/batteries)
    motor_xml = find_config(args.motor or "motor.xml", "motors")
    motor_xml = motor_xml if os.path.exists(motor_xml) else None
    battery_xml = find_config(args.battery or "battery.xml", "batteries")
    battery_xml = battery_xml if os.path.exists(battery_xml) else None
    n_mot = args.motors or sdf_prop_info["n_motors"]
    elec_pred = hover_pred = powertrain = None
    if motor_xml and battery_xml:
        try:
            Vpred = np.linspace(2.0, args.vmax, 80)
            elec_pred = predicted_elec(aero, mass, find_config(args.prop, "propellers"),
                                       motor_xml, battery_xml, n_mot, Vpred)
            hover_pred = predicted_hover(find_config(args.prop, "propellers"),
                                         motor_xml, battery_xml, n_mot, mass)
            powertrain = {"motor": load_motor(motor_xml),
                          "prop": load_propeller_csv(find_config(args.prop, "propellers")),
                          "battery": load_battery(battery_xml),
                          "n_motors": n_mot}
            print(f"powertrain model: {os.path.basename(motor_xml)} + "
                  f"{os.path.basename(battery_xml)}, {n_mot} motors")
        except Exception as e:
            print(f"powertrain prediction skipped: {e}")
    else:
        print("no motor/battery XML -> current/power prediction + power fit skipped")

    print("fitting model update (power-primary regression)...")
    revs, diag = recommend_revisions(real_sig, aero, mass, powertrain,
                                     v_step=args.vstep)

    make_plots(real_sig, sitl_sig, aero, mass, real, sitl, tm,
               args.outdir, not args.no_show, args.pre, args.post, args.dt, elec_pred,
               hover_real=hover_real, hover_sitl=hover_sitl, hover_pred=hover_pred,
               pow_curve=diag.get("power_thrust_curve"))
    make_track_plots(tracks, args.outdir, not args.no_show, args.track_pre)
    metrics = write_report(real_sig, sitl_sig, tm, real, sitl, aero, mass,
                           args.outdir, elec_pred, tracks,
                           hover_real=hover_real, hover_sitl=hover_sitl,
                           hover_pred=hover_pred, revs=revs, diag=diag)

    print(f"[metrics] {save_metrics(metrics, args.outdir)}")
    if not args.no_ledger:
        ledger = args.ledger or os.path.join(
            os.path.dirname(os.path.abspath(args.outdir)), "compare_runs.csv")
        append_ledger(ledger, metrics, sdf_path)
    if args.write_sdf:
        out_sdf = args.write_sdf
        if out_sdf == "auto":
            stem = os.path.splitext(os.path.basename(sdf_path))[0]
            out_sdf = os.path.join(args.outdir, stem + "-revised.sdf")
        write_revised_sdf(sdf_path, revs, out_sdf,
                          f"real={args.real} sitl={args.sitl}")


if __name__ == "__main__":
    main()
