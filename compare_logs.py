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

Outputs (into --outdir): drag-signature plot, lift/trim plot (with the SDF-predicted
curve overlaid), transition overlay plot, transition-tracks plot, and report.txt
(metrics + tuning map).

--real / --sitl accept EITHER a chop_log CSV directory OR a raw .bin log. A .bin is
auto-chopped (full data: tier 3 + all sensors) into <outdir>/<logname>_chopped and
reused on later runs (--rechop to force regeneration).

Usage:
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
import csv
import importlib.util
import math
import os
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
from motor_prop_performance import load_battery, load_motor
from config_paths import find_config

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


def cruise_signature(log: LogData, prop: PropTable, sdf_prop_info: dict,
                     v_lo: float, v_hi: float, v_step: float,
                     max_climb: float) -> CruiseSig | None:
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

    t_q = t_arsp[mask]
    v_q = v_arsp[mask]
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
    elec_src = "n/a"
    if log.has("BAT", "Curr", "Volt"):
        bt = log.rel("BAT")
        curr_raw, volt_raw = log.col("BAT", "Curr"), log.col("BAT", "Volt")
        binst = -1
        if "Inst" in log.msgs["BAT"]:
            # propulsion pack = the battery instance drawing the most current
            # (avoids picking an avionics/secondary pack that reads ~0 A)
            inst = log.col("BAT", "Inst")
            best = -1.0
            for i in sorted(set(int(x) for x in inst)):
                med = float(np.nanmedian(np.abs(curr_raw[inst == i])))
                if med > best:
                    best, binst = med, i
            sel = inst == binst
            bt, curr_raw, volt_raw = bt[sel], curr_raw[sel], volt_raw[sel]
        cur_q = interp_at(bt, curr_raw, t_q)
        pow_q = cur_q * interp_at(bt, volt_raw, t_q)
        elec_src = f"BAT[{binst}]" if binst >= 0 else "BAT"
    else:
        cur_q = np.full(len(t_q), np.nan)
        pow_q = np.full(len(t_q), np.nan)

    # bin by airspeed, take medians
    edges = np.arange(v_lo, v_hi + v_step, v_step)
    centers, t_b, a_b, p_b, q_b, cu_b, pw_b = [], [], [], [], [], [], []
    for i in range(len(edges) - 1):
        b = (v_q >= edges[i]) & (v_q < edges[i + 1])
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

    print(f"  [{log.name}] cruise: {len(t_q)} samples, {len(centers)} bins "
          f"(airspeed {vsrc}, thrust {thr_src}, elec {elec_src})")
    return CruiseSig(np.array(centers), np.array(t_b), np.array(a_b),
                     np.array(p_b), np.array(q_b), thr_src, aoa_src,
                     len(t_q), note, current=np.array(cu_b),
                     power=np.array(pw_b), elec_src=elec_src)


@dataclass
class Revision:
    tag: str
    old: float
    new: float            # NaN when held at prior
    kind: str             # 'identified' | 'combination' | 'held'
    note: str
    sigma: float = float("nan")


# Identifiability thresholds: what variation the data must show to separate
# coupled parameters (equation-error sense). Below these, the coupled pair is
# only known as a combination, so we move the offset onto the leading term and
# hold its partner at the Flow5 prior.
_CL_SPREAD_MIN = 0.25     # (CLmax-CLmin)/CLmean to split CD0 from induced (eff)
_AOA_SPAN_MIN = 3.0       # deg of AoA range to split CL0 from CLa
_MIN_BINS = 3


def recommend_revisions(real_sig: "CruiseSig | None", aero: AeroModel, mass: float):
    """Identifiability-aware model update.

    Returns (revisions, diag). Each Revision is tagged 'identified' (the data
    span resolves it), 'combination' (only a coupled sum is known -> offset put
    on the leading term, partner held), or 'held' (not excited -> keep Flow5
    prior). This replaces blind single-term fits: a coefficient is only changed
    when the flight actually constrains it.
    """
    revs: list[Revision] = []
    diag = {"n_bins": 0, "eff_identifiable": False, "cla_identifiable": False}
    if real_sig is None:
        return revs, diag
    W, S, rho = mass * G, aero.area, aero.rho

    CLs, dCDs, aoas, apreds, vs = [], [], [], [], []
    for v, T, aoa in zip(real_sig.v, real_sig.thrust, real_sig.aoa):
        if v <= 0:
            continue
        q = 0.5 * rho * v * v
        CL = W / (q * S)
        a = alpha_trim(aero, CL)
        if math.isnan(a):
            continue
        CLs.append(CL)
        vs.append(v)
        if np.isfinite(T):
            dCDs.append(T / (q * S) - float(CD_of_alpha(aero, np.array([a]))[0]))
        if real_sig.aoa_src.startswith("AOA") and np.isfinite(aoa):
            aoas.append(aoa)
            apreds.append(math.degrees(a))
    if not CLs:
        return revs, diag

    CLlo, CLhi, CLmean = min(CLs), max(CLs), float(np.mean(CLs))
    cl_spread = (CLhi - CLlo) / CLmean if CLmean else 0.0
    diag.update(n_bins=len(CLs), V=(min(vs), max(vs)), CL=(CLlo, CLhi),
                cl_spread=cl_spread, aoa=(min(aoas), max(aoas)) if aoas else None)

    # --- drag: CD0 vs induced (eff) ---
    if dCDs:
        dCD0 = float(np.mean(dCDs))
        sig = float(np.std(dCDs)) if len(dCDs) > 1 else float("nan")
        if cl_spread >= _CL_SPREAD_MIN and len(dCDs) >= _MIN_BINS:
            diag["eff_identifiable"] = True
            revs.append(Revision("CD0", aero.CD0, aero.CD0 + dCD0, "identified",
                                 f"CL spans {CLlo:.2f}-{CLhi:.2f} -> parasitic separable; "
                                 f"mean dCD={dCD0:+.4f}", sig))
        else:
            revs.append(Revision("CD0", aero.CD0, aero.CD0 + dCD0, "combination",
                                 f"single CL~{CLmean:.2f}: induced fixed, whole dCD={dCD0:+.4f} "
                                 "folded into CD0", sig))

    # --- lift: CL0 vs CLa ---
    if aoas:
        da = [math.radians(p - o) for p, o in zip(apreds, aoas)]
        mean_da = float(np.mean(da))
        dCL0 = aero.CLa * mean_da
        a_span = max(aoas) - min(aoas)
        if a_span >= _AOA_SPAN_MIN and len(aoas) >= _MIN_BINS:
            diag["cla_identifiable"] = True
            revs.append(Revision("CL0", aero.CL0, aero.CL0 + dCL0, "identified",
                                 f"AoA spans {a_span:.1f} deg -> CL0 separable; "
                                 f"mean da={math.degrees(mean_da):+.2f} deg", float("nan")))
        else:
            revs.append(Revision("CL0", aero.CL0, aero.CL0 + dCL0, "combination",
                                 f"single AoA~{np.mean(aoas):.1f} deg: fixes CL0+CLa*a, "
                                 "folded into CL0", float("nan")))
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
               outdir, show, pre, post, dt, elec_pred=None):
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)

    # ---- Drag signature: total thrust vs airspeed ----
    fig, ax = plt.subplots(figsize=(8, 6))
    if real_sig is not None and np.isfinite(real_sig.thrust).any():
        ax.plot(real_sig.v, real_sig.thrust, "o-", color="C0", label=f"real ({real_sig.thrust_src})")
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
                 elec_pred=None, tracks=None):
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
            P("  no overlapping airspeed bins between the two logs.")
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
    cur_bias = pow_bias = float("nan")
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
        ref = "SITL" if has_se else "model"
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
        P("")
        P(f"  mean current bias : {cur_bias:+.2f} A   ({ref} - real)")
        P(f"  mean power bias   : {pow_bias:+.1f} W   ({ref} - real)")
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

    # ---- tuning map ----
    P("--- TUNING MAP: gap -> SDF coefficient (model-aero-VITERNA-m.sdf) ---")
    suggestions = []
    if np.isfinite(thrust_bias):
        if thrust_bias > 0.3:
            suggestions.append(
                f"SITL needs MORE thrust than real (+{thrust_bias:.2f} N): SITL drag too high "
                "-> LOWER CD0 (parasitic); if gap grows with V^2 it's CD0, if worst at low "
                "V/high CL raise eff (induced).")
        elif thrust_bias < -0.3:
            suggestions.append(
                f"SITL needs LESS thrust than real ({thrust_bias:.2f} N): SITL drag too low "
                "-> RAISE CD0 / lower eff.")
    if np.isfinite(thrust_slope_ratio) and abs(thrust_slope_ratio - 1) > 0.15:
        suggestions.append(
            f"thrust-vs-V slope ratio {thrust_slope_ratio:.2f}!=1: parasitic-drag scaling off "
            "-> adjust CD0 (dominates the V^2 growth).")
    if np.isfinite(aoa_bias) and abs(aoa_bias) > 1.0:
        if aoa_bias > 0:
            suggestions.append(
                f"SITL trims at HIGHER AoA (+{aoa_bias:.2f} deg) for same V: SITL lift too low "
                "-> RAISE CL0 / CLa.")
        else:
            suggestions.append(
                f"SITL trims at LOWER AoA ({aoa_bias:.2f} deg): SITL lift too high "
                "-> LOWER CL0 / CLa.")
    if np.isfinite(pitch_bias) and abs(pitch_bias) > 1.0 and (
            not np.isfinite(aoa_bias) or abs(pitch_bias - aoa_bias) > 1.0):
        suggestions.append(
            f"near-constant pitch offset ({pitch_bias:+.2f} deg) not explained by AoA: "
            "zero-AoA pitching moment / CG -> adjust Cem0 (and CG vs <cp>).")
    if tm.have:
        if np.isfinite(tm.xcorr_peak) and tm.xcorr_peak < 0.6:
            suggestions.append(
                f"low pitch-rate xcorr ({tm.xcorr_peak:.2f}): transition pitch dynamics differ "
                "-> check post-stall block (alpha_stall, CLa_stall, CDa_stall) and Cema.")
        if np.isfinite(tm.rms_pitch) and tm.rms_pitch > 5.0:
            suggestions.append(
                f"high RMS pitch error ({tm.rms_pitch:.1f} deg) in transition "
                "-> high-AoA longitudinal aero (Viterna post-stall: alpha_stall, CLa_stall, "
                "CDa_stall) and pitch-damping suspect.")
        if (np.isfinite(tm.t2v_real) and np.isfinite(tm.t2v_sitl)
                and abs(tm.t2v_sitl - tm.t2v_real) > 1.0):
            d = tm.t2v_sitl - tm.t2v_real
            if d > 0:
                suggestions.append(
                    f"SITL accelerates SLOWER to target airspeed (+{d:.1f} s): excess drag or "
                    "weak thrust -> lower CD0 / check prop CSV & mass (SDF mass vs real).")
            else:
                suggestions.append(
                    f"SITL accelerates FASTER to target airspeed ({d:.1f} s): too little drag "
                    "-> raise CD0.")
    if overshoot_flag is not None:
        _, ratio = overshoot_flag
        if ratio > 1.15:
            suggestions.append(
                f"SITL over-coasts the back transition (peak excursion x{ratio:.2f} vs real): "
                "the aircraft pitches into deep stall to decelerate, so this is the post-stall "
                "DRAG -> RAISE CDa_stall (and check alpha_stall earlier / CLa_stall) so SITL "
                "sheds speed and stops at the hover point instead of overshooting in XY.")
        elif ratio < 0.85:
            suggestions.append(
                f"SITL under-coasts the back transition (peak excursion x{ratio:.2f} vs real): "
                "post-stall drag too high -> LOWER CDa_stall.")
    if not suggestions:
        suggestions.append("no gaps above threshold; aero model agrees with the data in scope.")
    for s in suggestions:
        P(f"  * {s}")
    P("")

    # ---- identifiability-aware model update ----
    P("--- MODEL UPDATE (identifiability-aware) ---")
    P(f"  basis: level cruise, lift=weight, mass={mass:.2f} kg (override with --mass)")
    revs, diag = recommend_revisions(real_sig, aero, mass)
    if diag.get("n_bins", 0) == 0:
        P("  (insufficient real cruise data to constrain any coefficient)")
    else:
        V = diag.get("V", (float("nan"),) * 2)
        CL = diag.get("CL", (float("nan"),) * 2)
        aoaspan = diag.get("aoa")
        aoa_str = (f"AoA {aoaspan[0]:.1f}-{aoaspan[1]:.1f} deg"
                   if aoaspan else "no AoA")
        P(f"  data span: {diag['n_bins']} airspeed bin(s), V {V[0]:.0f}-{V[1]:.0f} m/s, "
          f"CL {CL[0]:.2f}-{CL[1]:.2f} (spread {diag.get('cl_spread', 0):.0%}), {aoa_str}")

        ident = [r for r in revs if r.kind == "identified"]
        combo = [r for r in revs if r.kind == "combination"]
        if ident:
            P("  data-driven (identified):")
            for r in ident:
                s = f" sigma~{r.sigma:.4f}" if np.isfinite(r.sigma) else ""
                P(f"    <{r.tag}>  {r.old:+.6f} -> {r.new:+.6f}  (d {r.new - r.old:+.6f}{s})")
                P(f"        {r.note}")
        if combo:
            P("  combination only (one operating point -- offset put on leading term):")
            for r in combo:
                P(f"    <{r.tag}>  {r.old:+.6f} -> {r.new:+.6f}  (d {r.new - r.old:+.6f})")
                P(f"        {r.note}")

        # Cross-check the drag-based CD0 move against the (more reliable) battery
        # power gap: P_elec ~ T*V/eta, so at cruise the power ratio independently
        # estimates the drag ratio. If power already agrees but the RPM-derived
        # CD0 move is large, the CD0 figure is suspect (thrust-from-RPM is noisy).
        cd0_rev = next((r for r in revs if r.tag == "CD0"), None)
        if (cd0_rev and np.isfinite(pow_bias) and real_sig.power is not None
                and np.isfinite(real_sig.power).any()):
            p_real_mean = float(np.nanmean(real_sig.power))
            rel_pow = pow_bias / p_real_mean if p_real_mean else float("nan")
            rel_cd0 = (abs(cd0_rev.new - cd0_rev.old) / abs(cd0_rev.old)
                       if cd0_rev.old else float("inf"))
            if np.isfinite(rel_pow) and abs(rel_pow) < 0.15 and rel_cd0 > 0.3:
                P(f"  CONFLICT: the CD0 move is {rel_cd0:.0%} but the battery-power gap is only "
                  f"{rel_pow:+.0%}.")
                P(f"        Power says drag is already ~right; the CD0 figure rests on "
                  f"thrust-from-RPM ({real_sig.thrust_src}, noisy here).")
                P("        Prefer the power evidence -- change CD0 little, if at all.")

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
        P("  apply identified/combination terms ONE at a time, re-run SITL, re-compare.")

    report = "\n".join(lines)
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "report.txt"), "w") as f:
        f.write(report + "\n")
    print("\n" + report)
    print(f"\n[written] {os.path.join(outdir, 'report.txt')}")


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
    ap.add_argument("--real", required=True,
                    help="REAL flight: a chop_log CSV dir OR a .bin log (auto-chopped)")
    ap.add_argument("--sitl", required=True,
                    help="SITL run: a chop_log CSV dir OR a .bin log (auto-chopped)")
    ap.add_argument("--sdf", required=True, help="SDF model file (VITERNA aero)")
    ap.add_argument("--prop", required=True, help="propeller CSV (rpm,v_ms,thrust_N,torque_Nm)")
    ap.add_argument("--motor", default=None,
                    help="motor XML for the current/power prediction (default: ./motor.xml)")
    ap.add_argument("--battery", default=None,
                    help="battery XML for the current/power prediction (default: ./battery.xml)")
    ap.add_argument("--motors", type=int, default=None,
                    help="rotor count for the power prediction (default: SDF n_motors)")
    ap.add_argument("--outdir", default="plots/compare", help="output directory")
    ap.add_argument("--vmin", type=float, default=8.0, help="min cruise airspeed (m/s)")
    ap.add_argument("--vmax", type=float, default=60.0, help="max airspeed for bins/predict")
    ap.add_argument("--vstep", type=float, default=2.0, help="airspeed bin width (m/s)")
    ap.add_argument("--max-climb", type=float, default=1.5,
                    help="|climb rate| ceiling for level-cruise samples (m/s)")
    ap.add_argument("--mass", type=float, default=None,
                    help="real aircraft mass (kg) for level-cruise CL; default = SDF mass")
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

    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")

    aero, sdf_mass, propulsion, info = load_sdf_model(find_config(args.sdf, "aero"))
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
                                args.vstep, args.max_climb)
    sitl_sig = cruise_signature(sitl, prop, sdf_prop_info, args.vmin, args.vmax,
                                args.vstep, args.max_climb)

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
    elec_pred = None
    if motor_xml and battery_xml:
        try:
            Vpred = np.linspace(2.0, args.vmax, 80)
            elec_pred = predicted_elec(aero, mass, find_config(args.prop, "propellers"),
                                       motor_xml, battery_xml, n_mot, Vpred)
            print(f"powertrain model: {os.path.basename(motor_xml)} + "
                  f"{os.path.basename(battery_xml)}, {n_mot} motors")
        except Exception as e:
            print(f"powertrain prediction skipped: {e}")
    else:
        print("no motor/battery XML -> current/power prediction skipped")

    make_plots(real_sig, sitl_sig, aero, mass, real, sitl, tm,
               args.outdir, not args.no_show, args.pre, args.post, args.dt, elec_pred)
    make_track_plots(tracks, args.outdir, not args.no_show, args.track_pre)
    write_report(real_sig, sitl_sig, tm, real, sitl, aero, mass, args.outdir,
                 elec_pred, tracks)


if __name__ == "__main__":
    main()
