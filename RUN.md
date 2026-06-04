# Run Commands

## Config layout

Data files live under `config/`, grouped by purpose:

| Folder               | Contents                                              |
|----------------------|-------------------------------------------------------|
| `config/propellers/` | APC PERFILES propeller tables (`PER3_*.txt`)          |
| `config/motors/`     | motor spec XML (`motor.xml`, `motor_V2808_1950.xml`)  |
| `config/batteries/`  | battery spec XML (`battery.xml`, `battery_6S.xml`)    |
| `config/planes/`     | plane/airframe parameter XML (`plane.xml`, …)         |
| `config/aero/`        | Flow5 polars, T5 sweeps, VITERNA aero SDF             |
| `config/params/`     | generated ArduPilot `.param` files                    |

The scripts resolve a **bare filename** (e.g. `motor.xml`, `PER3_7x11E.txt`)
against these folders via `config_paths.find_config`, so the commands below
still work unchanged. An explicit relative/absolute path (e.g.
`../ardupilot_gazebo/.../PER3_7x11E.csv`) is used as-is.

## Performance estimator

```bash
py -3 performance.py PER3_7x11E.txt motor.xml battery.xml polars.txt --plane plane.xml --vmin 0 --vmax 90 --vstep 5
```

python performance.py PER3_7x11E.txt motor.xml battery.xml polars.txt --plane plane.xml --vmin 0 --vmax 90 --vstep 5


python motor_prop_performance.py PER3_525x8E.txt motor_V2808_1950.xml battery_6S.xml --vmin 0 --vmax 0 --vstep 2

python motor_prop_performance.py PER3_7x11E.txt motor.xml battery.xml --vmin 0 --vmax 0 --vstep 2
python motor_prop_performance.py PER3_7x11E.txt motor.xml battery.xml --vmin 0 --vmax 120 --vstep 5 --motors 4
python motor_prop_performance.py PER3_8x10E.txt motor.xml battery.xml --vmin 0 --vmax 0 --vstep 5 --motors 4


# 比较模型与飞行日志

python compare_logs.py   --real ../Ardu_Log/out   --sitl ~/ardupilot/logs/00000107.BIN   --sdf  ../ardupilot_gazebo/models/waterdrop/model-aero-VITERNA-m.sdf   --prop ../ardupilot_gazebo/models/waterdrop/propellers/PER3_7x11E.csv   --outdir plots/compare --no-show