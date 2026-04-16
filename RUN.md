# Run Commands

## Performance estimator

```bash
py -3 performance.py PER3_7x15E.txt motor.xml battery.xml polars.txt --plane plane.xml --vmin 0 --vmax 90 --vstep 5
```

python performance.py PER3_8x10E.txt motor.xml battery.xml polars.txt --plane plane.xml --vmin 0 --vmax 90 --vstep 5


python motor_prop_performance.py PER3_525x8E.txt motor_V2808_1950.xml battery_6S.xml --vmin 0 --vmax 0 --vstep 2

python motor_prop_performance.py PER3_7x15E.txt motor.xml battery.xml --vmin 0 --vmax 0 --vstep 2
python motor_prop_performance.py PER3_7x15E.txt motor.xml battery.xml --vmin 0 --vmax 60 --vstep 5 --motors 4
python motor_prop_performance.py PER3_8x10E.txt motor.xml battery.xml --vmin 0 --vmax 0 --vstep 5 --motors 4