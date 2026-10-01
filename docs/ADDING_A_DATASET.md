# Using another data set

The model only needs **gait-cycle folders**. There are two ways to make them.

## A. Write the gait-cycle folders yourself

Make one folder per gait cycle, at any depth below a set folder, and put three sets side by side:

```
my_data/
  training/    <any sub-folders>/<cycle>/cycle.npz + cycle.json
  validation/  (optional; without it 1/9 of training is used)
  testing/
```

Then run `gait-balance train --data my_data`. Without a `validation/` folder, use
`gait-balance train --train my_data/training --test my_data/testing`.

`single_imu_gait_balance.cycles.write_cycle(folder, imu, ia, rcia, info)` writes a folder. Its contents:

| File | Content |
|---|---|
| `cycle.npz` | `imu` (101 × 6), `ia` (101 × 2), `rcia` (101 × 2), float |
| `cycle.json` | `cycle_time_s` (required); `subject`, `trial`, `speed` and any other fields (optional) |
| `ground_truth.csv`, `imu_input.csv` | the same numbers as text, for reading |

The 101 samples run from 0 % to 100 % of the gait cycle. The cycle starts and ends with successive heel strikes of the reference limb.

- **`imu`**: sacral IMU in body axes, x anterior, y up, z to the right.
  - Columns are acc x, y, z (m/s², including gravity) and gyro x, y, z (rad/s).
  - Use a 15 Hz zero-phase low-pass filter.
  - Mirror cycles whose reference limb is the right one: negate acc z, gyro x and gyro y (see `imu.mirror_right_limb`).
  - `imu.to_body_axes` re-orders the sensor axes, and `imu.imu_cycle` does all of these steps.
- **`ia`**: sagittal and frontal inclination angle in degrees. See `inclination.inclination_angles` for the definition and signs.
- **`rcia`**: rate of change of the IA in deg/s. Use `inclination.rcia_from_ia` (GCV smoothing spline on the motion-capture frames).

## B. Add a data set module

To let others download and convert a public data set with two commands, add a module next to
`src/single_imu_gait_balance/datasets/kuopio.py` and register it in `datasets/__init__.py`:

```python
NAME = "mydata"                      # gait-balance download mydata / build mydata
CITATION = "Authors. Title. Journal Year. doi:... Data: doi:... (license)"

def download(raw_dir, method="partial", zip_dir=None, with_mtb=False):
    """Fetch the needed files into raw_dir (see download.py for helpers).
    with_mtb (optional): also fetch raw Xsens recordings; only passed with --with-mtb."""

def build(raw_dir, out_dir, workers=None):
    """Write out_dir/training, out_dir/validation and out_dir/testing gait-cycle folders."""
```

The building blocks in the package:

| Module | Provides |
|---|---|
| `c3d.read_c3d` | markers and force plates from a C3D file |
| `signals` | gap filling, filters, time normalization |
| `forceplate.ground_reaction` | summed ground reaction and its COP |
| `com.body_com` | 7-segment COM from named points (`com.REQUIRED_POINTS`) |
| `gait` | plate contacts, heel strikes from the heel marker |
| `inclination` | progression frame, IA, RCIA |
| `imu` | body axes, filtering, cycle cut, mirroring |
| `download` | checked file download, partial download of remote zip files |
| `xsens` | reading Xsens `.mtb` recordings (needs the Xsens MT Software Suite) |

`kuopio.py` is a complete example. It covers a marker set without trochanter or metatarsal-head markers,
three force plates, and an IMU stored as strapdown increments.

Respect the data set's license and cite it. Download the data from its original repository rather than
copying it into this repository.
