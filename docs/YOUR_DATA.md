# Using your own recordings

Your own data can be turned into gait-cycle folders in two ways. Then you can use them for **training**,
**testing** a trained model, or simply as **input and output files** (`imu_input.csv` and `ground_truth.csv` in
every gait-cycle folder).

| You have | Command | Window (`gait-balance gui`, tab 1) |
|---|---|---|
| A. Raw recordings: a static C3D, a walking C3D and the IMU file of each trial | `gait-balance convert-trials` | *my trial folders (C3D + IMU)* |
| B. Gait cycles processed elsewhere, as `.npz` or `.mat` arrays | `gait-balance import-arrays` | *my processed gait cycles* |

Both write one folder per gait cycle (see [ADDING_A_DATASET.md](ADDING_A_DATASET.md#a-write-the-gait-cycle-folders-yourself))
plus `trials.csv` (every trial: used, or the reason why not) and `README.txt` (a summary).

## A. Trial folders: static C3D + walking C3D + IMU file

### What a trial folder holds

```
my_study/
  P01/
    walk_01/
      static.c3d            standing still on the force plates (calibration)
      walk_01.c3d           one walk over three or four floor force plates
      walk_01.mtb           the IMU recording of the same walk (.mtb, .mat or .csv)
      trial_info.json       optional
    walk_02/
      ...
  P02/
    ...
```

- **Where.** Every folder below the chosen folder that holds two C3D files and an IMU file is one trial. The
  layout above it is up to you; it is kept in the output.
- **Static or walking.** The C3D whose name contains *static*, *subcali*, *calib* or *stand* is the static
  trial. Otherwise the C3D in which the pelvis moves less than 300 mm is the static one, and the one in which it
  moves more than 800 mm is the walk.
- **The same static file in many folders.** Copy it into each trial folder. The subject model is computed only
  once for identical files.
- **`trial_info.json`** (all keys optional):

  ```json
  {"subject": "P01", "group": "young", "speed": "comfortable", "body_mass_kg": 68.5,
   "static_c3d": "static.c3d", "walking_c3d": "walk_01.c3d", "imu_file": "walk_01.mtb",
   "sensor": "00B41234", "imu_offset_s": 0.0}
  ```

  The body mass is otherwise taken from the force plates during the static trial.

### Requirements for the recordings

- **Markers.** The default marker set is the one of Yu et al. (2023): pelvis (ASIS, PSIS), acromions, and clusters
  on the thighs, shanks and feet (`configs/markersets/yu2023.json`).
  - If your labels differ, copy that file, change only the right-hand side of `labels`, and choose your file as
    the marker set.
  - Required on both sides: ASIS, PSIS, greater trochanter, femoral epicondyles, malleoli, first metatarsal
    head, acromion and heel.
- **Force plates.** The walk crosses three or four force plates set in the floor, one foot per plate:
  - four plates: the gait cycle runs from the heel strike on the second plate to the heel strike on the fourth;
  - three plates: it runs from the heel strike on the second plate to the next heel strike of the same foot,
    found from the heel marker.
- **IMU.** One sensor on the sacrum, started together with the motion capture (otherwise give the delay as
  `imu_offset_s`).
  - `.mtb`: Xsens MT Manager files (needs the Xsens MT Software Suite, see the README).
  - `.mat`: one struct per sensor, named by the sensor ID, with fields `Acc` (m/s²) and `Gyro` (rad/s).
  - `.csv`: columns `acc_x_m_s2, acc_y_m_s2, acc_z_m_s2, gyro_x_rad_s, gyro_y_rad_s, gyro_z_rad_s` (as written by
    `gait-balance mtb2csv`).
  - If the file holds several sensors, give the ID of the sacral one (`--sensor`).
  - **Sensor axes.** Say which sensor axis points forward, up and to the right (`--imu-axes`). The default
    `-z,x,-y` is an Xsens MTw with its x axis up and its case facing backwards.

### Command

```
gait-balance convert-trials D:\my_study --out D:\my_study_cycles --split 80/10/10
```

| Option | Meaning |
|---|---|
| `--markerset` | marker set name in `configs/markersets` or the path of your own file (default `yu2023`) |
| `--sensor` | ID of the sacral sensor, if the IMU files hold several |
| `--imu-axes` | sensor axes pointing forward, up and right (default `-z,x,-y`) |
| `--imu-rate` | IMU sampling rate in Hz (default 100) |
| `--imu-offset` | start of the IMU recording after the first C3D frame, s (default 0) |
| `--split 80/10/10` | divide the gait cycles at random by trial into `training/`, `validation/` and `testing/` |
| `--split-file` | divide them by a JSON file: `{"training": [...], "validation": [...], "testing": [...]}` with trial folder names |
| (neither) | no division: all gait cycles go into the output folder, e.g. to test a model on them |

### What is checked

A trial is not used, and `trials.csv` says why, when:
- a foot is not fully on its force plate, or the body weight is not on the plates during the cycle;
- a marker label does not fit the static trial (a cluster marker whose median distance from its fitted position
  exceeds 10 mm, or an acromion or first-metatarsal marker whose distance to the pelvis or heel changed by more
  than 50 or 20 mm);
- the left and right hips or knees appear swapped;
- markers needed for the COM are missing.

## B. Processed gait cycles as arrays

If your gait cycles are already processed (the IMU input and the IA of each cycle), describe your files in a
small JSON **spec** file. Example for files `P01/cycle01.npz` that hold arrays `imu` (n × 6), `ia` (n × 2) and a
number `cycle_time_s`:

```json
{
  "imu_files": "*/cycle*.npz",
  "imu_key": "imu",
  "ia_key": "ia",
  "cycle_time_key": "cycle_time_s",
  "frame_rate_hz": 100,
  "fields_from_path": {"subject": 0},
  "folder": "{subject}/{name}"
}
```

| Key | Meaning |
|---|---|
| `imu_files` | pattern of the IMU files below the chosen folder (`**` searches all sub-folders) |
| `imu_key`, `imu_columns` | array with the IMU input; columns (default the first six): acc x, y, z, gyro x, y, z |
| `ia_file` | if the IA is in another file: text replacements that turn the IMU file's path into it, e.g. `[["/imu/", "/ia/"]]` |
| `ia_key`, `ia_columns` | array with the IA in degrees; columns (default `[0, 1]`): sagittal, frontal |
| `cycle_time_key` | number with the gait-cycle duration in seconds |
| `frame_rate_hz` | motion-capture rate of the IA; the RCIA is computed from the IA as in the paper |
| `rcia_key` | optional: use a stored RCIA (deg/s) instead |
| `fields_from_path` | optional: fields from the folder names (0 = the first folder below the chosen folder) |
| `folder` | optional: layout of the output, from `{name}` (the file name) and the fields |

Any number of samples per cycle works; everything is resampled to 101 time steps. The IMU must already follow
the model's convention (body axes x forward, y up, z right; 15 Hz low-pass; right-limb cycles mirrored).

```
gait-balance import-arrays D:\my_arrays --spec D:\my_arrays\spec.json --out D:\my_cycles --split 80/10/10
```

`--split` and `--split-file` work as for trial folders.

## Then

| Goal | Command (or window tab) |
|---|---|
| Train and test | `gait-balance train --data D:\my_cycles` (tab 2). Without `validation/`, 1/9 of the training cycles are used. |
| Tune the hyper-parameters | `gait-balance tune --data D:\my_cycles` (tab 3) |
| Test a trained model | `gait-balance evaluate --model <model.pt> --test D:\my_cycles` (tab 4) |
| Only the numbers | read `imu_input.csv` and `ground_truth.csv` in each gait-cycle folder |

Keep participant data on your own computer, and follow your ethics approval for any sharing.
