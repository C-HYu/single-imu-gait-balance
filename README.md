# Single IMU Gait Balance

Dynamic balance during walking, estimated from **one inertial measurement unit (IMU) on the sacrum**.

This repository implements the method of

> Yu C-H, Yeh C-C, Lu Y-F, Lu Y-L, Wang T-M, Lin FY-S, Lu T-W. Recurrent neural network methods for extracting
> dynamic balance variables during gait from a single inertial measurement unit. *Sensors* 2023;23(22):9040.
> https://doi.org/10.3390/s23229040

and demonstrates it on the public **Kuopio gait data set** (Lavikainen et al., 2024). The paper compared four
recurrent networks. This repository contains the one it recommends: a **bidirectional GRU (bi-GRU) trained with
a weighted mean squared error (MSE)**. The paper's own data are not public, so all results here come from the
Kuopio data set.

## What it computes

| | |
|---|---|
| **Ground truth** (motion capture + force plates) | Whole-body centre of mass (COM, 7-segment model) and centre of pressure (COP). From them, the **inclination angle (IA)** of the COP-to-COM vector in the sagittal and frontal planes, and its **rate of change (RCIA)** |
| **Model input** | Sacral IMU over one gait cycle: 3-axis acceleration and 3-axis angular velocity, 101 samples |
| **Model** | 2 bi-GRU layers → dense layer → IA at 101 samples. RCIA is the finite difference of the predicted IA |
| **Loss** | MSE(IA) + λ · MSE(RCIA) |
| **Errors** | RMSE and relative RMSE (rRMSE = RMSE / range of the measured curve × 100 %) |

## Quick start

You need [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda, about 9 GB of free disk
space and an internet connection. An NVIDIA GPU is optional.

**1. Install.** Open an Anaconda Prompt (Windows) or a terminal:

```
git clone https://github.com/C-HYu/single-imu-gait-balance.git
cd single-imu-gait-balance
conda env create -f environment.yml
conda activate gait-balance
```

**2. Get the Kuopio data and convert it.**

```
gait-balance download kuopio
gait-balance build kuopio
```

- `download` fetches only the files needed: the walking C3D files and the IMU files of each participant. That
  is about 7 GB of the 23 GB archives, from [Zenodo](https://doi.org/10.5281/zenodo.10559504), and takes
  roughly 1 hour. If it is interrupted, run it again: it continues where it stopped, and every file is checked
  against the archive's checksum.
- `build` (about 15 minutes) writes one folder per gait cycle into `data/kuopio/training`,
  `data/kuopio/validation` and `data/kuopio/testing`. The split is 80/10/10 by gait cycle. Every rejected
  trial and its reason are listed in `data/kuopio/trials.csv`.

**3. Train and test.**

```
gait-balance train --data data/kuopio --seeds 0 1 2 3 4
```

Results are written to `runs/train_<time>/`: one model per seed, the error of every test cycle, and
`report.md`. Training takes about 3 minutes per seed on a GPU and several times longer on a CPU.

### Commands

| Command | What it does |
|---|---|
| `gait-balance datasets` | list the data sets that can be downloaded |
| `gait-balance download kuopio` | fetch the data (`--method full` downloads the whole archives; `--zip-dir` uses archives you already have) |
| `gait-balance build kuopio` | convert the data into gait-cycle folders |
| `gait-balance train --data data/kuopio` | train and test (`--config configs/paper.json` for the paper's stated settings; `--seeds` for several runs) |
| `gait-balance tune --data data/kuopio --trials 50` | Bayesian search of the hyper-parameters on the validation set; then train with `--config runs/tune_<time>/best_config.json` |
| `gait-balance evaluate --model <model.pt> --test <folder>` | test a saved model on another set of gait cycles |
| `gait-balance xsens-check` | can Xsens `.mtb` files be read on this computer, and what to install if not |
| `gait-balance mtb2csv <file.mtb>` | export an Xsens `.mtb` recording to one CSV per sensor |

## Results on the Kuopio data set

Bi-GRU with weighted MSE (`configs/bigru_wmse.json`), five training runs (seeds 0–4).
- Data: 2,173 gait cycles of 46 participants: 1,738 for training, 217 for validation and 218 for testing.
- As in the paper, the split is by gait cycle, so cycles of one participant can appear in both training and
  test sets.
- Values are the mean (SD) over the test cycles, averaged over the five runs.

| Variable | RMSE | rRMSE (%) | Paper: RMSE | Paper: rRMSE (%) |
|---|---|---|---|---|
| Sagittal IA | 0.68° (0.28) | 3.23 (1.37) | 0.61° (0.24) | 3.82 (1.53) |
| Frontal IA | 0.39° (0.17) | 5.71 (2.95) | 0.46° (0.21) | 5.33 (3.76) |
| Sagittal RCIA | 9.61°/s (5.26) | 3.42 (1.69) | 13.13°/s (5.69) | 5.32 (2.17) |
| Frontal RCIA | 3.33°/s (1.48) | 3.22 (1.45) | 6.38°/s (1.98) | 4.01 (2.08) |

The mean rRMSE over the four variables is **3.90 %**, and its SD over the five runs is 0.04 %. The paper's
values come from its own data set (13 young and 13 older adults) and are shown for orientation only.

## Using another data set

The model reads **gait-cycle folders**. Each folder holds one cycle: `cycle.npz` with the IMU input (101 × 6),
IA (101 × 2) and RCIA (101 × 2), and `cycle.json` with the cycle duration. To train on your own data, write
these folders and run `gait-balance train --data <folder with training/, validation/, testing/>`.
To let others download and convert another public data set, add a small module like
`datasets/kuopio.py`. Both ways are described in [docs/ADDING_A_DATASET.md](docs/ADDING_A_DATASET.md).

## Xsens `.mtb` recordings (optional)

Recordings saved by Xsens MT Manager (`.mtb`) are in a proprietary format. Reading them needs the Xsens
Device API, which comes with the free **Xsens MT Software Suite**.

1. Install the MT Software Suite from the [Xsens download page](https://www.xsens.com/support/software-documentation).
   For MTw Awinda sensors, choose the Awinda release.
2. Run `gait-balance xsens-check`. It tells you whether the software is found. If one more step is needed
   (installing the suite's Python module into this environment), it prints the exact command.

Then:
- `gait-balance mtb2csv recording.mtb` writes one CSV per sensor: packet counter, acceleration (m/s²) and
  angular velocity (rad/s), in sensor axes.
- For the Kuopio data, `gait-balance download kuopio --with-mtb` also fetches the raw recordings (0.5 GB).
  `build` then uses them for the 79 trials whose extracted IMU file is missing. This gives 2,210 instead of
  2,173 gait cycles, and 47 instead of 46 participants.
  Without the MT Software Suite these trials are skipped, and everything else works as usual.

The `.mtb` route was tested with MT Software Suite 4.6 on Windows (its COM interface): it gives the same
signals as the authors' extracted files. Newer versions are supported through their Python module, but this
has not been tested yet.

## Method details

**Inclination angle.** u = (COM − COP)/|COM − COP| and t = Z × u, where X is the walking direction (the COM
displacement over the cycle), Z is vertical and Y = Z × X.
- Sagittal IA = asin(t·Y). It is positive when the COM is ahead of the COP.
- Frontal IA = s·asin(t·X), with s = +1 for a left and −1 for a right reference limb. It is positive when the
  COM is medial to the COP.

**RCIA.**
- Ground truth: the derivative of a cubic smoothing spline fitted to the IA on the motion-capture frames. The
  smoothing parameter is chosen by generalised cross-validation.
- Prediction: central differences of the predicted IA.

**COM.** Seven segments: thighs, shanks, feet and head-arms-trunk, with Winter's mass fractions and COM
positions. The thigh runs from the greater trochanter to the knee centre, the shank from the knee centre to
the medial malleolus, the foot from the lateral malleolus to the first metatarsal head, and head-arms-trunk
from mid-trochanters to mid-acromions.

**COP.** COP of the summed force of all plates, on the plate surface. Forces are low-pass filtered at 25 Hz.

**IMU.**
- Body axes: x anterior, y up, z right.
- 15 Hz zero-phase low-pass filter, then cut to the gait cycle.
- Right-limb cycles are mirrored to look like left-limb ones.
- Resampled to 501 points (cubic), then to 101 points.

**Network and training.**
- Inputs and outputs are scaled to [−1, 1] on the training set.
- Adam optimiser. The epoch with the lowest validation mean rRMSE is kept.
- The RCIA term of the loss is computed in scaled units (derivative over the normalised cycle time), so λ is
  not the same number as in the paper.
- `configs/bigru_wmse.json` holds the author's hyper-parameters, chosen by Bayesian optimisation.
  `configs/paper.json` holds the settings stated in the paper.

**Kuopio conversion** (`datasets/kuopio.py`):
- **Missing markers.** The data set has no trochanter or first-metatarsal-head markers. The functional hip
  centre and the midpoint of the hallux and 4th-toe markers take their place.
- **Gait cycle.** There are three floor plates. A cycle runs from the heel strike on the middle plate to the
  next heel strike of the same foot, which is found from the heel marker.
- **Rejected trials.** A trial is rejected when a foot is not fully on its plate.
- **IMU.** The pelvis IMU signals are taken from the authors' extracted files. Angular velocity comes from the
  strapdown increments, which equal the gyroscope output, so no Xsens software is needed.

## Repository layout

```
configs/                    hyper-parameters (bi-GRU defaults, paper settings, search space)
docs/ADDING_A_DATASET.md    how to use another data set
src/single_imu_gait_balance/
    c3d.py signals.py forceplate.py com.py gait.py rigid.py inclination.py imu.py   ground truth and input
    cycles.py                                     gait-cycle folders
    model.py training.py metrics.py experiment.py bi-GRU, training, errors, tuning
    download.py datasets/kuopio.py                downloading and converting the Kuopio data
    xsens.py                                      optional reading of Xsens .mtb recordings
    cli.py                                        the gait-balance command
tests/                      tests on invented data (pytest)
data/, runs/                created locally, not part of the repository
```

## Citation

If you use this code, please cite the article above (see also `CITATION.cff`). If you use the Kuopio data, also cite:

> Lavikainen J, Vartiainen P, Stenroth L, Karjalainen PA, Korhonen RK, Liukkonen MK, Mononen ME. Gait data from 51
> healthy participants with motion capture, inertial measurement units, and computer vision. *Data in Brief*
> 2024;56:110841. https://doi.org/10.1016/j.dib.2024.110841. Data: https://doi.org/10.5281/zenodo.10559504
> (CC BY 4.0).

BibTeX:

```bibtex
@article{yu2023recurrent,
  author  = {Yu, Cheng-Hao and Yeh, Chih-Ching and Lu, Yi-Fu and Lu, Yi-Ling and Wang, Ting-Ming and
             Lin, Frank Yeong-Sung and Lu, Tung-Wu},
  title   = {Recurrent Neural Network Methods for Extracting Dynamic Balance Variables during Gait
             from a Single Inertial Measurement Unit},
  journal = {Sensors},
  year    = {2023},
  volume  = {23},
  number  = {22},
  pages   = {9040},
  doi     = {10.3390/s23229040}
}

@article{lavikainen2024gait,
  author  = {Lavikainen, Jere and Vartiainen, Paavo and Stenroth, Lauri and Karjalainen, Pasi A. and
             Korhonen, Rami K. and Liukkonen, Mimmi K. and Mononen, Mika E.},
  title   = {Gait data from 51 healthy participants with motion capture, inertial measurement units,
             and computer vision},
  journal = {Data in Brief},
  year    = {2024},
  volume  = {56},
  pages   = {110841},
  doi     = {10.1016/j.dib.2024.110841}
}

@misc{lavikainen2024kuopio,
  author    = {Lavikainen, Jere and Vartiainen, Paavo and Stenroth, Lauri and Karjalainen, Pasi and
               Korhonen, Rami and Liukkonen, Mimmi and Mononen, Mika},
  title     = {Kuopio gait dataset: motion capture, inertial measurement and video-based sagittal-plane
               keypoint data from walking trials},
  publisher = {Zenodo},
  year      = {2024},
  version   = {1.0.0},
  doi       = {10.5281/zenodo.10559504}
}
```

## Licence

The code is released under the MIT licence (`LICENSE`), © 2026 Cheng-Hao Yu, PhD. The Kuopio data are not part of
this repository. They are downloaded from their original source and remain under their own licence (CC BY 4.0).
