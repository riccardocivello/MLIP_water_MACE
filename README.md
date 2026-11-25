# MACE-OFF NPT molecular dynamics of liquid water

Short NPT (constant number of atoms, pressure and temperature) molecular dynamics runs of a periodic box of liquid water, using the **MACE-OFF23** machine-learning interatomic potential on a GPU through [ASE](https://wiki.fysik.dtu.dk/ase/). The runs are submitted with SLURM.

The main purpose is to start from a given structure, rescale the box to a chosen starting density, and let the Berendsen barostat relax the density at 300 K and 1 atm. Running the same structure at several starting densities lets you check that the system converges to the same equilibrium density.

## Contents

- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Input structure: `box_n4.xyz`](#input-structure-box_n4xyz)
- [Python script: `short_NPT_run.py`](#python-script-short_npt_runpy)
- [Batch script: `run_NPT.sh`](#batch-script-run_nptsh)
- [Outputs](#outputs)
- [Running several densities](#running-several-densities)
- [Known limitations](#known-limitations)
- [License](#license)
- [Author](#author)

## Repository layout

| File | Purpose |
|---|---|
| `box_n4.xyz` | Starting structure: 64 water molecules (192 atoms) in a cubic periodic box |
| `short_NPT_run.py` | Python script that rescales the box, attaches MACE-OFF and runs NPT dynamics |
| `run_NPT.sh` | SLURM submission script that sets up the environment and calls the Python script |

## Requirements

- Python 3.11 (the batch script loads `python-scientific/3.11.5-foss-2023b`)
- A CUDA-capable GPU. The script stops with an error if CUDA is not available.
- Python packages: `numpy`, `torch` (CUDA build), `ase`, `mace-torch`
- The MACE-OFF23 **small** model file, `MACE-OFF23_small.model`, downloaded in advance (compute nodes may have no internet access). The models are published in the [MACE-OFF repository](https://github.com/ACEsuit/mace-off). The path to the model is read from the environment variable `MACE_MODEL_PATH` (see [Quick start](#quick-start)).
- A SLURM cluster to use `run_NPT.sh`. The Python script can also be run directly on any machine with a GPU.

Example environment setup (adapt to your cluster):

```bash
python -m venv ~/venvs/mace
source ~/venvs/mace/bin/activate
pip install numpy ase mace-torch      # install a CUDA build of torch matching your GPU/driver
```

## Quick start

1. Tell the script where the model is by setting the environment variable `MACE_MODEL_PATH` in `run_NPT.sh` :

   ```bash
   export MACE_MODEL_PATH="$HOME/models/mace/MACE-OFF23_small.model"
   ```

   If the variable is not set, the script uses `"small"`, and MACE downloads the model automatically. This only works on a machine with internet access.
2. Edit the `#SBATCH` lines and the `module load` / `source` lines in `run_NPT.sh` to match your cluster and virtual environment.
3. Submit the job, passing the structure file, the target density in g/cm³ and, optionally, the number of steps:

```bash
sbatch run_NPT.sh box_n4.xyz 1.00
sbatch run_NPT.sh box_n4.xyz 1.00 48000     # custom number of steps
```

Or run directly without SLURM:

```bash
python -u short_NPT_run.py box_n4.xyz 1.00 24000
```

## Input structure: `box_n4.xyz`

An extended XYZ file with:

- 192 atoms: 64 O and 128 H, i.e. 64 water molecules
- a cubic cell of about 12.43 Å per side (volume about 1920 Å³), periodic in all three directions (`pbc="T T T"`)
- per-atom columns `species`, `pos`, `forces`, `energies`, plus the total `energy` and `stress` in the header line

The initial density of this box is about 0.997 g/cm³. The forces, energies and stress stored in the file are ignored: the script recomputes everything with MACE-OFF.

Any other extended XYZ (or other format readable by `ase.io.read`) with periodic boundary conditions can be used instead.

## Python script: `short_NPT_run.py`

**Usage**

```
python short_NPT_run.py <input_file> <target_density> [nsteps]
```

| Argument | Meaning | Default |
|---|---|---|
| `input_file` | Structure file with periodic boundary conditions | required |
| `target_density` | Density in g/cm³ to which the box is rescaled before the run (must be positive) | required |
| `nsteps` | Number of MD steps | 24000 |

**What the script does, in order**

1. **Diagnostics.** Prints whether CUDA is available, the PyTorch and CUDA versions, and the GPU name. The job is aborted if no GPU is found.
2. **Read the structure.** Loads the file with ASE and checks that the system is periodic in all directions and that the cell volume is positive.
3. **Compute the initial density.** The density in g/cm³ is `rho = M / (0.602214076 * V)`, where `M` is the total mass in amu and `V` the volume in Å³. The constant 0.602214076 is Avogadro's number times the Å³ to cm³ conversion (6.022×10²³ × 10⁻²⁴).
4. **Rescale the box.** The cell and the atomic positions are scaled by `f = (rho_initial / rho_target)^(1/3)`, so that the volume matches the requested density. Because the scaling is isotropic and applied to all atoms, molecules are stretched or compressed as a whole. The Berendsen barostat later relaxes this.
5. **Attach MACE-OFF.** Loads `MACE-OFF23_small.model` on the GPU in single precision (`float32`) and times the initialisation and the first energy evaluation (the first call is slower because of CUDA warm-up).
6. **Initial velocities.** Draws velocities from a Maxwell-Boltzmann distribution at 300 K.
7. **Set up the NPT dynamics** with ASE's `NPTBerendsen`:

   | Parameter | Value |
   |---|---|
   | Time step | 0.5 fs |
   | Temperature | 300 K |
   | Thermostat time constant `taut` | 100 fs |
   | Pressure | 1.01325 bar (the value is written as `1.01325 * units.bar`, i.e. about 1 atm) |
   | Barostat time constant `taup` | 1000 fs |
   | Compressibility | 4.5×10⁻⁵ bar⁻¹ (approximate value for liquid water) |

   With the defaults, 24000 steps correspond to 12 ps of simulated time.
8. **Run and record.** Every 100 steps the script writes a line to the log file and a frame to the trajectory (see [Outputs](#outputs)). The initial configuration is also saved as the first frame. The trajectory is closed properly even if the run is interrupted.
9. **Timing summary.** Prints the initialisation time, first energy evaluation time, dynamics set-up time and the total MD runtime.

Berendsen thermostats and barostats give the correct average temperature and density but do not sample the exact NPT ensemble (fluctuations are suppressed). They are well suited to equilibration, not to computing fluctuation-dependent properties.

## Batch script: `run_NPT.sh`

A SLURM script. Everything is set at the top with `#SBATCH` directives.

| Setting | Value | Notes |
|---|---|---|
| `--job-name` | `mace_md` | Name in the queue |
| `--partition` | `skylake-gpu` | Cluster-specific, change to your GPU partition |
| `--ntasks` / `--cpus-per-task` | 1 / 4 | One process with 4 CPU threads |
| `--mem` | 16G | |
| `--gres` | `gpu:p100:1` | One NVIDIA P100 GPU, cluster-specific |
| `--time` | 00:10:00 | 10 minutes wall time. Increase it for longer runs |
| `--output` / `--error` | `mace_%j.out` / `mace_%j.err` | `%j` is the job ID |

**What it does**

1. Loads the Python module and activates the virtual environment at `~/venvs/mace`.
2. Sets `OMP_NUM_THREADS`, `MKL_NUM_THREADS` and `OPENBLAS_NUM_THREADS` to the number of CPUs allocated.
3. Moves to the directory from which the job was submitted.
4. Reads the arguments: structure file (required), target density (required) and number of steps (optional, default 24000). It exits with a message if one of the required arguments is missing.
5. Starts `nvidia-smi` in the background, logging GPU timestamp, utilisation, memory use and power draw every 5 seconds to `gpu_<jobid>.csv`.
6. Runs `short_NPT_run.py` with `srun`, then stops the GPU monitor.

Usage:

```bash
sbatch run_NPT.sh <input_file> <target_density> [nsteps]
```

## Outputs

All files are written in the submission directory. `<rho>` is the target density with four decimals, for example `1.0000`.

| File | Content |
|---|---|
| `mace_<jobid>.out` / `mace_<jobid>.err` | Standard output (diagnostics, timings, progress) and standard error of the job |
| `md.log_<rho>` | One line every 100 steps: step number, temperature (K), density (g/cm³) and potential energy (eV) |
| `md.traj_<rho>` | ASE trajectory with a frame every 100 steps (241 frames for the default 24000 steps, including the initial one) |
| `gpu_<jobid>.csv` | GPU utilisation, memory and power sampled every 5 s |

Example of reading the results in Python:

```python
import numpy as np
from ase.io import read

# density versus time from the log file
data = np.genfromtxt("md.log_1.0000", usecols=(0,), dtype=int)   # step numbers
frames = read("md.traj_1.0000", index=":")

mass = frames[0].get_masses().sum()
rho = [mass / (0.602214076 * f.get_volume()) for f in frames]
```

Or use the `rho=` column of the log file directly, for instance with `grep` or `pandas`.

## Running several densities

To check that the equilibrium density does not depend on the starting point, submit one job per target density:

```bash
for rho in 0.90 0.95 1.00 1.05 1.10; do
    sbatch run_NPT.sh box_n4.xyz $rho
done
```

Each run writes its own `md.log_<rho>` and `md.traj_<rho>`, so the files do not overwrite each other. Plot density against time for all runs and check that they converge to the same value.

## Known limitations

- No random seed is set for the initial velocities, so repeated runs are not bit-identical.
- The default 12 ps is a short equilibration. For production-quality densities, run longer and discard the initial transient.
- The calculation uses `float32` for speed. Use `default_dtype="float64"` in the script if you need higher numerical precision, at the cost of speed.
- The Berendsen barostat does not generate a rigorous NPT ensemble (see above).

## License

MIT

## Author

C. Riccardo Civello
