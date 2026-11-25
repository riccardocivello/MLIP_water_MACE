
import sys
import numpy as np
import torch
import os

from ase.io import read
from ase.io.trajectory import Trajectory
from ase import units

from ase.md.nptberendsen import NPTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from mace.calculators import mace_off

print("cuda available:", torch.cuda.is_available())
print("torch:", torch.__version__, "cuda build:", torch.version.cuda)
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("compiled arch list:", torch.cuda.get_arch_list())

# --------------------------------------------------
# Input arguments
# --------------------------------------------------

if len(sys.argv) < 3:
    raise SystemExit(
        "Usage: python short_NPT_run.py input_file starting_density [nsteps]"
    )

fname = sys.argv[1]
rho0 = float(sys.argv[2])
nsteps = int(sys.argv[3]) if len(sys.argv) > 3 else 24000

if rho0 <= 0:
    raise ValueError("Density must be positive.")

# --------------------------------------------------
# GPU configuration
# --------------------------------------------------

torch.set_num_threads(4)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is not available!")

print("GPU:", torch.cuda.get_device_name(0), flush=True)

# --------------------------------------------------
# Read and rescale simulation box
# --------------------------------------------------

atoms = read(fname)

if not np.all(atoms.pbc):
    raise ValueError("Input structure must have periodic boundary conditions.")

volume = atoms.get_volume()

if volume <= 0:
    raise ValueError("Invalid simulation cell volume.")

# Density in g/cm3:
# atomic masses are in amu, volume in Angstrom^3
mass = atoms.get_masses().sum()
rho_initial = mass / (0.602214076 * volume)

# Rescale to target density
f = (rho_initial / rho0) ** (1.0 / 3.0)

atoms.set_cell(atoms.cell * f, scale_atoms=True)

print(f"Initial density: {rho_initial:.4f} g/cm3")
print(f"Target density:  {rho0:.4f} g/cm3")
print(f"Scaling factor:  {f:.6f}")
print("Number of atoms:", len(atoms), flush=True)

# --------------------------------------------------
# MACE calculator
# --------------------------------------------------
import time

t0 = time.perf_counter()

MODEL_PATH = os.environ.get("MACE_MODEL_PATH", "small")

atoms.calc = mace_off(
    model=MODEL_PATH,
    device="cuda",
    default_dtype="float32"
)

torch.cuda.synchronize()
print(f"Initialisation: {time.perf_counter()-t0:.2f} s")

t1 = time.perf_counter()
# Trigger first calculation
energy = atoms.get_potential_energy()
torch.cuda.synchronize()

print(f"First energy evaluation: {time.perf_counter()-t1:.2f} s")

t2 = time.perf_counter()

# --------------------------------------------------
# Molecular dynamics
# --------------------------------------------------

MaxwellBoltzmannDistribution(atoms, temperature_K=300)

dyn = NPTBerendsen(
    atoms,
    timestep=0.5 * units.fs,
    temperature_K=300,
    taut=100 * units.fs,
    pressure_au=1.01325 * units.bar,
    taup=1000 * units.fs,
    compressibility_au=4.5e-5 / units.bar
)

print(f"Initializing dynamics: {time.perf_counter()-t2:.2f} s")

t3 = time.perf_counter()

# --------------------------------------------------
# Logging and trajectory
# --------------------------------------------------

def log():
    rho = atoms.get_masses().sum() / (
        0.602214076 * atoms.get_volume()
    )

    line = (
        f"{dyn.nsteps:6d} "
        f"T={atoms.get_temperature():7.1f} K "
        f"rho={rho:6.3f} g/cm3 "
        f"Epot={atoms.get_potential_energy():10.3f} eV"
    )

    with open(f"md.log_{rho0:.4f}", "w") as logfile:
        logfile.write(line + "\n")

trajectory = Trajectory(f"md.traj_{rho0:.4f}", "w", atoms)

try:
    # Save initial configuration
    trajectory.write()

    # Attach logging and trajectory output
    dyn.attach(log, interval=100)
    dyn.attach(trajectory.write, interval=100)

    print(f"Starting MD: {nsteps} steps", flush=True)

    dyn.run(nsteps)

finally:
    trajectory.close()

print("Simulation finished.", flush=True)

torch.cuda.synchronize()
print(f"MD runtime: {time.perf_counter()-t3:.2f} s")