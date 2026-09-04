import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

os.environ["XLA_FLAGS"] = (
    "--xla_cpu_multi_thread_eigen=true "
    "intra_op_parallelism_threads=20"
)


# ONLY AFTER the environment variables:
import jax

print(jax.devices())


import numpy as np

from mlip.data import BuilderMode, ExtxyzReader, GraphDatasetBuilder
from mlip.models import ForceField, Visnet, Nequip
from mlip.models.loss import MSELoss
from mlip.models.model_io import load_model_from_zip, save_model_to_zip
from mlip.models.params_loading import load_parameters_from_checkpoint

from mlip.training import TrainingLoop, get_default_mlip_optimizer
from mlip.typing.properties import Properties
from mlip.simulation.jax_md import JaxMDSimulationEngine
from ase.io import read as ase_read
from pathlib import Path
from mlip.data import ExtxyzReader

import random

from ase.io import read, write
from ase.visualize import view

from mlip.simulation.jax_md import JaxMDSimulationEngine
from mlip.simulation.enums import MDIntegrator

from mlip.simulation.ase import ASESimulationEngine
from ase.constraints import FixAtoms

import logging
import sys
mlip_logger = logging.getLogger("mlip")
mlip_logger.setLevel(logging.INFO)

if not mlip_logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(levelname)s - %(message)s")
    )
    mlip_logger.addHandler(handler)

mlip_logger.propagate = False


# Loading ForceField 

force_field = load_model_from_zip(
    Nequip, "/home/AL/MLIP/ASE/final_model_4A.zip", required_properties=Properties(stress=True)
)


force_field_md = force_field.replace_required_properties(
    Properties(
        energy=True,
        forces=True,
        stress=False,
    )
)

print("Dataset info:", force_field.dataset_info)




input_file = "Pt_50water_1x1x1_minimized.extxyz"

atoms = read(input_file)

# Remove any calculator that may have been stored/read
atoms.calc = None


print("\n========================================")
print("1 x 1 x 1 MINIMIZED SYSTEM")
print("========================================")

print("Number of atoms:", len(atoms))
print("Formula:", atoms.get_chemical_formula())

print("\nCell:")
print(atoms.cell)

print("\nCell lengths:")
print(atoms.cell.lengths())

print("\nPBC:")
print(atoms.pbc)


# Count species
symbols = np.array(
    atoms.get_chemical_symbols()
)

n_pt_1x1 = np.sum(symbols == "Pt")
n_o_1x1 = np.sum(symbols == "O")
n_h_1x1 = np.sum(symbols == "H")

print("\nComposition:")
print("Pt:", n_pt_1x1)
print("O :", n_o_1x1)
print("H :", n_h_1x1)

# Assuming all O/H belong to water
print("Water molecules:", n_o_1x1)





# ============================================================
# Freeze ALL Pt atoms
# ============================================================

pt_indices = [
    atom.index
    for atom in atoms
    if atom.symbol == "Pt"
]

atoms.set_constraint(
    FixAtoms(indices=pt_indices)
)

print("\nConstraints:")
print(atoms.constraints)

print("Total atoms:", len(atoms))
print("Fixed Pt atoms:", len(pt_indices))
print("Mobile atoms:", len(atoms) - len(pt_indices))






config = ASESimulationEngine.Config(

    # MD length
    num_steps=500_000,
    snapshot_interval=1000,
    log_interval=1000,

    # Ensemble
    md_integrator=MDIntegrator.NVT_LANGEVIN,
    temperature_kelvin=300.0,
    timestep_fs=1.0,

    # Charge handling
    set_none_charge_to_zero=True,
)



md_engine = ASESimulationEngine(
    atoms,
    force_field_md,
    config,
)

md_engine.run()

##########################################################
# Saving Coordinates, Energy, and Force for Post Analysis#
##########################################################



md_state = md_engine.state



######################### Coordinates ####################

positions = np.asarray(md_state.positions)
print("Trajectory shape:", positions.shape)
print("Number of snapshots:", len(positions))
print("Number of atoms:", positions.shape[1])

frames = []

for i, xyz in enumerate(positions):

    # atoms is the structure used to initialize MD
    # Therefore it already has:
    #   - atomic symbols
    #   - cell
    #   - PBC
    frame = atoms.copy()

    # Put MD coordinates into the ASE structure
    frame.set_positions(xyz)

    # If the simulation state contains the cell,
    # use the corresponding MD cell
    if md_state.cell is not None:

        cells = np.asarray(md_state.cell)

        if cells.ndim == 3:
            frame.set_cell(cells[i])

        elif cells.ndim == 2:
            frame.set_cell(cells)

    frame.set_pbc(atoms.pbc)

    # Put atoms inside primary periodic box
    frame.wrap()

    frames.append(frame)


print("ASE frames created:", len(frames))


write(
    "md_trajectory.extxyz",
    frames,
    format="extxyz",
)

print("Saved: md_trajectory.extxyz")


######################### Forces and Energies ####################
import numpy as np

md_state = md_engine.state

potential_energy = np.asarray(md_state.potential_energy)
kinetic_energy   = np.asarray(md_state.kinetic_energy)
forces           = np.asarray(md_state.forces)
temperature      = np.asarray(md_state.temperature)

volume = atoms.get_volume()
print("Constant NVT volume:", volume, "A^3")


print(
    f"{'Step':>8s} "
    f"{'Time(ps)':>10s} "
    f"{'Temp(K)':>12s} "
    f"{'PE(eV)':>15s} "
    f"{'KE(eV)':>15s} "
    f"{'Etot(eV)':>15s} "
    f"{'Fmax(eV/A)':>15s} "
    f"{'Volume(A^3)':>15s}"
)


SNAPSHOT_INTERVAL = 1000
TIMESTEP_FS = 1.0

for i in range(len(potential_energy)):

    # Since snapshot_interval = 1
    step = i * SNAPSHOT_INTERVAL

    # timestep = 1 fs
    time_ps = step * TIMESTEP_FS / 1000.0

    pe = potential_energy[i]
    ke = kinetic_energy[i]
    etot = pe + ke

    # Force magnitude on each atom
    force_magnitudes = np.linalg.norm(
        forces[i],
        axis=1
    )

    fmax = np.max(force_magnitudes)

    # Note : activate this volume below only if NPT is used.

    # volume = abs(
    #     np.linalg.det(cells[step])
    # )

    print(
        f"{step:8d} "
        f"{time_ps:10.4f} "
        f"{temperature[i]:12.3f} "
        f"{pe:15.6f} "
        f"{ke:15.6f} "
        f"{etot:15.6f} "
        f"{fmax:15.6f} "
        f"{volume:15.3f}"
    )


with open("thermo.dat", "w") as f:

    f.write(
        "# Step Temp_K PE_eV KE_eV Etot_eV "
        "Fmax_eV_A Volume_A3\n"
    )

    for step in range(len(potential_energy)):

        pe = potential_energy[step]
        ke = kinetic_energy[step]
        etot = pe + ke

        force_magnitudes = np.linalg.norm(
            forces[step],
            axis=1
        )

        fmax = np.max(force_magnitudes)

        # Note : activate this volume below only if NPT is used.
        # volume = abs(
        #     np.linalg.det(cells[step])
        # )

        f.write(
            f"{step:8d} "
            f"{temperature[step]:15.6f} "
            f"{pe:20.10f} "
            f"{ke:20.10f} "
            f"{etot:20.10f} "
            f"{fmax:15.8f} "
            f"{volume:15.6f}\n"
        )
