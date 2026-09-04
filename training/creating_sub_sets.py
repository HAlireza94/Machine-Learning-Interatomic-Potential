from pathlib import Path

import time as t
import numpy as np
from ase.io import iread, write
from ase.io.trajectory import Trajectory


# ============================================================
# Settings
# ============================================================
input_file = Path(
    "~/MLIP/dataset.traj"
).expanduser()

output_directory = Path("pt_water_split")
output_directory.mkdir(exist_ok=True)

train_file = output_directory / "train.extxyz"
validation_file = output_directory / "validation.extxyz"
test_file = output_directory / "test.extxyz"

train_fraction = 0.80
validation_fraction = 0.10
random_seed = np.random.seed(int(t.time() * np.random.rand()))

start = 40000


# ============================================================
# Determine the number of configurations
# ============================================================
trajectory = Trajectory(input_file, mode="r")
number_of_configurations = len(trajectory)
trajectory.close()

print("Total configurations:", number_of_configurations)


# ============================================================
# Create reproducible random indices
# ============================================================
rng = np.random.default_rng(random_seed)

shuffled_indices = rng.permutation(number_of_configurations)

number_train = int(train_fraction * number_of_configurations)
number_validation = int(
    validation_fraction * number_of_configurations
)
number_test = (
    number_of_configurations
    - number_train
    - number_validation
)

train_indices = set(
    shuffled_indices[:number_train]
)

validation_indices = set(
    shuffled_indices[
        number_train:number_train + number_validation
    ]
)

test_indices = set(
    shuffled_indices[
        number_train + number_validation:
    ]
)

print("Training configurations:", len(train_indices))
print("Validation configurations:", len(validation_indices))
print("Test configurations:", len(test_indices))


# ============================================================
# Remove old split files, if present
# ============================================================
for output_file in [
    train_file,
    validation_file,
    test_file,
]:
    if output_file.exists():
        output_file.unlink()


# ============================================================
# Write configurations without loading everything into memory
# ============================================================
train_count = 0
validation_count = 0
test_count = 0

with (
    open(train_file, "w") as train_output,
    open(validation_file, "w") as validation_output,
    open(test_file, "w") as test_output,
):

    for index, atoms in enumerate(
        iread(input_file, index=str(start)+":")
    ):

        if index in train_indices:
            write(
                train_output,
                atoms,
                format="extxyz",
            )
            train_count += 1

        elif index in validation_indices:
            write(
                validation_output,
                atoms,
                format="extxyz",
            )
            validation_count += 1

        elif index in test_indices:
            write(
                test_output,
                atoms,
                format="extxyz",
            )
            test_count += 1

        if (index + 1) % 1000 == 0:
            print(
                f"Processed {index + 1} / "
                f"{number_of_configurations}"
            )


# ============================================================
# Final report
# ============================================================
print("\nFinished splitting dataset")
print("--------------------------------")
print("Training:", train_count, train_file)
print("Validation:", validation_count, validation_file)
print("Test:", test_count, test_file)
