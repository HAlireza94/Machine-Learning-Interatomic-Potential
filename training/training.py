import os
from pathlib import Path
import logging
import sys

import matplotlib.pyplot as plt

# For data processing
from mlip.data import BuilderMode, ExtxyzReader, GraphDatasetBuilder

# For model
from mlip.models import ForceField, Visnet, Nequip

# For loss function
from mlip.models.loss import MSELoss
from mlip.models.model_io import load_model_from_zip, save_model_to_zip
from mlip.models.params_loading import load_parameters_from_checkpoint

# For optimizer
# For training
from mlip.training import TrainingLoop, get_default_mlip_optimizer
from mlip.typing.properties import Properties



mlip_logger = logging.getLogger("mlip")
mlip_logger.setLevel(logging.INFO)

if not mlip_logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(levelname)s - %(message)s")
    )
    mlip_logger.addHandler(handler)

mlip_logger.propagate = False



data_directory = Path("pt_water_split")

train_path = data_directory / "train.extxyz"
validation_path = data_directory / "validation.extxyz"
test_path = data_directory / "test.extxyz"

readers = {
    "train": ExtxyzReader(train_path),
    "validation": ExtxyzReader(validation_path),
    "test": ExtxyzReader(test_path),
}

builder_config = GraphDatasetBuilder.Config(
    graph_cutoff_angstrom=4.0,
    batch_size=5,
)

builder = GraphDatasetBuilder(readers, builder_config, BuilderMode.TRAINING)
splits = builder.get_datasets()

train_set, validation_set, test_set = splits["train"], splits["validation"], splits["test"]

mlip_network = Nequip(
    Nequip.Config(
        node_irreps="4x0e + 4x0o + 4x1o + 4x1e + 4x2e + 4x2o",
        num_layers=2,
    ),
    builder.dataset_info,
)

force_field = ForceField.from_mlip_network(mlip_network)

optimizer = get_default_mlip_optimizer()

loss = MSELoss()


from mlip.training import TrainingLoop
from mlip.training.training_io_handler import TrainingIOHandler
from mlip.training.training_loggers import log_metrics_to_line

io_handler = TrainingIOHandler()
io_handler.attach_logger(log_metrics_to_line)

training_config = TrainingLoop.Config(num_epochs=5)
training_loop = TrainingLoop(
    train_dataset=train_set,
    validation_dataset=validation_set,
    force_field=force_field,
    loss=loss,
    optimizer=optimizer,
    config=training_config,
    io_handler=io_handler,
)

training_loop.run()

training_loop.test(test_set)

optimized_force_field = training_loop.best_model

save_model_to_zip("/home/AL/MLIP/Cutoff-4/final_model.zip", optimized_force_field)

