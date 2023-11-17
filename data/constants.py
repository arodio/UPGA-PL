import numpy as np


ERROR = 1e-8

DATASETS = {
    "synthetic",
    "mnist",
    "cifar10"
}

N_CLASSES = {
    "mnist": 10,
    "cifar10": 10
}

N_TRAIN_SAMPLES = {
    "mnist": 60_000,
    "cifar10": 50_000
}

AVAILABILITY_TYPES = ["available", "unavailable"]
STABILITY_TYPES = ["stable", "shifting", "unstable"]

# each entry represents the probability of observing a client type (i.e., availability and stability type)
# rows indicate availability types and columns indicate stability types
JOINT_PROBABILITY_MATRIX = np.array(
    [
        [0., 0.5, 0.],
        [0.0, 0.5, 0.]
    ],
    dtype=np.float64
)
