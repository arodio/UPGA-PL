from models import *
from trainer import *

from datasets.cifar10 import *
from datasets.mnist import *
from datasets.tabular import *

from client import *

from aggregator import *

from activity_simulator import *
from clients_sampler import *

from .optim import *
from .metrics import *
from .constants import *

from torch.utils.data import DataLoader

from torchvision.transforms import Compose, ToTensor, Normalize

from torch.utils.tensorboard import SummaryWriter

import warnings


def experiment_not_implemented_message(experiment_name):
    error = f"{experiment_name} is not available! " \
            f"Possible are: 'cifar10', 'mnist', 'synthetic' and 'synthetic_leaf'."

    return error


def get_model(experiment_name, device):
    """
    create model

    Parameters
    ----------
    experiment_name: str

    device: str
        either cpu or cuda


    Returns
    -------
        model (torch.nn.Module)

    """

    if experiment_name == "synthetic_leaf":
        model = LinearLayer(input_dim=60, output_dim=10, bias=True)
    elif experiment_name == "synthetic_clustered":
        model = LinearLayer(input_dim=10, output_dim=1, bias=True)
    elif experiment_name == "mnist":
        # model = LinearLayer(input_dim=784, output_dim=10, bias=True)
        model = MnistCNN(num_classes=10)
    elif experiment_name == "emnist" or experiment_name == "femnist":
        model = FemnistCNN(num_classes=62)
    elif experiment_name == "cifar10":
        # model = CIFAR10CNN(num_classes=10)
        # TODO: choose model
        model = get_mobilenet(n_classes=10)
    elif experiment_name == "cifar100":
        model = get_mobilenet(n_classes=100)
    else:
        raise NotImplementedError(
            experiment_not_implemented_message(experiment_name=experiment_name)
        )

    model = model.to(device)

    return model


def get_trainer(experiment_name, device, optimizer_name, lr, seed):
    """
    constructs trainer for an experiment for a given seed

    Parameters
    ----------
    experiment_name: str
        name of the experiment to be used;
        possible are {"mnist"}

    device: str
        used device; possible `cpu` and `cuda`

    optimizer_name: str

    lr: float
        learning rate

    seed: int

    Returns
    -------
        Trainer

    """
    torch.manual_seed(seed)

    if experiment_name == "synthetic_leaf":
        criterion = nn.CrossEntropyLoss(reduction="mean").to(device)
        metric = accuracy
        is_binary_classification = False
    elif experiment_name == "synthetic_clustered":
        criterion = nn.BCEWithLogitsLoss(reduction='mean').to(device)
        metric = binary_accuracy
        is_binary_classification = True
    elif experiment_name == "cifar10" or experiment_name == "cifar100":
        criterion = nn.CrossEntropyLoss(reduction="mean").to(device)
        metric = accuracy
        is_binary_classification = False
    elif experiment_name == "mnist" or experiment_name == "emnist" or experiment_name == "femnist":
        criterion = nn.CrossEntropyLoss(reduction="mean").to(device)
        metric = accuracy
        is_binary_classification = False
    else:
        raise NotImplementedError(
            experiment_not_implemented_message(experiment_name=experiment_name)
        )

    model = \
        get_model(experiment_name=experiment_name, device=device)

    optimizer = \
        get_optimizer(
            optimizer_name=optimizer_name,
            model=model,
            lr=lr,
        )

    return Trainer(
        model=model,
        criterion=criterion,
        metric=metric,
        device=device,
        optimizer=optimizer,
        is_binary_classification=is_binary_classification
    )


def get_loader(experiment_name, client_data_path, batch_size, train, device):
    """

    Parameters
    ----------
    experiment_name: str

    client_data_path: str

    batch_size: int

    train: bool

    device: str
        used device; possible `cpu` and `cuda`

    Returns
    -------
        * torch.utils.data.DataLoader

    """

    if device == "cpu":
        pin_memory = False
    elif device == "cuda":
        pin_memory = True
    else:
        raise NotImplementedError(f"Device is neither 'cpu' nor 'cuda'")

    if experiment_name == "mnist":
        transform = Compose([
            ToTensor(),
            Normalize((0.1307,), (0.3081,))
        ])

        dataset = MNIST(root=client_data_path, train=train, transform=transform)

    elif experiment_name == "cifar10":
        transform = Compose([
            ToTensor(),
            Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)
            )
        ])

        dataset = CIFAR10(root=client_data_path, train=train, transform=transform)

    elif experiment_name == "synthetic_leaf" or experiment_name == "synthetic_clustered":
        if train:
            data = np.load(os.path.join(client_data_path, "train_data.npy"))
            targets = np.load(os.path.join(client_data_path, "train_targets.npy"))
        else:
            data = np.load(os.path.join(client_data_path, "test_data.npy"))
            targets = np.load(os.path.join(client_data_path, "test_targets.npy"))

        dataset = Tabular(data=data, targets=targets)

    else:
        raise NotImplementedError(
            experiment_not_implemented_message(experiment_name=experiment_name)
        )

    return DataLoader(dataset, batch_size=batch_size, shuffle=train, pin_memory=pin_memory)


def init_client(args, client_id, data_dir, logs_dir, verbose, device):
    """initialize one client


    Parameters
    ----------
    args:

    client_id: int

    data_dir: str

    logs_dir: str

    verbose: int

    device: str

    Returns
    -------
        * Client

    """
    train_loader = get_loader(
        experiment_name=args.experiment,
        client_data_path=data_dir,
        batch_size=args.train_bz,
        train=True,
        device=device
    )

    # TODO: add val loader
    val_loader = get_loader(
        experiment_name=args.experiment,
        client_data_path=data_dir,
        batch_size=args.test_bz,
        train=False,
        device=device
    )

    test_loader = get_loader(
        experiment_name=args.experiment,
        client_data_path=data_dir,
        batch_size=args.test_bz,
        train=False,
        device=device
    )

    trainer = \
        get_trainer(
            experiment_name=args.experiment,
            device=args.device,
            optimizer_name=args.local_optimizer,
            lr=args.local_lr,
            seed=args.seed
        )

    if verbose > 0:
        os.makedirs(logs_dir, exist_ok=True)
        logger = SummaryWriter(logs_dir)
    else:
        logger = None

    client = Client(
        client_id=client_id,
        trainer=trainer,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        local_steps=args.local_steps,
        verbose=verbose,
        logger=logger
    )

    return client


def get_aggregator(aggregator_type, clients_dict, clients_weights_dict, global_trainer, logger, verbose, seed):
    """
    Parameters
    ----------
    aggregator_type: str
        possible are {"gradients", "models"}

    clients_dict: Dict[int: Client]

    clients_weights_dict: Dict[int: Client]

    global_trainer: Trainer

    logger: torch.utils.tensorboard.SummaryWriter

    verbose: int

    seed: int


    Returns
    -------
        * Aggregator
    """
    if aggregator_type == "gradients":
        return GradientsAggregator(
            clients_dict=clients_dict,
            clients_weights_dict=clients_weights_dict,
            global_trainer=global_trainer,
            logger=logger,
            verbose=verbose,
            seed=seed,
        )
    elif aggregator_type == "params":
        return ParamsAggregator(
            clients_dict=clients_dict,
            clients_weights_dict=clients_weights_dict,
            global_trainer=global_trainer,
            logger=logger,
            verbose=verbose,
            seed=seed,
        )
    else:
        error_message = f"{aggregator_type} is not a possible aggregator type, possible are: "
        for type_ in AGGREGATOR_TYPES:
            error_message += f" {type_}"


def get_activity_simulator(all_clients_cfg, rng):
    clients_ids = []
    availability_types = []
    availabilities = []
    stability_types = []
    stabilities = []

    for client_id in all_clients_cfg.keys():
        clients_ids.append(client_id)
        availability_types.append(all_clients_cfg[client_id]["availability_type"])
        availabilities.append(all_clients_cfg[client_id]["availability"])
        stability_types.append(all_clients_cfg[client_id]["stability_type"])
        stabilities.append(all_clients_cfg[client_id]["stability"])

    clients_ids = np.array(clients_ids, dtype=np.int32)
    availabilities = np.array(availabilities, dtype=np.float32)
    stabilities = np.array(stabilities, dtype=np.float32)

    activity_simulator = \
        ActivitySimulator(clients_ids, availability_types, availabilities, stability_types, stabilities, rng=rng)

    return activity_simulator


def get_clients_sampler(
        sampler_type,
        activity_simulator,
        clients_weights_dict,
        rng
):
    if sampler_type == "unbiased":
        return UnbiasedClientsSampler(
            activity_simulator=activity_simulator,
            clients_weights_dict=clients_weights_dict,
            rng=rng
        )
    elif sampler_type == "biased":
        return BiasedClientsSampler(
            activity_simulator=activity_simulator,
            clients_weights_dict=clients_weights_dict,
            rng=rng
        )
    elif sampler_type == "ideal":
        return IdealClientsSampler(
            activity_simulator=activity_simulator,
            clients_weights_dict=clients_weights_dict,
            rng=rng
        )
    elif sampler_type == "oracle":
        return OracleClientsSampler(
            activity_simulator=activity_simulator,
            clients_weights_dict=clients_weights_dict,
            rng=rng
        )
    else:
        error_message = f"{sampler_type} is not an available sampler type, possible are:"

        for t in SAMPLER_TYPES:
            error_message += f"{t},"

        raise NotImplementedError(error_message)


def get_clients_weights(objective_type, n_samples_per_client):
    """compute the weights to be associated to every client

    If objective_type is "average", clients receive the same weight

    If objective_type is "weighted", clients receive weight proportional to the number of samples


    Parameters
    ----------
    objective_type: str
        type of the objective function; possible are: {"average", "weighted"}

    n_samples_per_client: Dict[int: float]


    Returns
    -------
        * Dict[int: float]

    """
    weights_dict = dict()
    n_clients = len(n_samples_per_client)

    if objective_type == "average":
        for client_id in n_samples_per_client:
            weights_dict[int(client_id)] = 1 / n_clients

    elif objective_type == "weighted":
        total_num_samples = 0

        for client_id in n_samples_per_client:
            total_num_samples += n_samples_per_client[client_id]

        for client_id in n_samples_per_client:
            weights_dict[int(client_id)] = n_samples_per_client[client_id] / total_num_samples

    else:
        raise NotImplementedError(
            f"{objective_type} is not an available objective type. Possible are 'average' and 'weighted'"
        )

    return weights_dict
