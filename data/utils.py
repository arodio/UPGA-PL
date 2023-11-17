import os
import time
import json
import warnings

import numpy as np
from tqdm import tqdm

from torchvision.datasets import CIFAR10, MNIST

from constants import *

from collections import Counter


def save_cfg(save_path, cfg):
    with open(save_path, "w") as f:
        json.dump(cfg, f)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum(np.exp(x))
    return ex / sum_ex


def save_data(save_dir, train_data, train_targets, test_data, test_targets):
    """save data and targets as `.npy` files

    Parameters
    ----------
    save_dir: str
        directory to save data; it will be created it it does not exist

    train_data: numpy.array

    train_targets: numpy.array

    test_data: numpy.array

    test_targets: numpy.array

    """
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "train_data.npy"), "wb") as f:
        np.save(f, train_data)

    with open(os.path.join(save_dir, "train_targets.npy"), "wb") as f:
        np.save(f, train_targets)

    with open(os.path.join(save_dir, "test_data.npy"), "wb") as f:
        np.save(f, test_data)

    with open(os.path.join(save_dir, "test_targets.npy"), "wb") as f:
        np.save(f, test_targets)


def get_dataset(dataset_name, raw_data_path):
    if dataset_name == "cifar10":
        dataset = CIFAR10(root=raw_data_path, download=True, train=True)
        test_dataset = CIFAR10(root=raw_data_path, download=True, train=False)

        dataset.data = np.concatenate((dataset.data, test_dataset.data))
        dataset.targets = np.concatenate((dataset.targets, test_dataset.targets))

    elif dataset_name == "mnist":

        dataset = MNIST(root=raw_data_path, download=True, train=True)
        test_dataset = MNIST(root=raw_data_path, download=True, train=False)

        dataset.data = np.concatenate((dataset.data, test_dataset.data))
        dataset.targets = np.concatenate((dataset.targets, test_dataset.targets))

    else:
        error_message = f"{dataset_name} is not available, possible datasets are:"
        for n in DATASETS:
            error_message += n + ",\t"

        raise NotImplementedError(error_message)

    return dataset


def iid_divide(l_, g):
    """
    https://github.com/TalwalkarLab/leaf/blob/master/data/utils/sample.py

    divide list `l` among `g` groups
    each group has either `int(len(l)/g)` or `int(len(l)/g)+1` elements
    returns a list of groups

    """
    num_elems = len(l_)
    group_size = int(len(l_) / g)
    num_big_groups = num_elems - g * group_size
    num_small_groups = g - num_big_groups
    glist = []
    for i in range(num_small_groups):
        glist.append(l_[group_size * i: group_size * (i + 1)])
    bi = group_size * num_small_groups
    group_size += 1
    for i in range(num_big_groups):
        glist.append(l_[bi + group_size * i:bi + group_size * (i + 1)])
    return glist


def split_list_by_indices(l_, indices):
    """
    divide list `l` given indices into `len(indices)` sub-lists
    sub-list `i` starts from `indices[i]` and stops at `indices[i+1]`
    returns a list of sub-lists
    """
    res = []
    current_index = 0
    for index in indices:
        res.append(l_[current_index: index])
        current_index = index

    return res


def iid_split(
        dataset,
        n_tasks,
        frac,
        rng=None
):
    """
    split classification dataset among `n_clients` in an IID fashion. The dataset is split as follows:
        1) The dataset is shuffled and partitioned among n_clients

    Parameters
    ----------
    dataset: torch.utils.Dataset
        a classification dataset;
         expected to have attributes `data` and `targets` storing `numpy.array` objects

    n_tasks: int
        number of clients

    frac: fraction of dataset to use

    rng: random number generator; default is None

    Returns
    -------
        * List[Dict]: list (size=`n_tasks`) of dictionaries, storing the data and metadata for each client

    """

    rng = (np.random.default_rng(int(time.time())) if (rng is None) else rng)

    n_samples = int(len(dataset) * frac)

    selected_indices = rng.choice(list(range(len(dataset))), n_samples, replace=False)
    rng.shuffle(selected_indices)

    tasks_indices = iid_divide(selected_indices, n_tasks)

    return tasks_indices


def by_labels_non_iid_split(dataset, n_classes, n_tasks, n_clusters, alpha, frac, rng=None):
    """
    split classification dataset among `n_tasks` in a non-IID fashion. The dataset is split as follows:
        1) classes are grouped into `n_clusters`
        2) for each cluster `c`, samples are partitioned across tasks using dirichlet distribution
    Inspired by the split in "Federated Learning with Matched Averaging"__(https://arxiv.org/abs/2002.06440)

    :param dataset:
    :type dataset: torch.utils.Dataset
    :param n_classes: number of classes present in `dataset`
    :param n_tasks: number of tasks
    :param n_clusters: number of clusters to consider; if it is `-1`, then `n_clusters = n_classes`
    :param alpha: parameter controlling the diversity among clients
    :param frac: fraction of dataset to use
    :param rng: random number generator; default is None
    :return: list (size `n_tasks`) of subgroups, each subgroup is a list of indices.
    """
    if n_clusters == -1:
        n_clusters = n_classes

    rng = (np.random.default_rng(int(time.time())) if (rng is None) else rng)

    all_labels = list(range(n_classes))
    rng.shuffle(all_labels)
    clusters_labels = iid_divide(all_labels, n_clusters)

    label2cluster = dict()  # maps label to its cluster
    for group_idx, labels in enumerate(clusters_labels):
        for label in labels:
            label2cluster[label] = group_idx

    # get subset
    n_samples = int(len(dataset) * frac)
    selected_indices = rng.choice(list(range(len(dataset))), n_samples, replace=False).tolist()

    clusters_sizes = np.zeros(n_clusters, dtype=int)
    clusters = {k: [] for k in range(n_clusters)}
    for idx in selected_indices:
        label = dataset.targets[idx]
        group_id = label2cluster[label]
        clusters_sizes[group_id] += 1
        clusters[group_id].append(idx)

    for _, cluster in clusters.items():
        rng.shuffle(cluster)

    tasks_counts = np.zeros((n_clusters, n_tasks), dtype=np.int64)  # number of samples by task from each cluster

    for cluster_id in range(n_clusters):
        weights = rng.dirichlet(alpha=alpha * np.ones(n_tasks))
        tasks_counts[cluster_id] = rng.multinomial(clusters_sizes[cluster_id], weights)

    tasks_counts = np.cumsum(tasks_counts, axis=1)

    tasks_indices = [[] for _ in range(n_tasks)]
    for cluster_id in range(n_clusters):
        cluster_split = split_list_by_indices(clusters[cluster_id], tasks_counts[cluster_id])

        for task_id, indices in enumerate(cluster_split):
            tasks_indices[task_id] += indices

    return tasks_indices


def pathological_non_iid_split(dataset, n_classes, n_tasks, n_classes_per_task, frac=1, rng=None):
    """
    split classification dataset among `n_tasks`. The dataset is split as follows:
        1) sort the data by label
        2) divide it into `n_tasks * n_classes_per_task` shards, of equal size.
        3) assign each of the `n_tasks` with `n_classes_per_task` shards
    Inspired by the split in
     "Communication-Efficient Learning of Deep Networks from Decentralized Data"__(https://arxiv.org/abs/1602.05629)

    :param dataset:
        :type dataset: torch.utils.Dataset
    :param n_classes: umber of classes present in `dataset`
    :param n_tasks: number of clients
    :param n_classes_per_task:
    :param frac: fraction of dataset to use
    :param rng:random number generator; default is None
    :return: list (size `n_clients`) of subgroups, each subgroup is a list of indices.
    """

    rng = (np.random.default_rng(int(time.time())) if (rng is None) else rng)

    # get subset
    n_samples = int(len(dataset) * frac)
    selected_indices = rng.choice(list(range(len(dataset))), n_samples, replace=False).tolist()

    label2index = {k: [] for k in range(n_classes)}
    for idx in selected_indices:
        label = dataset.targets[idx]
        label2index[label].append(idx)

    sorted_indices = []
    for label in label2index:
        sorted_indices += label2index[label]

    n_shards = n_tasks * n_classes_per_task
    shards = iid_divide(sorted_indices, n_shards)
    rng.shuffle(shards)
    tasks_shards = iid_divide(shards, n_tasks)

    tasks_indices = [[] for _ in range(n_tasks)]
    for task_id in range(n_tasks):
        for shard in tasks_shards[task_id]:
            tasks_indices[task_id] += shard

    return tasks_indices


def generate_synthetic(
        n_tasks,
        n_classes,
        n_samples,
        dimension,
        alpha,
        beta,
        iid,
        availability_parameter,
        stability_parameter,
        tasks_deterministic_split,
        tasks_proportion,
        log_normal,
        save_dir,
        rng
):
    """generate synthetic dataset

    Parameters
    ----------

    n_classes: int

    n_tasks: int

    n_samples: int

    dimension: int

    alpha: float

    beta: float

    iid: bool

    availability_parameter: float
        parameter controlling the asymptotic availability of the tasks;

    stability_parameter: float
        parameter controlling the stability of the tasks;

    tasks_deterministic_split: bool
        split tasks per task type in a deterministic or stochastic way, according to JOINT_PROBABILITY_MATRIX

    tasks_proportion: float
        if JOINT_PROBABILITY_MATRIX is None, determines the proportion of more available clients

    log_normal: bool
        if True, the number of samples among clients follows a log-normal distribution

    save_dir: str
        directory to save train_data for all tasks

    rng: random number generator; default is None

    Returns
    -------
        * List[Dict]: list (size=`n_tasks`) of dictionaries, storing the train_data and metadata for each client
    """
    rng = (np.random.default_rng(int(time.time())) if (rng is None) else rng)

    all_tasks_cfg = dict()

    tasks_types = generate_tasks_types(
        num_tasks=n_tasks,
        joint_probability=np.array(JOINT_PROBABILITY_MATRIX),
        tasks_deterministic_split=tasks_deterministic_split,
        rng=rng,
        tasks_proportion=tasks_proportion
    )

    samples_per_user = split_samples_per_user(
        n_tasks=n_tasks,
        tasks_types=tasks_types,
        n_samples=n_samples,
        rng=rng,
        log_normal=log_normal
    )

    X = [[] for _ in range(n_tasks)]
    y = [[] for _ in range(n_tasks)]

    # prior for parameters
    mean_W = rng.normal(0, alpha, n_tasks)
    mean_b = mean_W
    B = rng.normal(0, beta, n_tasks)

    mean_x = np.zeros((n_tasks, dimension))
    diagonal = np.zeros(dimension)
    for j in range(dimension):
        diagonal[j] = np.power((j + 1), -1.2)
    cov_x = np.diag(diagonal)

    for i in range(n_tasks):
        if iid == 1:
            mean_x[i] = np.ones(dimension) * B[i]  # all zeros
        else:
            mean_x[i] = rng.normal(B[i], 1, dimension)

    W_global = b_global = None
    if iid == 1:
        W_global = rng.normal(0, 1, (dimension, n_classes))
        b_global = rng.normal(0, 1, n_classes)

    for i in range(n_tasks):

        if iid == 1:
            assert W_global is not None and b_global is not None
            W = W_global
            b = b_global
        else:
            W = rng.normal(mean_W[i], 1, (dimension, n_classes))
            b = rng.normal(mean_b[i], 1, n_classes)

        xx = rng.multivariate_normal(mean_x[i], cov_x, samples_per_user[i])
        yy = np.zeros(samples_per_user[i])

        for j in range(samples_per_user[i]):
            tmp = np.dot(xx[j], W) + b
            yy[j] = np.argmax(softmax(tmp))

        X[i].extend(xx.tolist())
        y[i].extend(yy.tolist())

    for i in range(n_tasks):
        task_dir = \
            os.path.join(os.getcwd(), save_dir, f"task_{str(i)}")

        combined = list(zip(X[i], y[i]))
        rng.shuffle(combined)
        X[i][:], y[i][:] = zip(*combined)
        num_samples = len(X[i])
        train_len = int(0.8 * num_samples)
        train_data = X[i][:train_len]
        test_data = X[i][train_len:]
        train_targets = y[i][:train_len]
        test_targets = y[i][train_len:]

        save_data(
            save_dir=task_dir,
            train_data=train_data,
            train_targets=train_targets,
            test_data=test_data,
            test_targets=test_targets
        )

        availability_id = tasks_types[i][0]
        stability_id = tasks_types[i][1]

        availability = \
            compute_availability(
                availability_type=AVAILABILITY_TYPES[availability_id],
                availability_parameter=availability_parameter
            )

        stability = 0.0

        print(f"Client {i} has availability {availability:0.1f} and labels {Counter(train_targets)}")

        all_tasks_cfg[int(i)] = {
            "cluster_id": 0,
            "task_dir": task_dir,
            "availability_type": AVAILABILITY_TYPES[availability_id],
            "availability": availability,
            "stability_type": STABILITY_TYPES[stability_id],
            "stability": stability
        }

    return all_tasks_cfg


def generate_data(
        dataset,
        frac,
        split_type,
        n_classes,
        n_train_samples,
        n_tasks,
        tasks_proportion,
        tasks_deterministic_split,
        n_components,
        alpha,
        n_shards,
        availability_parameter,
        stability_parameter,
        save_dir,
        rng=None
):
    if split_type == "pathological_split":
        print(f"==> Pathological split (non-IID)")
        tasks_indices = \
            pathological_non_iid_split(
                dataset=dataset,
                n_classes=n_classes,
                n_tasks=n_tasks,
                n_classes_per_task=n_shards,
                frac=frac,
                rng=rng
            )
    elif split_type == "by_labels_split":
        print(f"==> Data are split by labels (non-IID)")
        tasks_indices = \
            by_labels_non_iid_split(
                dataset=dataset,
                n_classes=n_classes,
                n_tasks=n_tasks,
                n_clusters=n_components,
                alpha=alpha,
                frac=frac,
                rng=rng
            )
    else:
        print("==> Data are split IID")
        tasks_indices = \
            iid_split(
                dataset=dataset,
                n_tasks=n_tasks,
                frac=frac,
                rng=rng
            )

    all_tasks_cfg = dict()

    tasks_types = generate_tasks_types(
        num_tasks=n_tasks,
        joint_probability=np.array(JOINT_PROBABILITY_MATRIX),
        tasks_deterministic_split=tasks_deterministic_split,
        rng=rng,
        tasks_proportion=tasks_proportion
    )

    for task_id in tqdm(range(n_tasks), total=n_tasks, desc="Tasks.."):
        task_indices = np.array(tasks_indices[task_id])
        train_indices = task_indices[task_indices < n_train_samples]
        test_indices = task_indices[task_indices >= n_train_samples]

        train_data, train_targets = dataset.data[train_indices], dataset.targets[train_indices]
        test_data, test_targets = dataset.data[test_indices], dataset.targets[test_indices]

#        train_data, train_targets = keep_leading_target(train_data, train_targets)
#        test_data, test_targets = keep_leading_target(test_data, test_targets)

        task_dir = os.path.join(os.getcwd(), save_dir, f"task_{task_id}")

        save_data(
            save_dir=task_dir,
            train_data=train_data,
            train_targets=train_targets,
            test_data=test_data,
            test_targets=test_targets
        )

        availability_id = tasks_types[task_id][0]
        stability_id = tasks_types[task_id][1]

        availability = \
            compute_availability(
                availability_type=AVAILABILITY_TYPES[availability_id],
                availability_parameter=availability_parameter
            )

        stability = 0.0

        print(f"Client {task_id} has availability {availability:0.1f} and labels {Counter(train_targets)}")

        all_tasks_cfg[str(task_id)] = {
            "task_dir": task_dir,
            "availability_type": AVAILABILITY_TYPES[availability_id],
            "availability": availability,
            "stability_type": STABILITY_TYPES[stability_id],
            "stability": stability
        }

    return all_tasks_cfg


def generate_tasks_types(
        num_tasks,
        joint_probability,
        tasks_deterministic_split,
        rng,
        tasks_proportion=None
):
    """Generate tasks_types types

    The tasks_types has an `availability` type, and a `stability` type;

    The `availability` and `stability` types are sampled according to
        * `JOINT_PROBABILITY_MATRIX`, if not None
        * 'tasks_proportion', otherwise

    The result is given as a `numpy.array` of shape `(num_clients, 2)`. The columns
    correspond to the availability type, and stability type, respectively.

    Parameters
    ----------
    num_tasks: `int`
        number of tasks_types

    joint_probability: 2-D `numpy.array`
        every entry represents the probability of an arrival process and a capacity; should sum-up to 1

    tasks_deterministic_split: bool
        split tasks per task type in a deterministic or stochastic way, according to JOINT_PROBABILITY_MATRIX

    rng: `numpy.random._generator.Generator`

    tasks_proportion: float
        if JOINT_PROBABILITY_MATRIX is None, determines the proportion of more available clients

    Returns
    -------
        * `numpy.array` of shape `(num_clients, 2)`

    """
    if np.isnan(joint_probability).any():
        print(f"==> The proportion of more available clients is {tasks_proportion}")
        joint_probability = np.array(
            [
                [0., tasks_proportion, 0.],
                [1 - tasks_proportion, 0., 0.]
            ],
            dtype=np.float64
        )

    assert np.abs(joint_probability.sum() - 1) < ERROR, "`joint_probability` should sum-up to 1!"

    tasks_indices = rng.permutation(num_tasks)
    count_per_cluster = split_tasks_per_types(num_tasks, joint_probability, tasks_deterministic_split, rng)
    indices_per_cluster = np.split(tasks_indices, np.cumsum(count_per_cluster[:-1]))
    indices_per_cluster = np.array(indices_per_cluster, dtype=object).reshape(joint_probability.shape)

    tasks_types = np.zeros((num_tasks, 2), dtype=np.int8)

    for availability_type_idx in range(joint_probability.shape[0]):
        indices = np.concatenate(indices_per_cluster[availability_type_idx])
        tasks_types[indices, 0] = availability_type_idx

    for stability_idx in range(joint_probability.shape[1]):
        indices = np.concatenate(indices_per_cluster[:, stability_idx])
        tasks_types[indices, 1] = stability_idx

    return tasks_types


def split_tasks_per_types(num_tasks, joint_probability, tasks_deterministic_split, rng):
    """Split tasks into tasks types according to `JOINT_PROBABILITY_MATRIX`.

    If the split is deterministic, the number of tasks per task type is computed proportionally to joint_probability.

    Otherwise, the number of tasks per task type is computed from a multinomial with joint_probability.

    The result is given as a `numpy.array` of shape `(joint_probability.flatten(), )`. The elements
    correspond to the number of tasks for each task type.

    Parameters
    ----------
    num_tasks: `int`
        number of tasks_types

    joint_probability: 2-D `numpy.array`
        every entry represents the probability of an arrival process and a capacity; should sum-up to 1

    tasks_deterministic_split: bool
        split tasks per task type in a deterministic or stochastic way, according to JOINT_PROBABILITY_MATRIX

    rng: `numpy.random._generator.Generator`

    Returns
    -------
        * `numpy.array` of shape `(joint_probability.flatten(), )`

    """
    if tasks_deterministic_split:
        count_per_task_type = (num_tasks * joint_probability.flatten()).astype(int)
        remaining_clients = num_tasks - count_per_task_type.sum()
        for task_type in range(count_per_task_type.shape[0]):
            if (remaining_clients > 0) and (count_per_task_type[task_type] != 0):
                count_per_task_type[task_type] += 1
                remaining_clients -= 1
        print(f"==> Deterministic split: {count_per_task_type}")
        return count_per_task_type
    else:
        return rng.multinomial(num_tasks, joint_probability.flatten())


def split_samples_per_user(n_tasks, tasks_types, n_samples, rng, log_normal=False, data_proportion=1):
    """split n_samples among n_tasks

    Parameters
    ----------
    n_tasks
    tasks_types
    n_samples
    log_normal
    data_proportion
    rng

    Returns
    -------
        * List[int]: number of samples per user
    """
    if log_normal:

        print("==> Generating samples_per_user from a log-normal distribution")
        samples_per_user = rng.lognormal(4, 2, n_tasks).astype(int) + 50

    else:

        if data_proportion == 1:
            print("==> Generating samples_per_user deterministically, all clients receive same number of samples")
        else:
            print("==> Generating samples_per_user deterministically, more available clients receive more samples")

        availability_ids = [tasks_types[i][0] for i in range(n_tasks)]
        n_unavailable_users = sum(availability_ids)
        samples_per_user = \
            [n_samples / (data_proportion * n_tasks + (1 - data_proportion) * n_unavailable_users)] * n_tasks
        for client_id in range(n_tasks):
            if availability_ids[client_id] == 0:
                samples_per_user[client_id] *= data_proportion

    return [int(i) for i in samples_per_user]


def compute_availability(availability_type, availability_parameter):
    """ compute stability value for given stability type and parameter

    Parameters
    ----------
    availability_type: str

    availability_parameter: float

    Returns
    -------
        * float:

    """
    if not (-0.5 + ERROR <= availability_parameter <= 0.5 - ERROR):
        warnings.warn("availability_parameter is automatically clipped to the interval (-1/2, 1/2)")
        availability_parameter = np.clip(availability_parameter, a_min=-0.5 + ERROR, a_max=0.5 - ERROR)

    if availability_type == "available":
        availability = 1 / 2 + availability_parameter

    elif availability_type == "unavailable":
        availability = 1 / 2 - availability_parameter

    else:
        error_message = ""
        raise NotImplementedError(error_message)

    return availability


def keep_leading_target(data, targets):
    """

    Parameters
    ----------
    data
    targets
    """

    data = np.array(data)
    targets = np.array(targets)

    leading_target = max(targets, key=list(targets).count)

    for target in set(targets):
        if target != leading_target:
            target2delete_ids = np.where(targets == target)[0]
            data = np.delete(data, target2delete_ids, axis=0)
            targets = np.delete(targets, target2delete_ids)

    return data, targets


def keep_client_target(client_id, data, targets):
    """

    Parameters
    ----------
    client_id
    data
    targets
    """

    data = np.array(data)
    targets = np.array(targets)

    target_to_keep = client_id

    for target in set(targets):
        if target != target_to_keep:
            target2delete_ids = np.where(targets == target)[0]
            data = np.delete(data, target2delete_ids, axis=0)
            targets = np.delete(targets, target2delete_ids)

    return data, targets
