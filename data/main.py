"""Downloads a dataset and generates configuration file for federated simulation

Split a classification dataset, e.g., `MNIST` or `CIFAR-10`, among `n_clients`.

Two splitting strategies are available: `incongruent_split` and `iid_split`

### Incongruent split

If `incongruent_split` is selected, the dataset is split among `n_clients`,
 each of which belonging to one of two groups, as follows:

    1) the dataset is randomly shuffled and partitioned among `n_clients`
    2) clients_dict are randomly partitioned into `n_clusters`
    3) the data of clients_dict from the second group is  modified
       by  randomly  swapping out `k` pairs of labels

    Similar to
        "Clustered Federated Learning: Model-Agnostic Distributed
        Multi-Task Optimization under Privacy Constraints"
        __(https://arxiv.org/abs/1910.01991)

If  'iid_split'  is selected, the  dataset is split in an IID fashion.

Default usage is ''iid_split'

"""
import argparse

from utils import *
from constants import *


def parse_arguments(args_list=None):
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--dataset_name",
        help="name of dataset to use, possible are {'synthetic', 'mnist', 'cifar10'}",
        required=True,
        type=str
    )
    parser.add_argument(
        "--n_tasks",
        help="number of tasks/clients_dict",
        required=True,
        type=int
    )
    parser.add_argument(
        '--frac',
        help='fraction of the dataset to be used; default: 1.0;',
        type=float,
        default=1.0
    )
    parser.add_argument(
        "--n_classes",
        help="number of classes, only used when experiment_name=='synthetic';"
             "default=10",
        default=10,
        type=int
    )
    parser.add_argument(
        "--dimension",
        help="dimension of the data, only used when experiment_name=='synthetic';"
             "default=60",
        default=60,
        type=int
    )
    parser.add_argument(
        "--n_samples",
        help="total number of samples, only used when experiment_name=='synthetic';"
             "default=10000",
        default=10000,
        type=int
    )
    parser.add_argument(
        "--alpha",
        help="parameter controlling how much local models differ from each other;" 
             "only used when experiment_name=='synthetic';"
             "expected to be in the range (0,1);"
             "default=0.0",
        default=0.0,
        type=float
    )
    parser.add_argument(
        "--beta",
        help="parameter controlling how much the local data at each device differs from that of other devices;" 
             "only used when experiment_name=='synthetic';"
             "expected to be in the range (0,1);"
             "default=0.0",
        default=0.0,
        type=float
    )
    parser.add_argument(
        "--iid",
        help="if selected, data are split iid",
        action='store_true',
        default=False
    )
    parser.add_argument(
        "--tasks_deterministic_split",
        help="if selected, tasks are assigned to tasks types proportionally to joint_probability_matrix",
        action='store_true'
    )
    parser.add_argument(
        "--tasks_proportion",
        help="parameter controlling the proportion of more available clients;"
             "expected to be in the range (0,1);"
             "default=0.5",
        default=0.5,
        type=float
    )
    parser.add_argument(
        "--log_normal",
        help="if selected, the number of samples among clients follows a log-normal distribution,"
             "only used when experiment_name=='synthetic'",
        action='store_true'
    )
    parser.add_argument(
        "--by_labels_split",
        help="if selected, data are split non-iid (by labels),"
             "only used when experiment_name=='mnist' or experiment_name=='cifar10'",
        action='store_true'
    )
    parser.add_argument(
        '--n_components',
        help='number of components/clusters; ignored if `--by_labels_split` is not used; default is -1',
        type=int,
        default=-1
    )
    parser.add_argument(
        '--dirichlet_alpha',
        help='parameter controlling clients dissimilarity, the smaller alpha is the more clients are dissimilar;'
             'ignored if `--by_labels_split` is not used; default is 0.5',
        type=float,
        default=0.5
    )
    parser.add_argument(
        "--pathological_split",
        help="if selected, data are split non-iid (pathological)",
        action='store_true'
    )
    parser.add_argument(
        '--n_shards',
        help='number of shards given to each clients; ignored if `--pathological_split` is not used;'
             'default is 1',
        type=int,
        default=1
    )
    parser.add_argument(
        "--availability_parameter",
        help="parameter controlling the asymptotic availability of the tasks;"
             "default is 0.4",
        type=float,
        default=0.4
    )
    parser.add_argument(
        "--stability_parameter",
        help="parameters controlling the stability of the tasks;"
             "default is 0.9",
        type=float,
        default=0.9
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        help='path of the directory to save data and configuration;'
             'the directory will be created if not already created;'
             'if not specified the data is saved to "./{dataset_name}";',
        default=argparse.SUPPRESS
    )
    parser.add_argument(
        '--seed',
        help='seed for the random number generator;'
             'if not specified the system clock is used to generate the seed;',
        type=int,
        default=argparse.SUPPRESS,
    )

    if args_list:
        args = parser.parse_args(args_list)
    else:
        args = parser.parse_args()

    return args


if __name__ == "__main__":
    args_ = parse_arguments()

    seed = (args_.seed if (("seed" in args_) and (args_.seed >= 0)) else int(time.time()))
    rng = np.random.default_rng(seed)

    if "save_dir" in args_:
        save_dir = args_.save_dir
    else:
        save_dir = os.path.join(".", args_.dataset_name)
        warnings.warn(f"'--save_dir' is not specified, results are saved to {save_dir}!", RuntimeWarning)

    os.makedirs(save_dir, exist_ok=True)

    if args_.dataset_name == "synthetic":
        print(f"==> IID: {args_.iid}, alpha={args_.alpha}, beta={args_.beta}")
        all_tasks_cfg = generate_synthetic(
            n_tasks=args_.n_tasks,
            n_classes=args_.n_classes,
            n_samples=args_.n_samples,
            dimension=args_.dimension,
            alpha=args_.alpha,
            beta=args_.beta,
            iid=args_.iid,
            availability_parameter=args_.availability_parameter,
            stability_parameter=args_.stability_parameter,
            tasks_deterministic_split=args_.tasks_deterministic_split,
            tasks_proportion=args_.tasks_proportion,
            log_normal=args_.log_normal,
            save_dir=os.path.join(save_dir, "all_tasks"),
            rng=rng
        )
    elif args_.dataset_name == "mnist" or args_.dataset_name == "cifar10":
        if args_.pathological_split:
            split_type = "pathological_split"
        elif args_.by_labels_split:
            split_type = "by_labels_split"
        else:
            split_type = "iid"

        dataset = get_dataset(
            dataset_name=args_.dataset_name,
            raw_data_path=os.path.join(save_dir, "raw_data")
        )

        all_tasks_cfg = generate_data(
            dataset=dataset,
            frac=args_.frac,
            split_type=split_type,
            n_classes=N_CLASSES[args_.dataset_name],
            n_train_samples=N_TRAIN_SAMPLES[args_.dataset_name],
            n_tasks=args_.n_tasks,
            tasks_deterministic_split=args_.tasks_deterministic_split,
            tasks_proportion=args_.tasks_proportion,
            n_components=args_.n_components,
            alpha=args_.dirichlet_alpha,
            n_shards=args_.n_shards,
            availability_parameter=args_.availability_parameter,
            stability_parameter=args_.stability_parameter,
            save_dir=os.path.join(save_dir, "all_tasks"),
            rng=rng
        )
    else:
        error_message = f"{args_.dataset_name} is not available, possible datasets are:"
        for n in DATASETS:
            error_message += f" {n},"
        error_message = error_message[:-1]

        raise NotImplementedError(error_message)

    save_cfg(save_path=os.path.join(save_dir, "cfg.json"), cfg=all_tasks_cfg)
