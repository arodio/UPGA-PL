import time
import json

from abc import ABC, abstractmethod


def normalize_weights(clients_weights):
    """

    Parameters
    ----------
    clients_weights: List[float]


    Returns
    -------
        * List[float]

    """
    if isinstance(clients_weights, list):
        clients_weights = np.array(clients_weights)

    clients_weights /= clients_weights.sum()

    return clients_weights.tolist()


class ClientsSampler(ABC):
    r"""Base class for clients_dict sampler

    Attributes
    ----------
    activity_simulator: ActivitySimulator

    clients_ids:

    n_clients: int

    clients_weights_dict: Dict[int: float]
        maps clients ids to their corresponding weight/importance in the true objective function

    _availability_dict: Dict[int: float]
        maps clients ids to their stationary participation probability

    _stability_dict: Dict[int: float]
        maps clients ids to the spectral gap of their corresponding markov chains

    history: Dict[int: Dict[str: List]]
        stores the active and sampled clients and their weights at every time step

    _time_step: int
        tracks the number of steps

    rng:

    Methods
    ----------
    __init__

    _update_estimates

    sample_clients

    step

    save_history

    """

    def __init__(
            self,
            activity_simulator,
            clients_weights_dict,
            rng=None,
            *args,
            **kwargs
    ):
        """

        Parameters
        ----------
        activity_simulator: ActivitySimulator

        activity_estimator: ActivityEstimator

        clients_weights_dict: Dict[int: float]

        rng:

        """

        self.activity_simulator = activity_simulator

        self.clients_ids = list(clients_weights_dict.keys())
        self.n_clients = len(self.clients_ids)

        self.clients_weights_dict = clients_weights_dict

        self._availability_dict, self._stability_dict = self._gather_clients_parameters()

        self.history = dict()

        self._time_step = -1

        if rng is None:
            seed = int(time.time())
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = rng

    def get_active_clients(self):
        return self.activity_simulator.get_active_clients()

    def _gather_clients_parameters(self):
        availability_dict = dict()
        stability_dict = dict()

        for idx, client_id in enumerate(self.activity_simulator.clients_ids):
            availability_dict[int(client_id)] = float(self.activity_simulator.availabilities[idx])
            stability_dict[int(client_id)] = float(self.activity_simulator.stabilities[idx])

        return availability_dict, stability_dict

    def step(self, active_clients, sampled_clients_ids, sampled_clients_weights):
        """update the internal state of the clients sampler

        Parameters
        ----------
        active_clients: List[int]

        sampled_clients_ids: List[int]

        sampled_clients_weights: Dict[int: float]


        Returns
        -------
            None
        """
        self.activity_simulator.step()
        self._time_step += 1

        current_state = {
            "active_clients": active_clients,
            "sampled_clients_ids": sampled_clients_ids,
            "sampled_clients_weights": sampled_clients_weights
        }

        self.history[self._time_step] = current_state

    def save_history(self, json_path):
        """save history and clients ids

        save a dictionary with:
            * history: stores the active and sampled clients and their weights at every time step
            * clients_ids: list of clients ids stored as integers
            * true_weights_dict: dictionary mapping clients ids to their true weights

        Parameters
        ----------
        json_path: path of a .json file

        Returns
        -------
            None
        """
        metadata = {
            "history": self.history,
            "clients_ids": self.clients_ids,
            "clients_true_weights": self.clients_weights_dict,
            "clients_true_availability": self._availability_dict,
            "clients_true_stability": self._stability_dict
        }

        with open(json_path, "w") as f:
            json.dump(metadata, f)

    @abstractmethod
    def sample(self, active_clients):
        """sample clients_dict

        Parameters
        ----------
        active_clients: List[int]

        Returns
        -------
            * List[int]: indices of the sampled clients_dict
            * LIst[float]: weights to be associated to the sampled clients_dict
        """
        pass


class UnbiasedClientsSampler(ClientsSampler):
    """
    Samples all active clients_dict
    """

    def sample(self, active_clients):
        sampled_clients_ids, sampled_clients_weights = [], []

        for client_id in active_clients:
            sampled_clients_ids.append(client_id)

            sampled_clients_weights.append(
                self.clients_weights_dict[client_id] / self._availability_dict[client_id]
            )

        self.step(active_clients, sampled_clients_ids, sampled_clients_weights)

        return sampled_clients_ids, sampled_clients_weights


class BiasedClientsSampler(ClientsSampler):
    """
    Samples all active clients_dict
    """

    def sample(self, active_clients):
        sampled_clients_ids, sampled_clients_weights = [], []

        for client_id in active_clients:
            sampled_clients_ids.append(client_id)

            sampled_clients_weights.append(
                self.clients_weights_dict[client_id]
            )

        sampled_clients_weights = normalize_weights(sampled_clients_weights)

        self.step(active_clients, sampled_clients_ids, sampled_clients_weights)

        return sampled_clients_ids, sampled_clients_weights


class IdealClientsSampler(ClientsSampler):
    """
    Samples all clients_dict
    """

    def sample(self, active_clients):
        sampled_clients_ids, sampled_clients_weights = [], []

        for client_id in self.clients_ids:
            sampled_clients_ids.append(client_id)

            sampled_clients_weights.append(
                self.clients_weights_dict[client_id]
            )

        self.step(self.clients_ids, sampled_clients_ids, sampled_clients_weights)

        return sampled_clients_ids, sampled_clients_weights
