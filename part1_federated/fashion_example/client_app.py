from typing import Dict

from flwr.client import NumPyClient
from flwr.clientapp import ClientApp
from flwr.common import Context

from fashion_example.task import (
    FashionMLP,
    get_model_parameters,
    set_model_params,
    load_data,
    train_local,
    test_local,
)


class FlowerClient(NumPyClient):
    def __init__(self, model, trainloader, valloader, epochs: int, lr: float):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.epochs = epochs
        self.lr = lr

    def fit(self, parameters, config):
        """Entrenamiento local en el cliente."""
        set_model_params(self.model, parameters)

        n_train = len(self.trainloader.dataset)
        if n_train > 0:
            train_local(self.model, self.trainloader, epochs=self.epochs, lr=self.lr)

        # podemos calcular la accuracy de train usando val como proxy sencilla
        _, acc = test_local(self.model, self.trainloader)
        metrics: Dict[str, float] = {"train_accuracy": float(acc)}

        return get_model_parameters(self.model), n_train, metrics

    def evaluate(self, parameters, config):
        """Evaluación local en el cliente."""
        set_model_params(self.model, parameters)

        n_val = len(self.valloader.dataset)
        if n_val == 0:
            return 0.0, 0, {"test_accuracy": 0.0}

        loss, acc = test_local(self.model, self.valloader)
        metrics: Dict[str, float] = {"test_accuracy": float(acc)}
        return float(loss), n_val, metrics


def client_fn(context: Context):
    """Construye el cliente con su partición de datos (igual patrón que sklearn_example)."""
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])

    cfg = context.run_config
    batch_size = int(cfg["batch-size"])
    local_epochs = int(cfg["local-epochs"])
    lr = float(cfg["lr"])
    alpha = float(cfg["alpha-dirichlet"])

    trainloader, valloader = load_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        batch_size=batch_size,
        alpha=alpha,
        max_classes=3,
    )

    model = FashionMLP()
    return FlowerClient(model, trainloader, valloader, local_epochs, lr).to_client()


app = ClientApp(client_fn=client_fn)
