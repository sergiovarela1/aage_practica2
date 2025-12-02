from typing import Dict, List
import numpy as np
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
        """Entrenamiento local con soporte para FedProx."""
        # 1. Guardamos los parámetros globales antes de entrenar (para FedProx)
        global_params = parameters
        
        # 2. Actualizamos el modelo local con los parámetros globales
        set_model_params(self.model, parameters)

        # 3. Leemos 'mu' de la configuración enviada por el servidor
        # Si no existe (ej. en FedAvg), usamos 0.0
        mu = config.get("proximal_mu", 0.0)

        n_train = len(self.trainloader.dataset)
        if n_train > 0:
            train_local(
                self.model, 
                self.trainloader, 
                epochs=self.epochs, 
                lr=self.lr,
                mu=float(mu),             # Pasamos mu
                global_params=global_params # Pasamos los pesos originales
            )

        _, acc = test_local(self.model, self.trainloader)
        metrics: Dict[str, float] = {"train_accuracy": float(acc)}

        return get_model_parameters(self.model), n_train, metrics

    def evaluate(self, parameters, config):
        set_model_params(self.model, parameters)
        n_val = len(self.valloader.dataset)
        if n_val == 0:
            return 0.0, 0, {"test_accuracy": 0.0}

        loss, acc = test_local(self.model, self.valloader)
        metrics: Dict[str, float] = {"test_accuracy": float(acc)}
        return float(loss), n_val, metrics

def client_fn(context: Context):
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