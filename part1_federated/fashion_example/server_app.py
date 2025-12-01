from typing import Dict, List, Tuple

from flwr.common import Context, Metrics, Scalar, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from fashion_example.task import FashionMLP, get_model_parameters


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Dict[str, Scalar]:
    """Media (no estrictamente ponderada, pero suficiente) de las métricas."""
    if not metrics:
        return {}

    results: Dict[str, float] = {}
    for _, m in metrics:
        for k, v in m.items():
            if isinstance(v, (float, int)):
                results[k] = results.get(k, 0.0) + v / len(metrics)
    return results


def server_fn(context: Context) -> ServerAppComponents:
    cfg = context.run_config

    num_rounds = int(cfg["num-server-rounds"])
    fraction_fit = float(cfg["fraction-fit"])
    min_available_clients = int(cfg["min-available-clients"])

    model = FashionMLP()
    ndarrays = get_model_parameters(model)
    global_init = ndarrays_to_parameters(ndarrays)

    strategy = FedAvg(
        min_available_clients=min_available_clients,
        fraction_fit=fraction_fit,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=global_init,
    )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)
