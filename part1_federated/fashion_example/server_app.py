import json
import os
from typing import Dict, List, Tuple, Union

from flwr.common import Context, Metrics, Scalar, ndarrays_to_parameters, Parameters, FitRes, EvaluateRes
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg, FedProx

from fashion_example.task import FashionMLP, get_model_parameters

# --- Función de agregación de métricas ---
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Dict[str, Scalar]:
    if not metrics:
        return {}
    results: Dict[str, float] = {}
    for _, m in metrics:
        for k, v in m.items():
            if isinstance(v, (float, int)):
                results[k] = results.get(k, 0.0) + v / len(metrics)
    return results

# --- CLASE PERSONALIZADA PARA GUARDAR JSON ---
class SaveResultsStrategy(FedProx):
    def __init__(self, strategy_name="FedProx", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_name = strategy_name
        self.history = {
            "strategy": strategy_name,
            "rounds": [],
            "test_accuracy": [],
            "loss_distributed": []
        }

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Union[float, None], Dict[str, Scalar]]:
        
        # 1. Ejecutar la lógica original de FedProx/FedAvg
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(server_round, results, failures)

        # 2. Guardar los resultados en nuestra lista
        if loss_aggregated is not None:
            self.history["rounds"].append(server_round)
            self.history["loss_distributed"].append(loss_aggregated)
            
            if "test_accuracy" in metrics_aggregated:
                self.history["test_accuracy"].append(metrics_aggregated["test_accuracy"])

            # 3. Escribir al archivo JSON en cada ronda
            filename = f"results_{self.strategy_name}.json"
            with open(filename, "w") as f:
                json.dump(self.history, f, indent=4)
            
            # Mensaje visual en el log
            # print(f"Resultados guardados en {filename} (Ronda {server_round})")

        return loss_aggregated, metrics_aggregated

# --- CONFIGURACIÓN DEL SERVIDOR ---
def server_fn(context: Context) -> ServerAppComponents:
    cfg = context.run_config
    num_rounds = int(cfg["num-server-rounds"])
    fraction_fit = float(cfg["fraction-fit"])
    min_available_clients = int(cfg["min-available-clients"])
    
    # --- AJUSTES DE LA PRÁCTICA ---
    strategy_name = "FedAvg"  # Cambiar a "FedAvg" si quieres probar el otro
    proximal_mu = 0          # Mu para FedProx (0.1, 1.0, etc.), para FedAvg se ignora
    # ------------------------------

    print(f"\n Iniciando estrategia: {strategy_name} (mu={proximal_mu if strategy_name == 'FedProx' else 0})")

    model = FashionMLP()
    ndarrays = get_model_parameters(model)
    global_init = ndarrays_to_parameters(ndarrays)

    # Configuración que se envía al cliente
    def on_fit_config_fn(server_round: int):
        return {
            "current_round": server_round,
            "proximal_mu": proximal_mu if strategy_name == "FedProx" else 0.0
        }

    # Usamos nuestra clase personalizada que hereda de FedProx
    # (Nota: FedAvg es básicamente FedProx con mu=0, así que podemos usar la misma clase base)
    strategy = SaveResultsStrategy(
        strategy_name=strategy_name,
        min_available_clients=min_available_clients,
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=global_init,
        proximal_mu=proximal_mu,
        on_fit_config_fn=on_fit_config_fn
    )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)

app = ServerApp(server_fn=server_fn)