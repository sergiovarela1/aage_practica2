import numpy as np
from aage_practica2.part1_federated.fashion_example.task import (
    load_fashion_mnist,
    dirichlet_partition_non_iid,
    enforce_max_classes_per_client,
    make_client_datasets,
    get_client_train_val_loaders,
    FashionMLP,
    train_local,
    test_local,
)


def main():
    # 1. Cargar datos globales
    train_dataset, test_dataset = load_fashion_mnist()
    labels = train_dataset.targets.numpy()

    # 2. Crear partición Non-IID en 10 clientes
    client_indices = dirichlet_partition_non_iid(
        labels, n_clients=10, alpha=0.05, seed=42
    )
    client_indices = enforce_max_classes_per_client(
        client_indices, labels, max_classes=3, seed=0
    )

    client_datasets = make_client_datasets(train_dataset, client_indices)

    # 3. Cogemos solo el cliente 0 para probar
    client_id = 0
    client_dataset = client_datasets[client_id]
    print(f"Cliente {client_id} tiene {len(client_dataset)} ejemplos.")

    train_loader, val_loader = get_client_train_val_loaders(
        client_dataset, batch_size=32
    )

    # 4. Creamos un modelo y lo entrenamos localmente SOLO con los datos de ese cliente
    model = FashionMLP()
    print("Entrenando modelo local en cliente 0...")
    train_local(model, train_loader, epochs=1, lr=0.01)

    # 5. Evaluamos en el val del cliente 0
    val_loss, val_acc = test_local(model, val_loader)
    print(f"Cliente 0 - Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f}")


if __name__ == "__main__":
    main()
