import os
from collections import Counter
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms


# 1. Cargar Fashion-MNIST 
def load_fashion_mnist(data_dir: str = "./data"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    train = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    test = datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    return train, test


# 2. Particionado Dirichlet Non-IID
def dirichlet_partition_non_iid(
    labels: np.ndarray,
    n_clients: int = 10,
    alpha: float = 0.05,
    seed: int = 42,
) -> List[List[int]]:
    """
    labels: array de tamaño N con las etiquetas [0..9]
    Devuelve: lista de longitud n_clients donde cada elemento es una lista de índices.
    """
    rng = np.random.default_rng(seed)
    labels = np.array(labels)
    n_classes = labels.max() + 1  # debería ser 10 en Fashion-MNIST

    # índices por clase
    class_indices = [np.where(labels == c)[0] for c in range(n_classes)]
    client_indices: List[List[int]] = [[] for _ in range(n_clients)]

    for c, idxs in enumerate(class_indices):
        rng.shuffle(idxs)

        # Proporciones para esta clase c entre los clientes
        # Alpha pequeño => partición más desequilibrada (más Non-IID)
        proportions = rng.dirichlet(alpha=[alpha] * n_clients)

        # Número de muestras de esta clase que va a cada cliente
        sizes = (proportions * len(idxs)).astype(int)

        # Ajuste por redondeo
        diff = len(idxs) - sizes.sum()
        for i in range(diff):
            sizes[i % n_clients] += 1

        # Repartimos los índices según sizes
        start = 0
        for cid, size in enumerate(sizes):
            client_indices[cid].extend(idxs[start:start + size].tolist())
            start += size

    # Mezclamos los índices dentro de cada cliente
    for cid in range(n_clients):
        rng.shuffle(client_indices[cid])

    return client_indices


# 3. Limitar a 2-3 clases por cliente
def enforce_max_classes_per_client(
    client_indices: List[List[int]],
    labels: np.ndarray,
    max_classes: int = 3,
    seed: int = 0,
) -> List[List[int]]:
    """
    Recorre los índices de cada cliente y se queda sólo con ejemplos
    de como mucho 'max_classes' clases distintas.
    (Es un filtro simple pero suficiente para la práctica.)
    """
    rng = np.random.default_rng(seed)
    new_client_indices: List[List[int]] = []

    for idxs in client_indices:
        rng.shuffle(idxs)
        seen_classes = set()
        kept: List[int] = []

        for i in idxs:
            c = int(labels[i])
            if c in seen_classes:
                # ya hemos visto esta clase en el cliente, la podemos usar
                kept.append(i)
            elif len(seen_classes) < max_classes:
                # aún no hemos llegado al máximo de clases permitidas
                seen_classes.add(c)
                kept.append(i)
            else:
                # ya tenemos max_classes clases distintas, ignoramos nuevas clases
                continue

        new_client_indices.append(kept)

    return new_client_indices


# 4. Dibujar histogramas por cliente
def plot_client_histograms(
    train_dataset,
    client_indices: List[List[int]],
    out_dir: str = "./histograms",
):
    os.makedirs(out_dir, exist_ok=True)

    for cid, idxs in enumerate(client_indices):
        labels = [train_dataset[i][1] for i in idxs]  # cada item es (imagen, etiqueta)
        counts = Counter(labels)
        xs = sorted(counts.keys())
        ys = [counts[x] for x in xs]

        plt.figure()
        plt.bar(xs, ys)
        plt.xticks(xs)
        plt.xlabel("Clase (0-9)")
        plt.ylabel("Nº muestras")
        plt.title(f"Cliente {cid} - distribución de clases")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"client_{cid}_hist.png"))
        plt.close()

        print(f"Cliente {cid}: clases = {sorted(set(labels))}, total muestras = {len(idxs)}")


def main():
    # 1. Cargar datos
    train, _ = load_fashion_mnist()
    labels = train.targets.numpy()  # tensor -> numpy

    print("Total ejemplos en train:", len(train))

    # 2. Particionamos con Dirichlet
    client_indices = dirichlet_partition_non_iid(
        labels,
        n_clients=10,
        alpha=0.05,   # puedes jugar con 0.1, 0.05...
        seed=42,
    )

    # 3. Limitamos a como máximo 3 clases por cliente
    client_indices = enforce_max_classes_per_client(
        client_indices,
        labels,
        max_classes=3,
        seed=0,
    )

    # 4. Dibujamos histogramas
    plot_client_histograms(train, client_indices, out_dir="./histograms")

    print("Histogramas guardados en la carpeta ./histograms")


if __name__ == "__main__":
    main()
