import numpy as np
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Modelo MLP para Fashion-MNIST ---
class FashionMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, 28, 28)
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# --- cache global de dataset y particiones, igual idea que fds en sklearn_example ---
_train_dataset = None
_test_dataset = None
_client_indices: List[List[int]] | None = None


def load_global_datasets(data_dir: str = "./data"):
    global _train_dataset, _test_dataset
    if _train_dataset is not None and _test_dataset is not None:
        return _train_dataset, _test_dataset

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    train_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )
    return train_dataset, test_dataset


def dirichlet_partition_non_iid(
    labels: np.ndarray,
    n_clients: int,
    alpha: float,
    seed: int = 42,
) -> List[List[int]]:
    rng = np.random.default_rng(seed)
    labels = np.array(labels)
    n_classes = labels.max() + 1

    class_indices = [np.where(labels == c)[0] for c in range(n_classes)]
    client_indices: List[List[int]] = [[] for _ in range(n_clients)]

    for c, idxs in enumerate(class_indices):
        rng.shuffle(idxs)
        proportions = rng.dirichlet(alpha=[alpha] * n_clients)
        sizes = (proportions * len(idxs)).astype(int)
        diff = len(idxs) - sizes.sum()
        for i in range(diff):
            sizes[i % n_clients] += 1

        start = 0
        for cid, size in enumerate(sizes):
            client_indices[cid].extend(idxs[start : start + size].tolist())
            start += size

    for cid in range(n_clients):
        rng.shuffle(client_indices[cid])

    return client_indices


def enforce_max_classes_per_client(
    client_indices: List[List[int]],
    labels: np.ndarray,
    max_classes: int,
    seed: int = 0,
) -> List[List[int]]:
    rng = np.random.default_rng(seed)
    new_client_indices: List[List[int]] = []

    for idxs in client_indices:
        rng.shuffle(idxs)
        seen_classes = set()
        kept: List[int] = []

        for i in idxs:
            c = int(labels[i])
            if c in seen_classes:
                kept.append(i)
            elif len(seen_classes) < max_classes:
                seen_classes.add(c)
                kept.append(i)
            else:
                continue

        new_client_indices.append(kept)

    return new_client_indices


def load_data(
    partition_id: int,
    num_partitions: int,
    batch_size: int,
    alpha: float,
    max_classes: int = 3,
) -> Tuple[DataLoader, DataLoader]:
    """Equivalente a load_data del ejemplo sklearn, pero devolviendo DataLoaders."""
    global _client_indices
    train_dataset, _ = load_global_datasets()

    if _client_indices is None:
        labels = train_dataset.targets.numpy()
        client_indices = dirichlet_partition_non_iid(
            labels=labels,
            n_clients=num_partitions,
            alpha=alpha,
            seed=42,
        )
        client_indices = enforce_max_classes_per_client(
            client_indices, labels, max_classes=max_classes, seed=0
        )
        _client_indices = client_indices

    assert _client_indices is not None
    idxs = _client_indices[partition_id]
    client_dataset = Subset(train_dataset, idxs)

    n = len(client_dataset)
    if n <= 1:
        train_subset = client_dataset
        val_subset = client_dataset
    else:
        n_train = int(0.8 * n)
        n_val = n - n_train
        train_subset, val_subset = random_split(
            client_dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )

    train_loader = DataLoader(
        train_subset,
        batch_size=min(batch_size, len(train_subset)),
        shuffle=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=min(batch_size, max(1, len(val_subset))),
        shuffle=False,
    )
    return train_loader, val_loader


# --- funciones de parámetros (como get_model_parameters/set_model_params de sklearn_example) ---


def get_model_parameters(model: nn.Module) -> List[np.ndarray]:
    return [val.cpu().detach().numpy() for _, val in model.state_dict().items()]


def set_model_params(model: nn.Module, params: List[np.ndarray]) -> None:
    state_dict = model.state_dict()
    new_state_dict = {}
    for (k, old_v), new_v in zip(state_dict.items(), params):
        new_state_dict[k] = torch.tensor(new_v, dtype=old_v.dtype)
    model.load_state_dict(new_state_dict, strict=True)


# --- entrenamiento y evaluación local ---


def train_local(
    model: nn.Module,
    trainloader: DataLoader,
    epochs: int,
    lr: float,
) -> None:
    model.to(DEVICE)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


def test_local(
    model: nn.Module,
    dataloader: DataLoader,
) -> Tuple[float, float]:
    model.to(DEVICE)
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if total == 0:
        return 0.0, 0.0
    return total_loss / total, correct / total
