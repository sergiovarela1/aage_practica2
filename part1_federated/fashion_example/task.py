import numpy as np
from typing import List, Tuple
import os 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Modelo MLP para Fashion-MNIST ---
class FashionMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# --- cache global de dataset y particiones ---
_train_dataset = None
_test_dataset = None
_client_indices: List[List[int]] | None = None

def load_global_datasets(data_dir: str = "./data"):
    global _train_dataset, _test_dataset
    if _train_dataset is not None and _test_dataset is not None:
        return _train_dataset, _test_dataset

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    train_dataset = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)
    return train_dataset, test_dataset

# --- Función de visualización ---

def visualize_data_distribution(client_indices: List[List[int]], train_dataset, filename="distribucion_clientes.png"):
    """
    Genera y guarda un histograma apilado en la carpeta 'fashion_example/histograms'.
    """
    num_clients = len(client_indices)
    num_classes = 10
    classes = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    
    # Matriz [Cliente][Clase] -> Cantidad
    distribution = np.zeros((num_clients, num_classes))
    all_labels = np.array(train_dataset.targets.numpy())

    for i, indices in enumerate(client_indices):
        client_labels = all_labels[indices]
        counts = np.bincount(client_labels, minlength=num_classes)
        distribution[i] = counts

    # Graficar
    plt.figure(figsize=(12, 6))
    client_ids = range(num_clients)
    bottom = np.zeros(num_clients)
    
    for k in range(num_classes):
        plt.bar(client_ids, distribution[:, k], bottom=bottom, label=classes[k])
        bottom += distribution[:, k]
        
    plt.xlabel("ID del Cliente")
    plt.ylabel("Número de Muestras")
    plt.title("Distribución de Clases por Cliente (Non-IID)")
    plt.xticks(client_ids)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # --- Lógica para guardar en carpeta específica ---
    
    # 1. Obtenemos la ruta donde está este archivo (task.py)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Definimos la carpeta de destino: fashion_example/histograms
    save_dir = os.path.join(base_dir, "histograms")
    
    # 3. Creamos la carpeta si no existe
    os.makedirs(save_dir, exist_ok=True)
    
    # 4. Ruta completa del archivo
    filepath = os.path.join(save_dir, filename)
    
    plt.savefig(filepath)
    print(f"\nGráfico de distribución guardado en: {filepath}\n")
    plt.close()

def dirichlet_partition_non_iid(labels: np.ndarray, n_clients: int, alpha: float, seed: int = 42) -> List[List[int]]:
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

def enforce_max_classes_per_client(client_indices: List[List[int]], labels: np.ndarray, max_classes: int, seed: int = 0) -> List[List[int]]:
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

def load_data(partition_id: int, num_partitions: int, batch_size: int, alpha: float, max_classes: int = 3) -> Tuple[DataLoader, DataLoader]:
    global _client_indices
    train_dataset, _ = load_global_datasets()

    # Si es la primera vez que se ejecuta, generamos particiones
    if _client_indices is None:
        labels = train_dataset.targets.numpy()
        
        # 1. Partición Dirichlet
        client_indices = dirichlet_partition_non_iid(
            labels=labels,
            n_clients=num_partitions,
            alpha=alpha,
            seed=42,
        )
        
        # 2. Forzar máximo de clases 
        client_indices = enforce_max_classes_per_client(
            client_indices, labels, max_classes=max_classes, seed=0
        )
        
        _client_indices = client_indices
        
        # --- Llamada a la visualización ---
        # Solo lo ejecutamos si estamos en el proceso principal o es la primera vez
        # (El try/except es por seguridad en entornos multiproceso, aunque simple funciona bien)
        try:
            visualize_data_distribution(_client_indices, train_dataset)
        except Exception as e:
            print(f"No se pudo generar el gráfico: {e}")
        # ---------------------------------------------

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

    train_loader = DataLoader(train_subset, batch_size=min(batch_size, len(train_subset)), shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=min(batch_size, max(1, len(val_subset))), shuffle=False)
    
    return train_loader, val_loader

def get_model_parameters(model: nn.Module) -> List[np.ndarray]:
    return [val.cpu().detach().numpy() for _, val in model.state_dict().items()]

def set_model_params(model: nn.Module, params: List[np.ndarray]) -> None:
    state_dict = model.state_dict()
    new_state_dict = {}
    for (k, old_v), new_v in zip(state_dict.items(), params):
        new_state_dict[k] = torch.tensor(new_v, dtype=old_v.dtype)
    model.load_state_dict(new_state_dict, strict=True)

# En fashion_example/task.py

def train_local(
    model: nn.Module,
    trainloader: DataLoader,
    epochs: int,
    lr: float,
    mu: float = 0.0, # Nuevo parámetro para FedProx
    global_params: List[np.ndarray] = None, ) -> None: # Pesos globales para comparar
    
    model.to(DEVICE)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # Si usamos FedProx, convertimos los params globales a tensores una sola vez
    global_weight_tensors = None
    if mu > 0.0 and global_params is not None:
        global_weight_tensors = [
            torch.tensor(p, device=DEVICE) for p in global_params
        ]

    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            
            # 1. Pérdida original (CrossEntropy)
            loss = criterion(outputs, labels)

            # 2. Término Proximal (Solo si mu > 0)
            if mu > 0.0 and global_weight_tensors is not None:
                proximal_term = 0.0
                for local_weights, global_weights in zip(model.parameters(), global_weight_tensors):
                    proximal_term += (local_weights - global_weights).norm(2) ** 2
                
                loss += (mu / 2) * proximal_term

            loss.backward()
            optimizer.step()

def test_local(model: nn.Module, dataloader: DataLoader) -> Tuple[float, float]:
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