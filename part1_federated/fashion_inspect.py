from torchvision import datasets, transforms

def main():
    # 1. Definimos las transformaciones que queremos aplicar a las imágenes
    transform = transforms.Compose([
        transforms.ToTensor(),             # Pasa la imagen a tensor [0,1]
        transforms.Normalize((0.5,), (0.5,))  # Normaliza (de momento algo simple)
    ])

    # 2. Cargamos Fashion-MNIST (train y test)
    train_dataset = datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.FashionMNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )

    print("Tamaño train:", len(train_dataset))
    print("Tamaño test :", len(test_dataset))

    # 3. Miramos un par de ejemplos
    image0, label0 = train_dataset[0]
    print("Ejemplo 0 - shape imagen:", image0.shape)  # (1, 28, 28)
    print("Ejemplo 0 - etiqueta:", label0)

    # 4. Mapa de etiquetas de Fashion-MNIST
    label_map = {
        0: "T-shirt/top",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle boot",
    }
    print("Etiqueta 0 significa:", label_map[int(label0)])

if __name__ == "__main__":
    main()
