from torchvision import datasets, transforms

def get_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5), std=(0.5)),
    ])
    ds = datasets.MNIST("data", train=True, download=True, transform=transform)
    return ds
