import torch
import torchvision
import torchvision.transforms as transforms

def get_cifar10_dataloader(batch_size=4, num_workers=2):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers)

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers)

    return train_loader, test_set

def get_cifar100_dataloader(batch_size=4, num_workers=2):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = torchvision.datasets.CIFAR100(root='./data', train=True,
                                            download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers)

    test_set = torchvision.datasets.CIFAR100(root='./data', train=False,
                                           download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers)

    return train_loader, test_loader

if __name__ == "__main__":
    # CIFAR10
    train_loader, test_loader = get_cifar10_dataloader()
    print(f" Size of CIFAR10 Train set {len(train_loader)}")
    print(f" Size of CIFAR10 Test set {len(test_loader)}")
    images, labels = next(iter(train_loader))
    print(f"Shape of CIFAR10 Train Images {images.size()}")
    print(f"Shape of CIFAR10 Train Labels {labels.size()}")
    
    # CIFAR100
    train_loader, test_loader = get_cifar100_dataloader()
    print(f" Size of CIFAR100 Train set {len(train_loader)}")
    print(f" Size of CIFAR100 Test set {len(test_loader)}")
    images, labels = next(iter(train_loader))
    print(f"Shape of CIFAR100 of Train Images {images.size()}")
    print(f"Shape of CIFAR100 Train Labels {labels.size()}")