import torch
from torchvision import datasets, transforms


def get_mnist_dataloaders():
    try:
        # Define a transform to normalize the data
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

        # Download and load the training data
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

        # Download and load the test data
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

        # Print the size of the datasets
        print("Training dataset size:", len(train_dataset))
        print("Test dataset size:", len(test_dataset))

        return train_loader, test_loader
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        return None








