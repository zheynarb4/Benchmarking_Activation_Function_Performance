from torchvision import datasets
from torch.utils.data import random_split
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import config


data_transformer = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ]
)

train_dataset = datasets.ImageFolder(config.TRAIN_FOLDER, transform = data_transformer)
test_dataset = datasets.ImageFolder(config.TEST_FOLDER, transform = data_transformer)

train_size = int(config.TRAIN_VAL_SPLIT_RATIO * len(train_dataset))
val_size = len(train_dataset) - train_size

train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_subset, batch_size = config.BATCH_SIZE, shuffle = True, num_workers = 0, pin_memory = True)
valid_loader = DataLoader(val_subset, batch_size = config.BATCH_SIZE, shuffle = True, num_workers = 0, pin_memory = True)
test_loader = DataLoader(test_dataset, batch_size = config.BATCH_SIZE, shuffle = False)
