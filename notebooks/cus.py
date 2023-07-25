
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import Omniglot
from torchvision.models import resnet18
from tqdm import tqdm

from easyfsl.samplers import TaskSampler
from easyfsl.utils import plot_images, sliding_average

from pathlib import Path
from torch.utils.data import Dataset
# from torchvision.io import read_image
from PIL import Image



def _validate_root_dir(root):
    # todo: raise exception or warning
    pass


def _validate_train_flag(train: bool, valid: bool, test: bool):
    assert [train, valid, test].count(True) == 1, "one of train, valid & test must be true."


class CustomDataset(Dataset):
    def __init__(self, root,
                 train: bool = False, valid: bool = False, test: bool = False,
                 transform=None, target_transform=None, ):

        _validate_root_dir(root)
        _validate_train_flag(train, valid, test)
        self.transform = transform
        self.target_transform = target_transform
        if train:
            self.data_dir = Path(root) / 'train'
        elif valid:
            self.data_dir = Path(root) / 'valid'
        elif test:
            self.data_dir = Path(root) / 'test'

        self._image_paths = sorted(
            list(self.data_dir.glob("**/*.jpg")) +
            list(self.data_dir.glob("**/*.jpeg")) +
            list(self.data_dir.glob("**/*.png")))
        self._image_labels = [int(i.parent.name) for i in self._image_paths]
        assert len(self._image_paths) == len(self._image_labels)

    def __len__(self):
        return len(self._image_paths)

    def __getitem__(self, idx):
        x = Image.open(str(self._image_paths[idx]))
        y = self._image_labels[idx]
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(x)
        return x, y

    def get_labels(self):
        return self._image_labels

if __name__ == '__main__':
    data_dir = '/home/crx/150gdata/easyfsl_custom_data-main/data/download_data'
    image_size = 128
    N_WAY = 5  # 5  # Number of classes in a task
    N_SHOT = 3  # 5  # Number of images per class in the support set
    N_QUERY = 5  # 10  # Number of images per class in the query set
    N_TRAINING_EPISODES = 40000
    N_VALIDATION_TASKS = 100
    train_set = CustomDataset(
        root=data_dir,
        train=True,
        transform=transforms.Compose(
            [
                # Omniglot images have 1 channel, but our model will expect 3-channel images
                # transforms.Grayscale(num_output_channels=3),
                transforms.Resize([int(image_size * 1.5), int(image_size * 1.5)]),
                transforms.RandomPerspective(0.5, 0.8),
                transforms.CenterCrop(image_size),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.15, saturation=0, hue=0,
                ),
                transforms.ToTensor(),
            ]
        ),
    )
    train_sampler = TaskSampler(
        train_set, n_way=N_WAY, n_shot=N_SHOT, n_query=N_QUERY, n_tasks=N_TRAINING_EPISODES
    )
    train_loader = DataLoader(
        train_set,
        batch_sampler=train_sampler,
        # num_workers=12,
        pin_memory=True,
        collate_fn=train_sampler.episodic_collate_fn,
    )
