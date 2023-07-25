
import torch
from easyfsl.methods import PrototypicalNetworks
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import Omniglot
from torchvision.models import resnet18
from tqdm import tqdm

from easyfsl.samplers import TaskSampler
from easyfsl.datasets.easy_set import EasySet
from easyfsl.utils import plot_images, sliding_average

from pathlib import Path
from torch.utils.data import Dataset
# from torchvision.io import read_image
from PIL import Image



if __name__ == '__main__':
    data_dir = '/home/crx/150gdata/easyfsl_custom_data-main/data/face_few_shot_test/train'
    json_path = '/home/crx/150gdata/20230421_facedata/experiment_img/train/easy_set_train.json'
    image_size = 128
    N_WAY = 5  # 5  # Number of classes in a task
    N_SHOT = 2  # 5  # Number of images per class in the support set
    N_QUERY = 3  # 10  # Number of images per class in the query set
    N_TRAINING_EPISODES = 40000
    N_VALIDATION_TASKS = 100
    train_set = EasySet(
        specs_file= json_path,
        image_size=128,
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
    convolutional_network = resnet18(pretrained=True)
    convolutional_network.fc = nn.Flatten()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PrototypicalNetworks(convolutional_network).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    N_EVALUATION_TASKS = 100
    val_json_path = "/home/crx/150gdata/20230421_facedata/experiment_img/val/easy_set_val.json"
    # The sampler needs a dataset with a "get_labels" method. Check the code if you have any doubt!
    valid_set = EasySet(
        specs_file=val_json_path,
        image_size=128,
        transform=transforms.Compose(
            [
                # If images have 1 channel, our model will expect 3-channel images
                # transforms.Grayscale(num_output_channels=3),
                transforms.Resize([int(image_size * 1.15), int(image_size * 1.15)]),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ]
        ),
    )

    valid_sampler = TaskSampler(
        valid_set, n_way=5, n_shot=3, n_query=5, n_tasks=N_EVALUATION_TASKS
    )

    valid_loader = DataLoader(
        valid_set,
        batch_sampler=valid_sampler,
        # num_workers=12,
        pin_memory=True,
        collate_fn=valid_sampler.episodic_collate_fn,
    )

    (
        example_support_images,
        example_support_labels,
        example_query_images,
        example_query_labels,
        example_class_ids,
    ) = next(iter(valid_loader))

    plot_images(example_support_images, "support images", images_per_row=N_SHOT)
    plot_images(example_query_images, "query images", images_per_row=N_QUERY)

    model.eval()
    example_scores = model(
        example_support_images.to(device),  # .cuda(),
        example_support_labels.to(device),  # .cuda(),
        example_query_images.to(device),  # .cuda(),
    ).detach()

    _, example_predicted_labels = torch.max(example_scores.data, 1)

def evaluate_on_one_task(
    support_images: torch.Tensor,
    support_labels: torch.Tensor,
    query_images: torch.Tensor,
    query_labels: torch.Tensor,
) -> [int, int]:
    """
    Returns the number of correct predictions of query labels, and the total number of predictions.
    """
    return (
            torch.max(
            model(support_images.to(device), support_labels.to(device), query_images.to(device))
            .detach()
            .data,
            1,
            )[1]
            == query_labels.to(device)  # .cuda()
           ).sum().item(), len(query_labels)


def evaluate(data_loader: DataLoader):
    # We'll count everything and compute the ratio at the end
    total_predictions = 0
    correct_predictions = 0

    # eval mode affects the behaviour of some layers (such as batch normalization or dropout)
    # no_grad() tells torch not to keep in memory the whole computational graph (it's more lightweight this way)
    model.eval()
    with torch.no_grad():
        for episode_index, (
            support_images,
            support_labels,
            query_images,
            query_labels,
            class_ids,
        ) in tqdm(enumerate(data_loader), total=len(data_loader)):

            correct, total = evaluate_on_one_task(
                support_images, support_labels, query_images, query_labels
            )

            total_predictions += total
            correct_predictions += correct

    print(
        f"Model tested on {len(data_loader)} tasks. Accuracy: {(100 * correct_predictions/total_predictions):.2f}%"
    )

##########################################################################################################################
#train
# Train the model yourself with this cell


N_TRAINING_EPISODES = 40000
N_VALIDATION_TASKS = 100

# train_set.get_labels = lambda: [instance[1] for instance in train_set._flat_character_images]
train_sampler = TaskSampler(
    train_set, n_way=N_WAY, n_shot=N_SHOT, n_query=N_QUERY, n_tasks=N_TRAINING_EPISODES
)
train_loader = DataLoader(
    train_set,
    batch_sampler=train_sampler,
    num_workers=12,
    pin_memory=True,
    collate_fn=train_sampler.episodic_collate_fn,
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def fit(
    support_images: torch.Tensor,
    support_labels: torch.Tensor,
    query_images: torch.Tensor,
    query_labels: torch.Tensor,
) -> float:
    optimizer.zero_grad()
    classification_scores = model(
        support_images.to(device), support_labels.to(device), query_images.to(device)
    )

    loss = criterion(classification_scores, query_labels.to(device))
    loss.backward()
    optimizer.step()

    return loss.item()



log_update_frequency = 10

all_loss = []
model.train()
with tqdm(enumerate(train_loader), total=len(train_loader)) as tqdm_train:
    for episode_index, (
        support_images,
        support_labels,
        query_images,
        query_labels,
        _,
    ) in tqdm_train:
        loss_value = fit(support_images, support_labels, query_images, query_labels)
        all_loss.append(loss_value)

        if episode_index % log_update_frequency == 0:
            tqdm_train.set_postfix(loss=sliding_average(all_loss, log_update_frequency))


test_set = EasySet(
    specs_file=val_json_path,
    image_size=128,
    # test=True,
    transform=transforms.Compose(
        [
            # If images have 1 channel, our model will expect 3-channel images
            # transforms.Grayscale(num_output_channels=3),
            transforms.Resize([int(image_size * 1.15), int(image_size * 1.15)]),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    ),
)

N_WAY = 100 # 5  # Number of classes in a task
N_SHOT = 2 # 5  # Number of images per class in the support set
N_QUERY = 5 # 10  # Number of images per class in the query set
N_EVALUATION_TASKS = 100

test_sampler = TaskSampler(
    valid_set, n_way=N_WAY, n_shot=N_SHOT, n_query=N_QUERY, n_tasks=N_EVALUATION_TASKS
)

test_loader = DataLoader(
    valid_set,
    batch_sampler=test_sampler,
    # num_workers=12,
    pin_memory=True,
    collate_fn=test_sampler.episodic_collate_fn,
)

#%%

evaluate(test_loader)