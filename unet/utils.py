from torch.utils.data import DataLoader
from dataset import BirdDataset
import torch


def load_data_set(image_paths, image_dir, segmentation_dir, transforms, batch_size=8, shuffle=True):
    dataset = BirdDataset(image_paths,
                          image_dir,
                          segmentation_dir,
                          transform_image=transforms[0],
                          transform_mask=transforms[1])

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [11772, 16])

    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle
    ), DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )