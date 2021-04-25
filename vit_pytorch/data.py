import os 
import random
import numpy as np 
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as trns


class ImageDataset(Dataset):
    def __init__(self, image_paths, targets, transform):
        self.image_paths = image_paths
        self.targets = targets
        self.transform = transform

    def __getitem__(self, i):
        image = Image.open(self.image_paths[i])
        return self.transform(image), self.targets[i]

    def __len__(self):
        return len(self.image_paths)


def create_loaders(args):
    assert not (args.valid_dir and args.valid_rate)

    train_paths, train_targets, labels = \
        _prepare_data(args.train_dir)

    if args.valid_dir is not None:
        valid_paths, valid_targets, _ = \
            _prepare_data(args.valid_dir)

    elif args.valid_rate is not None:
        indices = np.arange(len(train_paths))
        np.random.shuffle(indices)
        split = len(indices) * args.valid_rate
        train_indices = indices[:n]
        valid_paths = train_paths[:split]
        train_paths = train_paths[split:]
        valid_targets = train_targets[:split]
        train_targets = train_targets[split:]
    else:
        valid_paths = None
        valid_targets = None 

    train_transform, valid_transform = _get_transforms(args)

    train_dataset = ImageDataset(
        train_paths, train_targets, train_transform)
    
    if valid_paths is not None:
        valid_dataset = ImageDataset(
            valid_paths, valid_targets, valid_transform)
    else:
        valid_dataset = None 
    
    train_loader = DataLoader(train_dataset, 
                              batch_size=args.batch_size, 
                              shuffle=True, 
                              pin_memory=True, 
                              num_workers=2)

    if valid_dataset is not None:
        valid_loader = DataLoader(valid_dataset,
                                  batch_size=args.batch_size, 
                                  shuffle=False, 
                                  pin_memory=True, 
                                  num_workers=2)

    else:
        valid_loader = None 

    num_classes = len(labels)

    return train_loader, valid_loader, num_classes


def _prepare_data(image_dir):
    image_paths = []
    targets = []
    labels = {}
    idx = 0

    for label in os.listdir(image_dir):
        sub_dir = os.path.join(image_dir, label)

        for path in os.listdir(sub_dir):
            image_paths.append(os.path.join(sub_dir, path))

            if label not in labels:
                labels[label] = idx
                idx += 1

            targets.append(labels[label])

    return image_paths, targets, labels

 
def _get_transforms(args):
    image_size = args.image_size
    resize = image_size + args.crop_margin

    train_transform = trns.Compose([trns.Resize((resize, resize)),
                                    trns.RandomCrop((image_size, image_size)),
                                    trns.RandomHorizontalFlip(args.horizontal_flip),
                                    trns.RandomRotation(args.rotation),
                                    trns.ToTensor(),
                                    trns.Normalize(0.5, 0.5)])

    valid_transform = trns.Compose([trns.Resize((image_size, image_size)),
                                    trns.ToTensor(),
                                    trns.Normalize(0.5, 0.5)])

    return train_transform, valid_transform
