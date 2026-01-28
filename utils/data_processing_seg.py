import os
import random
import numpy as np
from PIL import Image
from typing import Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import logging


logger = logging.getLogger(__name__)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class UltrasoundSegmentationDataset(Dataset):
    def __init__(self, image_dir: str, label_dir: str, num_classes: int, transform: Optional[bool] = False, image_size: Tuple[int, int] = (512, 512), task_type: str = "tumor"):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_size = image_size
        self.num_classes = num_classes
        self.task_type = task_type

        self.image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

        assert len(self.image_files) > 0, "No image files found"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        label_path = os.path.join(self.label_dir, self.image_files[idx])  

        image = Image.open(img_path).convert('RGB')
        label = Image.open(label_path).convert('L') 

        image = TF.resize(image, self.image_size, interpolation=Image.BILINEAR)
        label = TF.resize(label, self.image_size, interpolation=Image.NEAREST)

        if self.transform:
            image, label = self._joint_transform(image, label)

        image = TF.to_tensor(image)
        image = self.normalize(image)  # [C, H, W]
        
        if self.num_classes == 2:
            label = torch.from_numpy(np.array(label)).float() / 255.0  
            label = (label > 0.5).float()  
        else:
            label = torch.from_numpy(np.array(label)).long()
        return image, label

    def _joint_transform(self, image, label):
        if self.task_type == "tumor":
            if random.random() > 0.5:
                image = TF.hflip(image)
                label = TF.hflip(label)

            if random.random() > 0.5:
                image = TF.vflip(image)
                label = TF.vflip(label)

        if random.random() > 0.5:
            angle = random.uniform(-30, 30)
            image = TF.rotate(image, angle)
            label = TF.rotate(label, angle)
        
        if random.random() > 0.5:
            g = np.random.randint(5, 20) / 10.0
            image_np = np.array(image)
            image_np = (np.power(image_np / 255, 1.0 / g)) * 255
            image_np = image_np.astype(np.uint8)
            image = Image.fromarray(image_np)  
        
        if random.random() > 0.5:
            scale = np.random.uniform(1, 1.3)
            h, w = self.image_size
            new_h, new_w = int(h * scale), int(w * scale)
            image = TF.resize(image, (new_h, new_w), interpolation=Image.BILINEAR)
            label = TF.resize(label, (new_h, new_w), interpolation=Image.NEAREST)
            i, j, crop_h, crop_w = T.RandomCrop.get_params(image, self.image_size)
            image = TF.crop(image, i, j, crop_h, crop_w)
            label = TF.crop(label, i, j, crop_h, crop_w)

        if random.random() > 0.5:
            contr_tf = T.ColorJitter(contrast=(0.5, 2.0))
            image = contr_tf(image)

        return image, label


class DataProcessor:
    @staticmethod
    def get_data_loaders(args) -> Tuple[DataLoader, DataLoader, DataLoader]:
        def worker_init_fn(worker_id):
            seed = 42 + worker_id
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        train_dataset = UltrasoundSegmentationDataset(
            os.path.join(args.data_dir, 'train/img'),
            os.path.join(args.data_dir, 'train/label'),
            args.num_classes,
            transform=True,
            image_size=(args.img_size, args.img_size),
            task_type = args.task_type
        )

        val_dataset = UltrasoundSegmentationDataset(
            os.path.join(args.data_dir, 'val/img'),
            os.path.join(args.data_dir, 'val/label'),
            args.num_classes,
            transform=False,
            image_size=(args.img_size, args.img_size)
        )

        test_dataset = UltrasoundSegmentationDataset(
            os.path.join(args.data_dir, 'test/img'),
            os.path.join(args.data_dir, 'test/label'),
            args.num_classes,
            transform=False,
            image_size=(args.img_size, args.img_size)
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )

        return train_loader, val_loader, test_loader
