import torch  
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms  
from PIL import Image  
import os  
import logging  
import random  
import numpy as np  
from typing import Tuple, Optional, List  


logger = logging.getLogger(__name__)  

def set_seed(seed=42):  
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False  

class CustomImageDataset(Dataset):  
    def __init__(self, root_dir: str, transform: Optional[transforms.Compose] = None, balance_classes: bool = False):  
        self.root_dir = root_dir  
        self.transform = transform  
        self.classes = sorted(os.listdir(root_dir))  # class1, class2  
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}  
        
        self.samples = []  
        for class_name in self.classes:  
            class_dir = os.path.join(root_dir, class_name)  
            for img_name in os.listdir(class_dir):  
                if img_name.endswith(('.png', '.jpg', '.jpeg', '.JPG')):  
                    self.samples.append((  
                        os.path.join(class_dir, img_name),  
                        self.class_to_idx[class_name]  
                    ))  
        logger.info(f"Loaded {len(self.samples)} samples from {root_dir}, have {len(self.classes)} classes")  

        # 如果需要平衡类别  
        if balance_classes:  
            self.samples = self._balance_classes()
    
    def _balance_classes(self) -> List[Tuple[str, int]]: 
        from collections import defaultdict  
        import random  

        class_samples = defaultdict(list)  
        for img_path, label in self.samples:  
            class_samples[label].append((img_path, label))  
        negative_label = 0  
        positive_label = 1  
        positive_samples = class_samples[positive_label]  
        negative_samples = class_samples[negative_label]  
        positive_count = len(positive_samples)  
        if len(negative_samples) > positive_count:  
            negative_samples = random.sample(negative_samples, positive_count)  
        balanced_samples = positive_samples + negative_samples  

        logger.info(f"Balanced dataset to have {len(positive_samples)} samples per class.")  
        return balanced_samples
    
    def __len__(self) -> int:  
        return len(self.samples)  
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:  
        img_path, label = self.samples[idx]  
        image = Image.open(img_path).convert('RGB')  
        if self.transform:  
            image = self.transform(image)  
        return image, label  

class DataProcessor:  
    @staticmethod  
    def get_transform(args, is_training: bool = True) -> transforms.Compose:  
        transform_list = []  
        
        if is_training:
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=30),
            ])

        transform_list.extend([  
            transforms.Resize([args.img_size, args.img_size]),  
        ])  
        
        transform_list.extend([  
            transforms.ToTensor(),  
            transforms.Normalize(  
                mean=[0.485, 0.456, 0.406],  
                std=[0.229, 0.224, 0.225] 
            ),  
        ])  
        
        return transforms.Compose(transform_list)  

    @staticmethod  
    def get_data_loaders(args) -> Tuple[DataLoader, DataLoader, DataLoader]:  
        # set_seed(42)  
        def worker_init_fn(worker_id):  
            seed = 42 + worker_id  
            np.random.seed(seed)  
            random.seed(seed)  
            torch.manual_seed(seed)  
        
        train_transform = DataProcessor.get_transform(args, is_training=True)  
        val_transform = DataProcessor.get_transform(args, is_training=False)  
        
        train_dataset = CustomImageDataset(  
            os.path.join(args.data_dir, 'train'),  
            transform=train_transform,
            balance_classes=False
        )  
        
        val_dataset = CustomImageDataset(  
            os.path.join(args.data_dir, 'val'), 
            transform=val_transform  
        )  
        
        test_dataset = CustomImageDataset(  
            os.path.join(args.data_dir, 'test'), 
            transform=val_transform  
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

    @staticmethod  
    def verify_image_files(images_dir: str, file_list: str) -> List[str]:  
        error_files = []  
        
        with open(file_list, 'r') as f:  
            for line in f:  
                img_name, _ = line.strip().split()  
                img_path = os.path.join(images_dir, img_name)  
                
                if not os.path.exists(img_path):  
                    error_files.append(img_path)  
                else:  
                    try:  
                        with Image.open(img_path) as img:  
                            img.verify()  
                    except Exception as e:  
                        error_files.append(f"{img_path}: {str(e)}")  
        
        return error_files
