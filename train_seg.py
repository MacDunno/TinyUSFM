import torch
import torch.nn as nn
import torch.optim as optim
import os
import random
import numpy as np
import argparse
from datetime import datetime
from tensorboardX import SummaryWriter

from utils.data_processing_seg import DataProcessor
from utils.evaluate import Evaluator_seg
from utils.logger import setup_logger
from utils.load_model_seg import load_model_seg
from utils.schedule import build_scheduler,get_lr_decay_param_groups 


def parse_args():
    parser = argparse.ArgumentParser(description='Segmentation Training')
    parser.add_argument('--model_name', type=str, default='TinyUSFM_Seg')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--data_dir', type=str, default='./data/Seg/CAMUS')
    parser.add_argument('--pretrained', type=str, default='True')
    parser.add_argument('--optim', type=str, default='AdamW')
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--checkpoint', type=str, default='checkpoints/TinyUSFM.pth')
    parser.add_argument('--num_classes', type=int, default=4) 
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--task_type', type=str, default='organ', help='Task type: tumor or organ')
    args = parser.parse_args()
    return args


def set_seed(seed=42):
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False  


def train_model(args):
    pretrained = args.pretrained == 'True'
    dataset_name = os.path.basename(args.data_dir.rstrip('/'))
    base_log_dir = f"./logs/seg/{dataset_name}/{args.model_name}_{'pretrained' if pretrained else 'scratch'}"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(base_log_dir, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    models_dir = os.path.join(log_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    logger = setup_logger(log_dir)
    logger.info(f"Using pretrained: {pretrained}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    train_loader, val_loader, _ = DataProcessor.get_data_loaders(args)

    model = load_model_seg(args, device)
    logger.info(f"Model structure:\n{model}")
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'Total parameters: {total_params:,}')

    if args.num_classes == 2:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss() 

    if args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    elif args.optim == 'AdamW':
        param_groups = get_lr_decay_param_groups(
            model=model,
            base_lr=args.lr,
            weight_decay=args.wd,
            num_layers=12,
            layer_decay=0.8
        )
        optimizer = optim.AdamW(param_groups, betas=(0.9, 0.999))
    else:
        raise ValueError("Unsupported optimizer")

    scheduler = build_scheduler(optimizer, args)
 
    hyperparams_path = os.path.join(log_dir, "hyperparameters.txt")  
    with open(hyperparams_path, "w") as f:  
        f.write("Arguments:\n")  
        for arg, value in vars(args).items():  
            f.write(f"{arg}: {value}\n") 

    writer = SummaryWriter(os.path.join(log_dir, "tensorboard"))

    best_dice = 0.0
    evaluator = Evaluator_seg()

    for epoch in range(args.num_epochs):
        # Warmup
        if epoch < args.warmup_epochs:
            lr = args.lr * (epoch + 1) / args.warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device).float()
            outputs = model(images)  # [B, num_classes, H, W]
            if args.num_classes == 2:
                loss = criterion(outputs, masks.unsqueeze(1))
            else:
                loss = criterion(outputs, masks.long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        if epoch >= args.warmup_epochs:
            scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device).float()
                outputs = model(images)
                if args.num_classes == 2:
                    loss = criterion(outputs, masks.unsqueeze(1))
                else:
                    loss = criterion(outputs, masks.long()) 

                val_loss += loss.item() * images.size(0)

        val_loss /= len(val_loader.dataset)
        metrics = evaluator.evaluate_model(model, val_loader, device, args.num_classes)

        logger.info(
            f"\nEpoch {epoch+1}/{args.num_epochs} | "
            f"Train Loss: {epoch_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f" | LR: {current_lr:.6f}"
        )
        evaluator.print_metrics(metrics, phase='validation')  

        writer.add_scalar('Loss/train', epoch_loss, epoch + 1)
        writer.add_scalar('Loss/val', val_loss, epoch + 1)
        writer.add_scalar('Dice/val', metrics['Dice'], epoch + 1)
        writer.add_scalar('HD95/val', metrics['HD95'], epoch + 1)
        writer.add_scalar('PixelAcc/val', metrics['PixelAcc'], epoch + 1)
        writer.add_scalar('LearningRate', current_lr, epoch + 1)

        if metrics['Dice'] > best_dice:
            best_dice = metrics['Dice']
            model_path = os.path.join(models_dir, f"epoch{epoch+1}_dice{best_dice:.4f}.pth")
            torch.save(model.state_dict(), model_path)
            logger.info(f"Saved best model to {model_path}")
    writer.close()


if __name__ == '__main__':
    set_seed()
    args = parse_args()
    train_model(args)