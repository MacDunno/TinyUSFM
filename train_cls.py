import torch  
import torch.nn as nn  
import torch.optim as optim  
import os  
import random  
import numpy as np   
from utils.evaluate import Evaluator  
from utils.data_processing import DataProcessor  
import argparse  
from tensorboardX import SummaryWriter  
from datetime import datetime 
from utils.logger import setup_logger 
from utils.load_model import load_model
from utils.schedule_seg import build_scheduler 


def get_layerwise_lr_decay_param_groups(model, base_lr, weight_decay, num_layers=12, layer_decay=0.8):
    def get_layer_id(param_name):
        if param_name.startswith("backbone"):
            if "blocks." in param_name:
                block_id = int(param_name.split("blocks.")[1].split(".")[0])
                return block_id
            elif "patch_embed" in param_name:
                return 0
            else:
                return num_layers - 1
        else:
            return num_layers 

    param_groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        layer_id = get_layer_id(name)
        group_name = f"layer_{layer_id}"

        if group_name not in param_groups:
            scale = layer_decay ** (num_layers - layer_id)
            param_groups[group_name] = {
                "params": [],
                "lr": base_lr * scale,
                "weight_decay": weight_decay
            }

        param_groups[group_name]["params"].append(param)

    return list(param_groups.values())


def parse_args():  
    parser = argparse.ArgumentParser(description='hyperparameters')  
    parser.add_argument('--model_name', type=str, default='TinyUSFM', help='name of the model')  
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers')
    parser.add_argument('--num_classes', type=int, default=2, help='number of classes')
    parser.add_argument('--pretrained', type=str, default='True', help='pretrained')
    parser.add_argument('--data_dir', type=str, default='./data/Cls/tn3k', help='data directory')
    parser.add_argument('--img_size', type=int, default=224, help='image size')
    parser.add_argument('--wd', type=float, default=0, help='weight decay')
    parser.add_argument('--optim', type=str, default='AdamW', help='optimizer')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/TinyUSFM.pth', help='tiny usfm checkpoint')
    parser.add_argument('--lr_decay', type=str, default='cosine', help='learning rate decay')
    parser.add_argument('--layerwise', action='store_true', help='Use layer-wise learning rate decay')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='warmup epochs')
    parser.add_argument('--ls_name', type=str, required=True, help='experiment name for logging')
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
    set_seed(42) 
    pretrained = args.pretrained == 'True'

    dataset_name = os.path.basename(args.data_dir.rstrip('/')) 
    base_logs_dir = f"./logs/cls/{args.ls_name}/{dataset_name}/{args.model_name}_{'pretrained' if pretrained else 'scratch'}"  
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")  
    current_logs_dir = os.path.join(base_logs_dir, current_time)  
    os.makedirs(current_logs_dir, exist_ok=True)  
    models_dir = os.path.join(current_logs_dir, "models")  
    os.makedirs(models_dir, exist_ok=True)

    logger = setup_logger(current_logs_dir)
    logger.info(f"Pretrained: {pretrained}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    logger.info(f"Using device: {device}")  
      
    train_loader, val_loader, _ = DataProcessor.get_data_loaders(args)  

    model = load_model(args, device) 
    logger.info("Illustration of model strcutures:\n{}".format(str(model)))  

    total_params = sum(p.numel() for p in model.parameters())  
    logger.info(f'Total parameters: {total_params:,}')  

    criterion = nn.CrossEntropyLoss()  

    if args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)  
    elif args.optim == 'Sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)
    elif args.optim == 'AdamW':
        if args.layerwise:
            param_groups = get_layerwise_lr_decay_param_groups(
                model=model,
                base_lr=args.lr,
                weight_decay=args.wd,
                num_layers=12, 
                layer_decay=0.8
            )
            optimizer = optim.AdamW(param_groups, betas=(0.9, 0.999))
        else:
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        raise ValueError('Optimizer not supported')
    
    scheduler = build_scheduler(optimizer, args)

    hyperparams_path = os.path.join(current_logs_dir, "hyperparameters.txt")  
    with open(hyperparams_path, "w") as f:  
        f.write("Arguments:\n")  
        for arg, value in vars(args).items():  
            f.write(f"{arg}: {value}\n")  

    writer = SummaryWriter(os.path.join(current_logs_dir, "tensorboard"))  
    
    best_val_acc = 0.0  
    evaluator = Evaluator()  

    for epoch in range(args.num_epochs): 
        if epoch < args.warmup_epochs:  
            for param_group in optimizer.param_groups:  
                param_group['lr'] = args.lr * (epoch + 1) / args.warmup_epochs
        
        model.train()  
        running_loss = 0.0  
        
        for inputs, labels in train_loader:  
            inputs, labels = inputs.to(device), labels.to(device)  
            optimizer.zero_grad()  
            outputs = model(inputs)  
            loss = criterion(outputs, labels)  
            loss.backward()  
            optimizer.step()  
            
            running_loss += loss.item() * inputs.size(0)  
        
        epoch_loss = running_loss / len(train_loader.dataset)  
  
        if epoch >= args.warmup_epochs:  
            scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']

        model.eval()  
        val_loss = 0.0  
        with torch.no_grad():  
            for inputs, labels in val_loader:  
                inputs, labels = inputs.to(device), labels.to(device)  
                outputs = model(inputs)  
                loss = criterion(outputs, labels)  
                val_loss += loss.item() * inputs.size(0)  
        
        val_loss /= len(val_loader.dataset)  

        metrics, _, _ = evaluator.evaluate_model(model, val_loader, device)  

        logger.info(f'\nEpoch {epoch+1}/{args.num_epochs}: Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}')  
        evaluator.print_metrics(metrics, phase='validation')  
 
        writer.add_scalar('Loss/train', epoch_loss, epoch + 1)   
        writer.add_scalar('Loss/validation', val_loss, epoch + 1)  
        writer.add_scalar('Accuracy/validation', metrics['accuracy'], epoch + 1)  
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch + 1)
        writer.add_scalar('AUC/validation', metrics['auc'], epoch + 1)  
        writer.add_scalar('AUC/validation', metrics['recall'], epoch + 1)  
        
        val_acc = metrics['accuracy'] 
        if val_acc > best_val_acc:  
            best_val_acc = val_acc  
            best_model_path = os.path.join(  
                models_dir,  
                f"epoch{epoch+1}_val_{val_acc:.4f}.pth"  
            )  
            torch.save(model.state_dict(), best_model_path)  
            logger.info(f'New best model saved at: {best_model_path} | Val ACC: {best_val_acc:.4f}')
    writer.close()  

if __name__ == '__main__':  
    args = parse_args()  
    train_model(args)