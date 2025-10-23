import torch  
import torch.nn as nn  
import torch.optim as optim  
import os  
import random  
import numpy as np  
import argparse  
from tensorboardX import SummaryWriter  
from datetime import datetime 

from utils.evaluate import Evaluator  
from utils.data_processing import DataProcessor  
from utils.logger import setup_logger 
from utils.load_model import load_model
from utils.schedule import build_scheduler, get_lr_decay_param_groups 


def parse_args():  
    parser = argparse.ArgumentParser(description='hyperparameters')  
    parser.add_argument('--model_name', type=str, default='TinyUSFM', help='name of the model')  
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=400, help='number of epochs')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers')
    parser.add_argument('--num_classes', type=int, default=2, help='number of classes')
    parser.add_argument('--pretrained', type=str, default='True', help='pretrained')
    parser.add_argument('--data_dir', type=str, default='./data/Cls/tn3k', help='data directory')
    parser.add_argument('--img_size', type=int, default=224, help='image size')
    parser.add_argument('--wd', type=float, default=0, help='weight decay')
    parser.add_argument('--optim', type=str, default='AdamW', help='optimizer')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/TinyUSFM.pth', help='tiny usfm checkpoint')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='warmup epochs')
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
    base_logs_dir = f"./logs/cls/{dataset_name}/{args.model_name}_{'pretrained' if pretrained else 'scratch'}"  
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
        param_groups = get_lr_decay_param_groups(
            model=model,
            base_lr=args.lr,
            weight_decay=args.wd,
            num_layers=12, 
            layer_decay=0.8
        )
        optimizer = optim.AdamW(param_groups, betas=(0.9, 0.999))
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