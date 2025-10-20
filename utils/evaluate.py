import torch
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_score, recall_score, f1_score, balanced_accuracy_score, roc_auc_score, mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error, roc_curve, auc
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import pandas as pd
from sklearn.preprocessing import label_binarize
from medpy.metric.binary import hd95, dc, precision, recall, jc, hd


logger = logging.getLogger(__name__)

class Evaluator:
    @staticmethod
    def evaluate_model(model: torch.nn.Module,
                      data_loader: torch.utils.data.DataLoader,
                      device: torch.device) -> Tuple[Dict[str, float], List[int], List[int]]:
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                num_classes = probs.size(1)
                
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        metrics = {}

        if num_classes == 2:  
            try:  
                auc = roc_auc_score(all_labels, [p[1] for p in all_probs]) 
            except Exception as e:  
                logging.warning(f"AUC calculation failed in binary classification: {e}")  
                auc = 0.0  
            
            metrics['accuracy'] = accuracy_score(all_labels, all_preds)
            metrics['precision'] = precision_score(all_labels, all_preds, pos_label=1, average='binary')  
            metrics['recall'] = recall_score(all_labels, all_preds, pos_label=1, average='binary')  
            metrics['f1'] = f1_score(all_labels, all_preds, pos_label=1, average='binary')  
            metrics['auc'] = auc  
            metrics['mcc'] = matthews_corrcoef(all_labels, all_preds) 
            metrics['b_accuracy'] = balanced_accuracy_score(all_labels, all_preds) 

        elif num_classes > 2:  
            try:  
                auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro') 
                 
            except Exception as e:  
 
                logging.warning(f"AUC calculation failed in multi-class classification: {e}")  
                auc = 0.0  
            
            metrics['accuracy'] = accuracy_score(all_labels, all_preds)
            metrics['precision'] = precision_score(all_labels, all_preds, average='macro')  
            metrics['recall'] = recall_score(all_labels, all_preds, average='macro')  
            metrics['f1'] = f1_score(all_labels, all_preds, average='macro')  
            metrics['auc'] = auc  
            metrics['mcc'] = matthews_corrcoef(all_labels, all_preds)
            metrics['b_accuracy'] = balanced_accuracy_score(all_labels, all_preds)
        else: 
            raise ValueError(f"Invalid number of classes: {num_classes}. Check the model output.")  
        
        return metrics, all_preds, all_labels

    @staticmethod
    def print_metrics(metrics: Dict[str, float], phase: str = 'Validation') -> None:
        logger.info(f'{phase} Metrics: ACC: {metrics["accuracy"]:.4f}, Precision: {metrics["precision"]:.4f}, Recall: {metrics["recall"]:.4f}, F1: {metrics["f1"]:.4f}, AUC: {metrics["auc"]:.4f}, MCC: {metrics["mcc"]:.4f}, BACC: {metrics["b_accuracy"]:.4f}')

    @staticmethod
    def generate_classification_report(labels: List[int], 
                                    predictions: List[int], 
                                    save_dir: str) -> None:
        os.makedirs(save_dir, exist_ok=True)
        report = classification_report(labels, predictions)
        with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
            f.write(report)
        logger.info("\nClassification Report:")
        logger.info(report)
        cm = confusion_matrix(labels, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
        plt.close()

    @staticmethod
    def save_predictions(image_names: List[str], 
                        predictions: List[int], 
                        labels: List[int],
                        save_path: str) -> None:
        
        file_names = [os.path.basename(img_name) for img_name in image_names]  
        
        correctness = ["Correct" if pred == label else "Wrong" for pred, label in zip(predictions, labels)]  
        
        df = pd.DataFrame({  
            "image_name": file_names,  
            "prediction": predictions,  
            "label": labels,  
            "correct": correctness  
        })  
        
        df.to_excel(save_path, index=False) 

        print(f"Prediction save to {save_path}") 


class Evaluator_seg:
    @staticmethod
    def evaluate_model(model: torch.nn.Module,
                       data_loader: torch.utils.data.DataLoader,
                       device: torch.device,
                       num_classes: int = 2,
                       threshold: float = 0.5) -> Dict[str, float]:
        model.eval()
        if num_classes == 2:
            return Evaluator_seg._evaluate_binary(model, data_loader, device, threshold)
        else:
            return Evaluator_seg._evaluate_multiclass(model, data_loader, device, num_classes)

    @staticmethod
    def _evaluate_binary(model, data_loader, device, threshold):
        dice_list = []
        hd95_list = []
        iou_list = []
        sensitivity_list = []
        specificity_list = []
        pixel_acc_list = []

        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(device)
                labels = labels.to(device)  # [B, H, W], ground truth mask (binary)

                outputs = model(images)  # [B, 1, H, W] (sigmoid or logits)

                if outputs.shape[1] == 1:
                    probs = torch.sigmoid(outputs)  # binary: sigmoid
                    preds = (probs > threshold).float()
                else:
                    raise ValueError("Only binary segmentation supported for now.")

                for pred, gt in zip(preds, labels):
                    pred_np = pred.squeeze().cpu().numpy().astype(bool)
                    gt_np = gt.squeeze().cpu().numpy().astype(bool)

                    dice = dc(pred_np, gt_np)
                    if pred_np.any() and gt_np.any():
                        hausdorff = hd95(pred_np, gt_np)
                    elif not pred_np.any() and not gt_np.any():
                        hausdorff = 0
                    else:
                        hausdorff = 224
                    iou = jc(pred_np, gt_np)
                    sens = recall(pred_np, gt_np)
                    spec = Evaluator_seg.compute_specificity(pred_np, gt_np)
                    pixel_acc = (pred_np == gt_np).sum() / gt_np.size

                    dice_list.append(dice)
                    hd95_list.append(hausdorff)
                    iou_list.append(iou)
                    sensitivity_list.append(sens)
                    specificity_list.append(spec)
                    pixel_acc_list.append(pixel_acc)

        metrics = {
            'Dice': np.mean(dice_list),
            'Dice_std': np.std(dice_list),
            'HD95': np.mean(hd95_list),
            'HD95_std': np.std(hd95_list),
            'IoU': np.mean(iou_list),
            'IoU_std': np.std(iou_list),
            'Sensitivity': np.mean(sensitivity_list),
            'Sensitivity_std': np.std(sensitivity_list),
            'Specificity': np.mean(specificity_list),
            'Specificity_std': np.std(specificity_list),
            'PixelAcc': np.mean(pixel_acc_list),
            'PixelAcc_std': np.std(pixel_acc_list)
        }

        return metrics
    
    @staticmethod
    def _evaluate_multiclass(model, data_loader, device, num_classes):
        dice_per_class = [[] for _ in range(num_classes)]
        iou_per_class = [[] for _ in range(num_classes)]
        pixel_acc_list = []
        sensitivity_per_class = [[] for _ in range(num_classes)]
        specificity_per_class = [[] for _ in range(num_classes)]
        hd95_per_class = [[] for _ in range(num_classes)]

        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(device)
                labels = labels.to(device)  # [B, H, W]

                outputs = model(images)  # [B, num_classes, H, W]
                preds = torch.argmax(outputs, dim=1)  # [B, H, W]

                for pred, gt in zip(preds, labels):
                    pred_np = pred.cpu().numpy()
                    gt_np = gt.cpu().numpy()

                    pixel_acc = (pred_np == gt_np).mean()
                    pixel_acc_list.append(pixel_acc)

                    for class_id in range(num_classes):
                        pred_class = (pred_np == class_id)
                        gt_class = (gt_np == class_id)

                        if not (gt_class.any() or pred_class.any()):
                            dice = 1.0
                            iou = 1.0
                            sensitivity = np.nan
                            specificity = np.nan
                            hd = np.nan
                        else:
                            intersection = np.logical_and(pred_class, gt_class).sum()
                            union = np.logical_or(pred_class, gt_class).sum()
                            dice = 2 * intersection / (pred_class.sum() + gt_class.sum() + 1e-8)
                            iou = intersection / (union + 1e-8)
                            tp = intersection
                            fn = np.logical_and(~pred_class, gt_class).sum()
                            if (tp + fn) > 0:
                                sensitivity = tp / (tp + fn + 1e-8)
                            else:
                                sensitivity = np.nan
                            tn = np.logical_and(~pred_class, ~gt_class).sum()
                            fp = np.logical_and(pred_class, ~gt_class).sum()
                            if (tn + fp) > 0:
                                specificity = tn / (tn + fp + 1e-8)
                            else:
                                specificity = np.nan
                            if gt_class.any() and pred_class.any():
                                hd = hd95(pred_class.astype(np.bool_), gt_class.astype(np.bool_))
                            else:
                                hd = 224

                        dice_per_class[class_id].append(dice)
                        iou_per_class[class_id].append(iou)
                        sensitivity_per_class[class_id].append(sensitivity)
                        specificity_per_class[class_id].append(specificity)
                        hd95_per_class[class_id].append(hd)

            foreground_ids = list(range(1, num_classes))

            metrics = {}
            metrics['PixelAcc'] = np.nanmean(pixel_acc_list)
            metrics['PixelAcc_std'] = np.nanstd(pixel_acc_list)
            metrics['Dice'] = np.nanmean([np.nanmean(dice_per_class[i]) for i in foreground_ids])
            metrics['Dice_std'] = np.nanstd([item for i in foreground_ids for item in dice_per_class[i]])
            metrics['IoU'] = np.nanmean([np.nanmean(iou_per_class[i]) for i in foreground_ids])
            metrics['IoU_std'] = np.nanstd([item for i in foreground_ids for item in iou_per_class[i]])
            metrics['Sensitivity'] = np.nanmean([np.nanmean(sensitivity_per_class[i]) for i in foreground_ids])
            metrics['Sensitivity_std'] = np.nanstd([item for i in foreground_ids for item in sensitivity_per_class[i]])
            metrics['Specificity'] = np.nanmean([np.nanmean(specificity_per_class[i]) for i in foreground_ids])
            metrics['Specificity_std'] = np.nanstd([item for i in foreground_ids for item in specificity_per_class[i]])
            metrics['HD95'] = np.nanmean([np.nanmean(hd95_per_class[i]) for i in foreground_ids])
            metrics['HD95_std'] = np.nanstd([item for i in foreground_ids for item in hd95_per_class[i]])
        
        return metrics

    @staticmethod
    def compute_specificity(pred: np.ndarray, gt: np.ndarray) -> float:
        tn = np.logical_and(pred == 0, gt == 0).sum()
        fp = np.logical_and(pred == 1, gt == 0).sum()
        return tn / (tn + fp + 1e-8)

    @staticmethod
    def print_metrics(metrics: Dict[str, float], phase: str = 'Validation') -> None:
        if phase.lower() == 'test':
            logger.info(f'{phase} Metrics - Dice: {metrics["Dice"]:.4f}±{metrics["Dice_std"]:.4f}, '
                        f'HD95: {metrics["HD95"]:.2f}±{metrics["HD95_std"]:.2f}, '
                        f'IOU: {metrics["IoU"]:.4f}±{metrics["IoU_std"]:.4f}, '
                        f'Sensitivity: {metrics["Sensitivity"]:.4f}±{metrics["Sensitivity_std"]:.4f}, '
                        f'Specificity: {metrics["Specificity"]:.4f}±{metrics["Specificity_std"]:.4f}, '
                        f'PixelAcc: {metrics["PixelAcc"]:.4f}±{metrics["PixelAcc_std"]:.4f}')
        else:
            logger.info(f'{phase} Metrics - Dice: {metrics["Dice"]:.4f}, '
                        f'HD95: {metrics["HD95"]:.2f}, IOU: {metrics["IoU"]:.4f}, Sensitivity: {metrics["Sensitivity"]:.4f}, '
                        f'Specificity: {metrics["Specificity"]:.4f}, PixelAcc: {metrics["PixelAcc"]:.4f}')
        
    @staticmethod
    def save_visual_results(images: torch.Tensor,
                            preds: torch.Tensor,
                            gts: torch.Tensor,
                            save_dir: str,
                            file_names: List[str]) -> None:
        os.makedirs(save_dir, exist_ok=True)

        images = images.cpu().permute(0, 2, 3, 1).numpy()
        preds = preds.cpu().squeeze(1).numpy()
        gts = gts.cpu().numpy()

        for i in range(len(images)):
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(images[i], cmap='gray')
            axs[0].set_title('Image')
            axs[1].imshow(preds[i], cmap='gray')
            axs[1].set_title('Prediction')
            axs[2].imshow(gts[i], cmap='gray')
            axs[2].set_title('Ground Truth')
            for ax in axs:
                ax.axis('off')

            fname = os.path.splitext(file_names[i])[0]
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'{fname}_result.png'))
            plt.close()