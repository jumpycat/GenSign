import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

def evaluate_basic(model, val_loader, device, max_batches=None):
    """
    Evaluates the DualStreamModel and returns accuracy.
    """
    model.eval() # Sets the entire model (including backbones) to eval mode
    correct = 0
    total = 0
    batch_count = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            if max_batches and batch_count >= max_batches:
                break
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            outputs = model(imgs).squeeze()
            probs = torch.sigmoid(outputs)
            predicted = (probs > 0.5).long()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            batch_count += 1
    
    return 100 * correct / total

def evaluate_advanced(model, val_loader, device, max_batches=None, dataset_name=""):
    """
    Evaluates the DualStreamModel and returns (acc, auc, ap).
    """
    model.eval()
    correct = 0
    total = 0
    batch_count = 0
    
    all_labels = []
    all_probs = []

    with torch.no_grad():
        pbar_desc = f"  > Evaluating {dataset_name}"
        pbar = tqdm(val_loader, desc=pbar_desc, leave=False, unit="batch")
        
        for imgs, labels in pbar:
            if max_batches and batch_count >= max_batches:
                break
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            outputs = model(imgs).squeeze()
            probs = torch.sigmoid(outputs)
            predicted = (probs > 0.5).long()
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())
            
            batch_count += 1
    
    acc = 100 * correct / total
    auc = -1.0
    ap = -1.0
    
    try:
        all_labels_np = torch.cat(all_labels).numpy()
        all_probs_np = torch.cat(all_probs).numpy()
        
        if len(np.unique(all_labels_np)) > 1:
            auc = roc_auc_score(all_labels_np, all_probs_np) * 100.0
            ap = average_precision_score(all_labels_np, all_probs_np) * 100.0
        else:
            print(f"  (Warning: {dataset_name} has only one class, AUC/AP not calculated)")
            
    except Exception as e:
        print(f"  (Warning: Failed to calculate AUC/AP for {dataset_name}: {e})")

    return acc, auc, ap