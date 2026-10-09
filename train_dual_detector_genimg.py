import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import argparse
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from datetime import datetime
import importlib

# --- Local Modules ---
from dual_model import DualStreamModel
from datasets import load_genimg_eval_datasets
from eval_utils import evaluate_basic
from utils import format_time
from model import DenoisingFCNWithSkip  # Assuming model.py contains this natively

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_parser():
    parser = argparse.ArgumentParser()
    # Data paths
    parser.add_argument("--train_dir", type=str, default="GenImage/imagenet_ai_0419_sdv4/train")
    parser.add_argument("--val_dir", type=str, default="GenImage/imagenet_ai_0419_sdv4/val")
    parser.add_argument("--eval_datasets_root", type=str, default="GenImage/valdata")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    # Logging and evaluation frequency
    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument("--save_model_freq", type=int, default=5000)
    parser.add_argument("--eval_freq", type=int, default=5000)
    parser.add_argument("--eval_batch", type=int, default=1000)

    # Models and Checkpoints for Dual-Stream
    parser.add_argument("--stream1_model", type=str, default="openai/clip-vit-large-patch14", help="Backbone for Stream 1 (CLIP)")
    parser.add_argument("--stream2_model", type=str, default="efficientnet-b0", help="Backbone for Stream 2 (Residual)")
    parser.add_argument("--lora_rank", type=int, default=4, help="Rank for LoRA on Stream 1")
    
    parser.add_argument("--resume", type=str, default=None)

    # Checkpoints for pre-trained components
    parser.add_argument("--encoder_module", type=str, default="dair.encoder2", help="AE encoder module (for augmentation)")
    parser.add_argument("--decoder_module", type=str, default="dair.decoder24_with_graph", help="AE decoder module (for augmentation)")
    parser.add_argument("--ae_resume", type=str, default=None, help="Pre-trained AE checkpoint (for augmentation)")
    parser.add_argument("--noiser_resume", type=str, default="denosier-250824225428-400000.pth", help="Pre-trained Denoiser checkpoint (for Stream 2)")

    # Augmentation logic
    parser.add_argument("--aug_target", type=str, default='all', choices=['none', 'ai', 'nature', 'all'], help="Augmentation target")
    parser.add_argument("--aug_prob", type=float, default=0.2, help="Probability of augmentation on the target")
    
    return parser

def main(params):
    # Set seeds
    torch.manual_seed(params.seed)
    np.random.seed(params.seed)
    random.seed(params.seed)

    # --- Experiment Directory Setup ---
    start_step = 1
    best_acc = 0

    if params.resume and os.path.isfile(params.resume):
        # --- Resume Mode ---
        print(f"Resume mode detected, loading checkpoint: {params.resume}")
        model_dir = os.path.dirname(params.resume)
        exp_path = os.path.dirname(model_dir)
        
        os.makedirs(model_dir, exist_ok=True)
        print(f"Resuming writing in experiment directory: {exp_path}")
        
        try:
            exp_name = os.path.basename(exp_path)
            dt_string = exp_name.split('_')[1] 
        except:
            dt_string = datetime.now().strftime("%y%m%d%H%M%S")
            print(f"Failed to parse dt_string, using current time: {dt_string}")
        
        best_model_path = os.path.join(model_dir, "best_model.pth")
        if os.path.exists(best_model_path):
            try:
                best_model_cp = torch.load(best_model_path, map_location=device)
                best_acc = best_model_cp.get("val_acc", 0)
                print(f"Loaded from {best_model_path}, best_acc set to {best_acc:.2f}%")
            except Exception as e:
                print(f"Warning: Unable to load best_acc from best_model.pth: {e}. best_acc starts at 0.")
        else:
            print("best_model.pth not found. best_acc starts at 0.")
            
    else:
        # --- New Training Mode ---
        print("No valid 'resume' provided, starting a new training.")
        now = datetime.now()
        dt_string = now.strftime("%y%m%d%H%M%S")
        
        exp_name_parts = [
            f"dualstream_{dt_string}",
            f"s1_{params.stream1_model.split('/')[-1]}-lora{params.lora_rank}",
            f"s2_{params.stream2_model}",
            f"bs{params.batch_size}",
            f"ga{params.grad_accum_steps}",
            f"lr{params.lr}",
            f"aug_{params.aug_target}{params.aug_prob}"
        ]
        exp_name = "_".join(exp_name_parts)
        
        exp_path = os.path.join("runs", exp_name)
        model_dir = os.path.join(exp_path, "models")
        os.makedirs(model_dir, exist_ok=True)

        with open(os.path.join(exp_path, 'args.txt'), 'w') as f:
            for arg in vars(params):
                f.write(f"{arg}: {getattr(params, arg)}\n")
            f.write(f"Command: {' '.join(sys.argv)}\n")
            
    # --- Data Transforms ---
    normalize_vqgan = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    
    train_transform = transforms.Compose([
        lambda img: transforms.Resize(params.img_size)(img) if min(img.size) < params.img_size else img,
        transforms.RandomCrop(params.img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize_vqgan
    ])
    
    val_transform = transforms.Compose([
        lambda img: transforms.Resize(params.img_size)(img) if min(img.size) < params.img_size else img,
        transforms.CenterCrop(params.img_size),
        transforms.ToTensor(),
        normalize_vqgan
    ])

    # --- DataLoaders ---
    print("=== Loading training and validation data ===")
    train_dataset = ImageFolder(params.train_dir, transform=train_transform)
    val_dataset = ImageFolder(params.val_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, persistent_workers=False, num_workers=8, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True, persistent_workers=False, num_workers=8, pin_memory=True, drop_last=True)
    
    print(f"Training dataset - Path: {params.train_dir}")
    print(f"Training dataset - Class mapping: {train_dataset.class_to_idx}")
    
    ai_label = train_dataset.class_to_idx.get('ai', 0)
    nature_label = train_dataset.class_to_idx.get('nature', 1)
    print(f"AI label index: {ai_label}, Nature label index: {nature_label}")

    eval_loaders, eval_info = load_genimg_eval_datasets(params.eval_datasets_root, val_transform)

    # --- Loading Prerequisite Models ---
    
    # 1. AE (for augmentation)
    encoder = None
    decoder = None
    if params.aug_target != 'none' and params.ae_resume:
        print(f"Loading AE modules: {params.encoder_module}, {params.decoder_module}")
        try:
            decoder_module = importlib.import_module(params.decoder_module)
            encoder_module = importlib.import_module(params.encoder_module)
            Decoder_class = getattr(decoder_module, "Decoder")
            Encoder_class = getattr(encoder_module, "Encoder")
            encoder = Encoder_class().to(device)
            decoder = Decoder_class().to(device)
            
            if os.path.isfile(params.ae_resume):
                print(f"Loading pre-trained AE model: {params.ae_resume}")
                checkpoint = torch.load(params.ae_resume, map_location=device)
                encoder.load_state_dict(checkpoint["encoder"])
                decoder.load_state_dict(checkpoint["decoder"])
                encoder.eval()
                decoder.eval()
            else:
                print(f"Warning: AE Checkpoint {params.ae_resume} not found. Augmentation will be disabled.")
                encoder = None
                decoder = None
        except Exception as e:
            print(f"Error loading AE: {e}. Augmentation will be disabled.")
            encoder = None
            decoder = None
    elif params.aug_target != 'none':
         print("Warning: aug_target defined but ae_resume not provided. Augmentation will be disabled.")

    # 2. Noiser (for Stream 2)
    print(f"Loading Noiser (DenoisingFCNWithSkip)")
    noiser = DenoisingFCNWithSkip().to(device)
    
    if params.noiser_resume and os.path.isfile(params.noiser_resume):
        print(f"Loading pre-trained Noiser model: {params.noiser_resume}")
        checkpoint = torch.load(params.noiser_resume, map_location=device)
        
        if "noiser" in checkpoint:
              noiser_state_dict = checkpoint["noiser"]
        elif "model_state_dict" in checkpoint:
             noiser_state_dict = checkpoint["model_state_dict"]
        else:
             noiser_state_dict = checkpoint
        
        try:
            noiser.load_state_dict(noiser_state_dict)
            print("Noiser loaded successfully.")
        except Exception as e:
            print(f"Error loading noiser state_dict: {e}. Attempting with strict=False...")
            noiser.load_state_dict(noiser_state_dict, strict=False)
            
    else:
        print(f"Warning: Noiser Checkpoint {params.noiser_resume} not found. Stream 2 will use an untrained noiser!")

    # --- Main Model Initialization ---
    model = DualStreamModel(
        stream1_model_name=params.stream1_model,
        stream2_model_name=params.stream2_model,
        noiser=noiser,
        lora_rank=params.lora_rank
    ).to(device)

    # --- Optimizer ---
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
    
    # --- Resume Logic ---
    if params.resume and os.path.isfile(params.resume):
        print(f"Loading dual-stream model checkpoint: {params.resume}")
        clf_checkpoint = torch.load(params.resume, map_location=device)
        
        try:
            model.load_state_dict(clf_checkpoint["model"], strict=False)
            print("Dual-stream model weights loaded (strict=False).")
        except Exception as e:
             print(f"Error loading model state dict: {e}")
             
        if "optimizer" in clf_checkpoint:
            try:
                optimizer.load_state_dict(clf_checkpoint["optimizer"])
                print("Optimizer state loaded successfully.")
            except Exception as e:
                print(f"Warning: Failed to load optimizer state: {e}. Using new optimizer.")

        if "step" in clf_checkpoint:
            start_step = clf_checkpoint["step"] + 1
            print(f"Resuming training from step {start_step}.")
    else:
        print("Starting a new training (no 'resume' checkpoint provided).")
        print("Stream 1 (CLIP) uses pre-trained weights + random LoRA.")
        print("Stream 2 (EffNet) uses ImageNet pre-trained weights.")
        print("Stream 2 (Noiser) uses noiser_resume checkpoint.")
        print("The fusion head is randomly initialized.")

    # --- Training Loop ---
    criterion_unbalanced = nn.BCEWithLogitsLoss()

    start_time = time.time()
    data_iter = iter(train_loader)

    print("\n=== Starting Training (Dual-Stream Mode) ===")
    
    for step in range(start_step, params.steps + 1):
        model.train() 
        
        try:
            imgs, original_labels = next(data_iter)
        except:
            data_iter = iter(train_loader)
            imgs, original_labels = next(data_iter)

        imgs, original_labels = imgs.to(device, non_blocking=True), original_labels.to(device, non_blocking=True)
        
        augmented_imgs = imgs.clone()
        is_augmented = torch.zeros_like(original_labels, dtype=torch.bool, device=device)

        # --- 1. Data Augmentation (REQ 1) ---
        if encoder and decoder and params.aug_target != 'none' and params.aug_prob > 0:
            with torch.no_grad():
                aug_mask = torch.zeros_like(original_labels, dtype=torch.bool, device=device)
                prob_mask = torch.rand(imgs.size(0), device=device) < params.aug_prob

                if params.aug_target == 'ai':
                    aug_mask = (original_labels == ai_label) & prob_mask
                elif params.aug_target == 'nature':
                    aug_mask = (original_labels == nature_label) & prob_mask
                elif params.aug_target == 'all':
                    aug_mask = prob_mask

                if aug_mask.any():
                    imgs_to_aug = imgs[aug_mask]
                    latents = encoder(imgs_to_aug)
                    recon_imgs = decoder(latents)
                    augmented_imgs[aug_mask] = recon_imgs
                    is_augmented[aug_mask] = True

        # --- 2. New Labels Logic ---
        # Positive (1): Original Nature (non-augmented)
        # Negative (0): Original AI, Augmented AI, Augmented Nature
        final_labels = torch.zeros_like(original_labels, device=device)
        final_labels[(original_labels == nature_label) & (~is_augmented)] = 1
        
        # --- 3. Forward Pass (Dual-Stream) ---
        outputs = model(augmented_imgs).squeeze()
        
        # --- 4. Loss Calculation ---
        loss = criterion_unbalanced(outputs, final_labels.float())
            
        loss = loss / params.grad_accum_steps
        loss.backward()

        if step % params.grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # --- Logging ---
        if step % params.log_freq == 0:
            with torch.no_grad():
                probs = torch.sigmoid(outputs)
                predicted = (probs > 0.5).long()
                acc = (predicted == final_labels).float().mean().item()
            
            duration = time.time() - start_time
            log_msg = f"{dt_string} [{step:07d}] Loss: {loss.item() * params.grad_accum_steps:.4f} Acc(train_logic): {acc*100:.2f} Time: {format_time(duration)}"
            print(log_msg)
            
            with open(os.path.join(exp_path, 'logs.txt'), 'a') as f:
                f.write(log_msg + "\n")

        # --- Evaluation (REQ 7) ---
        if step % params.eval_freq == 0:
            val_acc = evaluate_basic(model, val_loader, device, params.eval_batch)
            print(f"Validation Accuracy (Original Task): {val_acc:.2f}%")
            
            with open(os.path.join(exp_path, 'eval_results.txt'), 'a') as f:
                f.write(f"Step {step:07d} - Val Acc (Original Task): {val_acc:.2f}%\n")

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "val_acc": val_acc
                }, os.path.join(model_dir, "best_model.pth"))
                print(f"*** New best model saved (Acc: {val_acc:.2f}%) ***")

            if eval_loaders:
                print(f"\n--- Multi-Dataset Evaluation (Step {step}) ---")
                
                with open(os.path.join(exp_path, 'eval_results.txt'), 'a') as f:
                    f.write(f"--- Step {step} Multi-Dataset ---\n")
                    
                for dataset_name, eval_loader in eval_loaders.items():
                    eval_acc = evaluate_basic(model, eval_loader, device, params.eval_batch)
                    print(f"{dataset_name}: {eval_acc:.2f}%")
                    with open(os.path.join(exp_path, 'eval_results.txt'), 'a') as f:
                        f.write(f"{dataset_name}: {eval_acc:.2f}%\n")
                
                with open(os.path.join(exp_path, 'eval_results.txt'), 'a') as f:
                    f.write("\n")
                print("--- Multi-Dataset Evaluation Completed ---\n")

            model.train() 

        # --- Checkpoint Saving ---
        if step % params.save_model_freq == 0:
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step
            }, os.path.join(model_dir, f"checkpoint-{step:06d}.pth"))

    print(f"Training completed, best validation accuracy (Original Task): {best_acc:.2f}%")

if __name__ == "__main__":   
    parser = get_parser()
    params = parser.parse_args()
        
    main(params)
