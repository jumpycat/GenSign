# HF_ENDPOINT a-t-il été défini ? (De votre premier script)
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
from datetime import datetime, timedelta
import importlib # Requis pour AE
from collections import defaultdict

# --- Importations des deux scripts ---
# De Script 1 (Baseline)
from transformers import CLIPModel
import loralib as lora

# De Script 2 (Merged)
from efficientnet_pytorch import EfficientNet
from model import DenoisingFCNWithSkip  # Assurez-vous que model.py est accessible
import timm
import torchvision.models as models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def denorm(x):
    """ Dénormalise de VQGAN [-1, 1] à [0, 1] """
    return (x + 1.0) / 2.0

def format_time(seconds):
    delta = timedelta(seconds=seconds)
    days = delta.days
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{days}d {hours}h {minutes}m {seconds}s'

# --- NOUVEAU : Fonction d'assistance LoRA (de Script 1) ---
def replace_with_lora(module, rank):
    """ Remplace une couche nn.Linear par une couche lora.Linear et copie les poids. """
    lora_layer = lora.Linear(
        module.in_features,
        module.out_features,
        r=rank,
        lora_alpha=rank, 
        lora_dropout=0.1, # Peut être ajusté
        bias=(module.bias is not None)
    )
    # Copier les poids et le biais d'origine
    lora_layer.weight.data.copy_(module.weight.data)
    if module.bias is not None:
        lora_layer.bias.data.copy_(module.bias.data)
    return lora_layer

# --- NOUVEAU : Modèle Dual-Stream ---
class DualStreamModel(nn.Module):
    def __init__(self, stream1_model_name, stream2_model_name, noiser, lora_rank, clip_input_size=224):
        """
        Initialise le modèle dual-stream.
        
        Args:
            stream1_model_name (str): Nom du modèle CLIP (ex: 'openai/clip-vit-large-patch14').
            stream2_model_name (str): Nom du modèle EfficientNet (ex: 'efficientnet-b0').
            noiser (nn.Module): Instance pré-entraînée du DenoisingFCNWithSkip.
            lora_rank (int): Rang à utiliser pour LoRA sur le backbone CLIP.
            clip_input_size (int): Taille d'entrée pour le backbone CLIP (ex: 224).
        """
        super(DualStreamModel, self).__init__()

        print(f"Initializing Dual-Stream Model:")
        
        # --- Stream 1: CLIP + LoRA ---
        print(f"  Stream 1: Loading {stream1_model_name}...")
        try:
            clip_model = CLIPModel.from_pretrained(stream1_model_name)
            self.clip_backbone = clip_model.vision_model
            # Obtenir la dim de manière robuste
            self.clip_feat_dim = clip_model.config.vision_config.hidden_size 
        except Exception as e:
            print(f"ERREUR : Échec du chargement du modèle CLIP depuis Hugging Face : {e}")
            print("Veuillez vérifier votre connexion ou le HF_ENDPOINT.")
            raise e

        print(f"  Stream 1: Applying LoRA with rank {lora_rank}...")
        for layer in self.clip_backbone.encoder.layers:
            attn = layer.self_attn
            attn.q_proj = replace_with_lora(attn.q_proj, lora_rank)
            attn.k_proj = replace_with_lora(attn.k_proj, lora_rank)
            attn.v_proj = replace_with_lora(attn.v_proj, lora_rank)
            attn.out_proj = replace_with_lora(attn.out_proj, lora_rank)
        
        lora.mark_only_lora_as_trainable(self.clip_backbone, bias='lora_only')
        print(f"  Stream 1: CLIP backbone (dim={self.clip_feat_dim}) and LoRA ready.")


        # --- Stream 2: Residual + EffNet ---
        print(f"  Stream 2: Loading {stream2_model_name}...")
        try:
            self.effnet_backbone = EfficientNet.from_pretrained(stream2_model_name)
            self.effnet_feat_dim = self.effnet_backbone._fc.in_features
        except Exception as e:
            print(f"ERREUR : Échec du chargement d'EfficientNet : {e}")
            raise e
        print(f"  Stream 2: EfficientNet backbone (dim={self.effnet_feat_dim}) ready.")

        # --- Noiser (pour Stream 2) ---
        print(f"  Stream 2: Attaching Noiser...")
        self.noiser = noiser
        # Geler le noiser - c'est un extracteur de caractéristiques fixe
        self.noiser.eval()
        for param in self.noiser.parameters():
            param.requires_grad = False
        print(f"  Stream 2: Noiser attached and frozen.")


        # --- Transformations de pré-traitement (au sein du modèle) ---
        self.crop224 = transforms.CenterCrop(clip_input_size)
        # REQ 3: Normalisation spécifique à CLIP
        self.normalize_clip = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073], 
            std=[0.26862954, 0.26130258, 0.27577711]
        )

        # --- Tête de Fusion (Shallow Classifier) ---
        self.fusion_dim = self.clip_feat_dim + self.effnet_feat_dim
        print(f"  Fusion: CLS={self.clip_feat_dim}, EFF={self.effnet_feat_dim}, Total={self.fusion_dim}")
        # REQ 6: Classifieur peu profond
        self.fusion_head = nn.Sequential(
            nn.Linear(self.fusion_dim, 256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(256, 1)
        )        
        # S'assurer que la tête de fusion est entraînable
        for param in self.fusion_head.parameters():
            param.requires_grad = True

        print("Dual-Stream Model initialized.")

    def forward(self, x):
        # x est (B, 3, 256, 256), normalisé VQGAN (-1 à 1)

        # --- Stream 1: CLIP ---
        # REQ 3: Dénormaliser VQGAN [-1, 1] -> [0, 1]
        x_0_1 = denorm(x)
        # REQ 1: Recadrer au centre à 224
        x_224 = self.crop224(x_0_1)
        # REQ 3: Appliquer la normalisation CLIP
        x_clip = self.normalize_clip(x_224)
        # REQ 5: Obtenir le vecteur de caractéristiques
        feat1 = self.clip_backbone(x_clip)['pooler_output'] # (B, 1024)

        # --- Stream 2: Residual ---
        # REQ 4: Obtenir le résidu (le noiser est gelé et en mode eval)
        residual = self.noiser(x)
        residual_fp = residual - residual.floor() # (B, 3, 256, 256)
        
        # REQ 5: Extraire les caractéristiques d'EfficientNet
        feat2 = self.effnet_backbone.extract_features(residual_fp)
        feat2 = self.effnet_backbone._avg_pooling(feat2) # (B, 1280, 1, 1)
        feat2 = feat2.flatten(start_dim=1) # (B, 1280)
        
        # --- REQ 6: Fusion ---
        combined_feat = torch.cat([feat1, feat2], dim=1) # (B, 2304)
        output = self.fusion_head(combined_feat) # (B, 1)
        
        return output

# --- Fin du nouveau modèle ---


def get_parser():
    parser = argparse.ArgumentParser()
    # Chemins des données
    parser.add_argument("--train_dir", type=str, default="GenImage/imagenet_ai_0419_sdv4/train")
    parser.add_argument("--val_dir", type=str, default="GenImage/imagenet_ai_0419_sdv4/val")
    parser.add_argument("--eval_datasets_root", type=str, default="GenImage/valdata")
    
    # Paramètres d'entraînement
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    # Fréquence des logs et évaluations
    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument("--save_model_freq", type=int, default=5000)
    parser.add_argument("--eval_freq", type=int, default=5000)
    parser.add_argument("--eval_batch", type=int, default=1000)

    # --- NOUVEAU : Modèles et Checkpoints pour Dual-Stream ---
    parser.add_argument("--stream1_model", type=str, default="openai/clip-vit-large-patch14", help="Backbone pour Stream 1 (CLIP)")
    parser.add_argument("--stream2_model", type=str, default="efficientnet-b0", help="Backbone pour Stream 2 (Résidu)")
    parser.add_argument("--lora_rank", type=int, default=4, help="Rang pour LoRA sur Stream 1")
    
    parser.add_argument("--resume", type=str, default=None)

    # Checkpoints pour les composants pré-entraînés (utilisés au début)
    parser.add_argument("--encoder_module", type=str, default="encoder2", help="Module encodeur AE (pour l'augmentation)")
    parser.add_argument("--decoder_module", type=str, default="decoder24_with_graph", help="Module décodeur AE (pour l'augmentation)")
    parser.add_argument("--ae_resume", type=str, default=None, help="Checkpoint AE pré-entraîné (pour l'augmentation)")
    parser.add_argument("--noiser_resume", type=str, default="denosier_250824225428_bs8_ga16_lr0.0003/models/checkpoint-400000.pth", help="Checkpoint Denoiser pré-entraîné (pour Stream 2)")

    # Logique d'augmentation et de perte (de Script 2)
    parser.add_argument("--aug_target", type=str, default='all', choices=['none', 'ai', 'nature', 'all'], help="Cible d'augmentation")
    parser.add_argument("--aug_prob", type=float, default=0.2, help="Probabilité d'augmentation sur la cible")
    
    return parser

def load_eval_datasets(eval_datasets_root, val_transform):
    """
    Charge les ensembles de données d'évaluation.
    (Identique à vos scripts)
    """
    eval_loaders = {}
    eval_info = {}
    
    if not os.path.exists(eval_datasets_root):
        print(f"Avertissement: Répertoire racine des datasets d'évaluation non trouvé: {eval_datasets_root}")
        return eval_loaders, eval_info
    
    print(f"\n=== Chargement des datasets d'évaluation ===")
    print(f"Répertoire racine: {eval_datasets_root}")
    
    dataset_folders = [item for item in os.listdir(eval_datasets_root) if os.path.isdir(os.path.join(eval_datasets_root, item))]
    dataset_folders.sort()
    print(f"Trouvé {len(dataset_folders)} dossiers de datasets: {dataset_folders}")
    
    for dataset_name in dataset_folders:
        dataset_path = os.path.join(eval_datasets_root, dataset_name)
        print(f"\n--- Traitement du dataset: {dataset_name} ---")
        
        possible_paths = []
        direct_val_path = os.path.join(dataset_path, "val")
        if os.path.exists(direct_val_path):
            possible_paths.append(("direct", direct_val_path))
        
        for sub_item in os.listdir(dataset_path):
            sub_path = os.path.join(dataset_path, sub_item)
            if os.path.isdir(sub_path):
                val_path = os.path.join(sub_path, "val")
                if os.path.exists(val_path):
                    possible_paths.append((f"sub:{sub_item}", val_path))
        
        if possible_paths:
            selected_path_type, selected_path = possible_paths[0]
            print(f"  Chemin choisi: {selected_path} (Type: {selected_path_type})")
            
            try:
                subfolders = [f for f in os.listdir(selected_path) if os.path.isdir(os.path.join(selected_path, f))]
                print(f"  Sous-dossiers: {subfolders}")
                
                if 'ai' in subfolders and 'nature' in subfolders:
                    dataset = ImageFolder(selected_path, transform=val_transform)
                    # Utiliser drop_last=True pour la cohérence de la taille de lot
                    loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=8, persistent_workers=False, pin_memory=True, drop_last=True)
                    
                    class_to_idx = dataset.class_to_idx
                    class_counts = defaultdict(int)
                    for _, label in dataset.samples:
                        class_counts[label] += 1
                    
                    idx_to_class = {v: k for k, v in class_to_idx.items()}
                    class_name_counts = {idx_to_class[idx]: count for idx, count in class_counts.items()}
                    
                    eval_loaders[dataset_name] = loader
                    eval_info[dataset_name] = {
                        'path': selected_path,
                        'class_to_idx': class_to_idx,
                        'class_counts': class_name_counts,
                        'total_samples': len(dataset.samples)
                    }
                    
                    print(f"  ✓ Chargé avec succès")
                    print(f"     Mapping des classes: {class_to_idx}")
                    print(f"     Nombre d'échantillons: {class_name_counts}")
                else:
                    print(f"  ✗ Ignoré - Dossiers 'ai' et 'nature' non trouvés (Trouvés: {subfolders})")
            except Exception as e:
                print(f"  ✗ Échec du chargement: {str(e)}")
        else:
            print(f"  ✗ Ignoré - Chemin 'val' valide non trouvé")
    
    print(f"\n=== Chargement des datasets d'évaluation terminé ===")
    print(f"{len(eval_loaders)} datasets chargés avec succès: {list(eval_loaders.keys())}")
    
    return eval_loaders, eval_info


# --- NOUVEAU : Fonction d'évaluation simplifiée ---
def evaluate(model, val_loader, device, max_batches=None):
    """
    Évalue le DualStreamModel.
    Le modèle gère en interne tout le pré-traitement (denorm, crop, norm, noiser).
    """
    model.eval() # Met le modèle entier (y compris les backbones) en mode eval
    correct = 0
    total = 0
    batch_count = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            if max_batches and batch_count >= max_batches:
                break
            # imgs est (B, 3, 256, 256), normalisé VQGAN
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            # Le modèle gère tout le reste
            outputs = model(imgs).squeeze()
            
            probs = torch.sigmoid(outputs)
            predicted = (probs > 0.5).long()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            batch_count += 1
    
    return 100 * correct / total

def main(params):
    # Définir les graines
    torch.manual_seed(params.seed)
    np.random.seed(params.seed)
    random.seed(params.seed)

    # --- Configuration du répertoire d'expérimentation ---
    start_step = 1
    best_acc = 0

    if params.resume and os.path.isfile(params.resume):
        # --- Mode Reprise ---
        print(f"Mode Reprise détecté, chargement du checkpoint: {params.resume}")
        model_dir = os.path.dirname(params.resume)
        exp_path = os.path.dirname(model_dir)
        
        os.makedirs(model_dir, exist_ok=True)
        
        print(f"Reprise de l'écriture dans le répertoire d'expérimentation: {exp_path}")
        
        try:
            exp_name = os.path.basename(exp_path)
            dt_string = exp_name.split('_')[1] 
        except:
            dt_string = datetime.now().strftime("%y%m%d%H%M%S")
            print(f"Impossible de parser dt_string, utilisation de l'heure actuelle: {dt_string}")
        
        best_model_path = os.path.join(model_dir, "best_model.pth")
        if os.path.exists(best_model_path):
            try:
                best_model_cp = torch.load(best_model_path, map_location=device)
                best_acc = best_model_cp.get("val_acc", 0)
                print(f"Chargé depuis {best_model_path}, best_acc défini à {best_acc:.2f}%")
            except Exception as e:
                print(f"Avertissement: Impossible de charger best_acc depuis best_model.pth: {e}. best_acc commence à 0.")
        else:
            print("best_model.pth non trouvé. best_acc commence à 0.")
            
    else:
        # --- Mode Nouvel Entraînement ---
        print("Aucun 'resume' valide fourni, démarrage d'un nouvel entraînement.")
        now = datetime.now()
        dt_string = now.strftime("%y%m%d%H%M%S")
        
        exp_name_parts = [
            f"dualstream_{dt_string}",
            f"s1_{params.stream1_model.split('/')[-1]}-lora{params.lora_rank}", # ex: s1_clip-vit-large-patch14-lora4
            f"s2_{params.stream2_model}", # ex: s2_efficientnet-b0
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
            f.write(f"Commande: {' '.join(sys.argv)}\n")
            
    # --- Transformations de données (pour le DataLoader) ---
    # REQ 1 & 4: Les images sont chargées à 256x256, normalisées VQGAN
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
    print("=== Chargement des données d'entraînement et de validation ===")
    train_dataset = ImageFolder(params.train_dir, transform=train_transform)
    val_dataset = ImageFolder(params.val_dir, transform=val_transform)

    # Utiliser drop_last=True pour éviter les problèmes avec la perte équilibrée si le dernier lot est petit
    train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, persistent_workers=False, num_workers=8, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True,  persistent_workers=False, num_workers=8, pin_memory=True, drop_last=True)
    
    print(f"Dataset d'entraînement - Chemin: {params.train_dir}")
    print(f"Dataset d'entraînement - Mapping des classes: {train_dataset.class_to_idx}")
    
    ai_label = train_dataset.class_to_idx.get('ai', 0)
    nature_label = train_dataset.class_to_idx.get('nature', 1)
    print(f"Index label AI: {ai_label}, Index label Nature: {nature_label}")

    eval_loaders, eval_info = load_eval_datasets(params.eval_datasets_root, val_transform)

    # --- Chargement des modèles pré-requis ---
    
    # 1. AE (pour l'augmentation)
    encoder = None
    decoder = None
    if params.aug_target != 'none' and params.ae_resume:
        print(f"Chargement des modules AE: {params.encoder_module}, {params.decoder_module}")
        try:
            decoder_module = importlib.import_module(params.decoder_module)
            encoder_module = importlib.import_module(params.encoder_module)
            Decoder_class = getattr(decoder_module, "Decoder")
            Encoder_class = getattr(encoder_module, "Encoder")
            encoder = Encoder_class().to(device)
            decoder = Decoder_class().to(device)
            
            if os.path.isfile(params.ae_resume):
                print(f"Chargement du modèle AE pré-entraîné: {params.ae_resume}")
                checkpoint = torch.load(params.ae_resume, map_location=device)
                encoder.load_state_dict(checkpoint["encoder"])
                decoder.load_state_dict(checkpoint["decoder"])
                encoder.eval()
                decoder.eval()
            else:
                print(f"Avertissement: Checkpoint AE {params.ae_resume} non trouvé. L'augmentation sera désactivée.")
                encoder = None
                decoder = None
        except Exception as e:
            print(f"Erreur lors du chargement de l'AE: {e}. L'augmentation sera désactivée.")
            encoder = None
            decoder = None
    elif params.aug_target != 'none':
         print("Avertissement: aug_target défini mais ae_resume non fourni. L'augmentation sera désactivée.")

    # 2. Noiser (pour Stream 2)
    print(f"Chargement du Noiser (DenoisingFCNWithSkip)")
    noiser = DenoisingFCNWithSkip().to(device) # Initialiser
    
    if params.noiser_resume and os.path.isfile(params.noiser_resume):
        print(f"Chargement du modèle Noiser pré-entraîné: {params.noiser_resume}")
        checkpoint = torch.load(params.noiser_resume, map_location=device)
        # Adapter aux formats de checkpoint
        if "noiser" in checkpoint:
              noiser_state_dict = checkpoint["noiser"]
        elif "model_state_dict" in checkpoint:
             noiser_state_dict = checkpoint["model_state_dict"]
        else:
             noiser_state_dict = checkpoint
        
        try:
            noiser.load_state_dict(noiser_state_dict)
            print("Noiser chargé avec succès.")
        except Exception as e:
            print(f"Erreur lors du chargement du state_dict du noiser: {e}. Tentative avec strict=False...")
            noiser.load_state_dict(noiser_state_dict, strict=False)
            
    else:
        print(f"Avertissement: Checkpoint Noiser {params.noiser_resume} non trouvé. Le Stream 2 utilisera un noiser non entraîné !")
    
    # Le noiser est réglé sur eval() et gelé à l'intérieur de DualStreamModel

    # --- Initialisation du Modèle Principal ---
    
    model = DualStreamModel(
        stream1_model_name=params.stream1_model,
        stream2_model_name=params.stream2_model,
        noiser=noiser, # Passer l'instance de noiser chargée
        lora_rank=params.lora_rank
    ).to(device)

    # --- Optimiseur ---
    # Cela attrapera les poids entraînnables : LoRA (Stream 1), EffNet (Stream 2), Tête de Fusion
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
    
    # --- Logique de Reprise (pour le modèle dual-stream) ---
    if params.resume and os.path.isfile(params.resume):
        print(f"Chargement du checkpoint du modèle dual-stream: {params.resume}")
        clf_checkpoint = torch.load(params.resume, map_location=device)
        
        # Charger les poids du modèle. strict=False est plus sûr.
        # Cela chargera les poids pour CLIP, EffNet, Fusion, *et* écrasera le Noiser
        # avec les poids du checkpoint, ce qui est le comportement souhaité.
        try:
            model.load_state_dict(clf_checkpoint["model"], strict=False)
            print("Poids du modèle dual-stream chargés (strict=False).")
        except Exception as e:
             print(f"Erreur lors du chargement du state dict du modèle : {e}")
             
        if "optimizer" in clf_checkpoint:
            try:
                optimizer.load_state_dict(clf_checkpoint["optimizer"])
                print("État de l'optimiseur chargé avec succès.")
            except Exception as e:
                print(f"Avertissement: Échec du chargement de l'état de l'optimiseur: {e}. Utilisation d'un nouvel optimiseur.")

        if "step" in clf_checkpoint:
            start_step = clf_checkpoint["step"] + 1
            print(f"Reprise de l'entraînement à partir de l'étape {start_step}.")
    else:
        print("Démarrage d'un nouvel entraînement (pas de checkpoint 'resume' fourni).")
        print("Stream 1 (CLIP) utilise les poids pré-entraînés + LoRA aléatoire.")
        print("Stream 2 (EffNet) utilise les poids pré-entraînés d'ImageNet.")
        print("Stream 2 (Noiser) utilise le checkpoint noiser_resume.")
        print("La tête de fusion est initialisée aléatoirement.")


    # --- Boucle d'entraînement ---
    criterion_unbalanced = nn.BCEWithLogitsLoss()

    start_time = time.time()
    data_iter = iter(train_loader)

    print("\n=== Démarrage de l'entraînement (Mode Dual-Stream) ===")
    
    for step in range(start_step, params.steps + 1):
        model.train() # S'assurer que le modèle est en mode entraînement (active LoRA, EffNet, Fusion)
        
        try:
            imgs, original_labels = next(data_iter)
        except:
            data_iter = iter(train_loader)
            imgs, original_labels = next(data_iter)

        imgs, original_labels = imgs.to(device, non_blocking=True), original_labels.to(device, non_blocking=True)
        
        augmented_imgs = imgs.clone()
        is_augmented = torch.zeros_like(original_labels, dtype=torch.bool, device=device)

        # --- 1. Augmentation de données (REQ 1) ---
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

        # --- 2. Logique des Nouveaux Labels ---
        # Positif (1): Nature Original (non-augmenté)
        # Négatif (0): AI Original, AI Augmenté, Nature Augmenté
        final_labels = torch.zeros_like(original_labels, device=device)
        final_labels[(original_labels == nature_label) & (~is_augmented)] = 1
        
        # --- 3. Forward Pass (Dual-Stream) ---
        # augmented_imgs est (B, 3, 256, 256) VQGAN-norm
        outputs = model(augmented_imgs).squeeze()
        
        # --- 4. Calcul de la Perte ---
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

        # --- Évaluation (REQ 7) ---
        if step % params.eval_freq == 0:
            # model.eval() est appelé à l'intérieur de la fonction evaluate
            val_acc = evaluate(model, val_loader, device, params.eval_batch)
            print(f"Précision de validation (Tâche Originale): {val_acc:.2f}%")
            
            with open(os.path.join(exp_path, 'eval_results.txt'), 'a') as f:
                f.write(f"Step {step:07d} - Val Acc (Tâche Originale): {val_acc:.2f}%\n")

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    "model": model.state_dict(), # Sauvegarde le state_dict du modèle entier
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "val_acc": val_acc
                }, os.path.join(model_dir, "best_model.pth"))
                print(f"*** Nouveau meilleur modèle sauvegardé (Acc: {val_acc:.2f}%) ***")

            if eval_loaders:
                print(f"\n--- Évaluation Multi-Dataset (Étape {step}) ---")
                
                with open(os.path.join(exp_path, 'eval_results.txt'), 'a') as f:
                    f.write(f"--- Étape {step} Multi-Dataset ---\n")
                    
                for dataset_name, eval_loader in eval_loaders.items():
                    eval_acc = evaluate(model, eval_loader, device, params.eval_batch)
                    print(f"{dataset_name}: {eval_acc:.2f}%")
                    with open(os.path.join(exp_path, 'eval_results.txt'), 'a') as f:
                        f.write(f"{dataset_name}: {eval_acc:.2f}%\n")
                
                with open(os.path.join(exp_path, 'eval_results.txt'), 'a') as f:
                    f.write("\n")
                print("--- Évaluation Multi-Dataset terminée ---\n")

            # Remettre le modèle en mode entraînement
            model.train() 

        # --- Sauvegarde du Checkpoint ---
        if step % params.save_model_freq == 0:
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step
            }, os.path.join(model_dir, f"checkpoint-{step:06d}.pth"))

    print(f"Entraînement terminé, meilleure précision de validation (Tâche Originale): {best_acc:.2f}%")

if __name__ == "__main__":   
    parser = get_parser()
    params = parser.parse_args()
        
    main(params)