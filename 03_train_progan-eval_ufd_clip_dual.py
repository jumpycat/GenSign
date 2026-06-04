# HF_ENDPOINT a-t-il été défini ?
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import argparse
import sys
import time
import random
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from datetime import datetime, timedelta
import importlib  # <<< REQUIS pour l'AE
from collections import defaultdict
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

# --- Importations des différents scripts ---
from transformers import CLIPModel
import loralib as lora
from efficientnet_pytorch import EfficientNet
import timm
import torchvision.models as models

try:
    from model import DenoisingFCNWithSkip
except ImportError:
    print("ERREUR: Impossible d'importer 'DenoisingFCNWithSkip' depuis 'model.py'.")
    print("Veuillez vous assurer que le fichier 'model.py' est dans le même répertoire.")
    sys.exit(1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --- Fonctions d'assistance (de clip_dual) ---

def denorm(x):
    """ Dénormalise de VQGAN [-1, 1] à [0, 1] """
    return (x + 1.0) / 2.0

def format_time(seconds):
    delta = timedelta(seconds=seconds)
    days = delta.days
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{days}d {hours}h {minutes}m {seconds}s'

def replace_with_lora(module, rank):
    """ Remplace une couche nn.Linear par une couche lora.Linear et copie les poids. """
    lora_layer = lora.Linear(
        module.in_features,
        module.out_features,
        r=rank,
        lora_alpha=rank, 
        lora_dropout=0.1,
        bias=(module.bias is not None)
    )
    lora_layer.weight.data.copy_(module.weight.data)
    if module.bias is not None:
        lora_layer.bias.data.copy_(module.bias.data)
    return lora_layer


# --- Classes de Dataset (de progan_baseline et eval_UniversalFakeDetect) ---

class CustomTrainDataset(Dataset):
    """
    (De train_progan4_baseline.py)
    Charge les données d'entraînement progan_train.
    Labels: 0=fake/ai, 1=real/nature
    """
    def __init__(self, root_dir, transform=None, subfolders_to_use=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.subfolders_to_use = subfolders_to_use
        
        self.label_map = {'0_real': 1, '1_fake': 0}
        
        print(f"=== Chargement des données d'entraînement personnalisées: {root_dir} ===")
        
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Répertoire d'entraînement non trouvé: {root_dir}")
            
        all_semantic_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        
        if self.subfolders_to_use:
            print(f"Total de {len(all_semantic_dirs)} dossiers sémantiques trouvés.")
            semantic_dirs = [d for d in all_semantic_dirs if d in self.subfolders_to_use]
            print(f"Utilisation des {len(semantic_dirs)} dossiers spécifiés: {semantic_dirs}")
            
            missing = [d for d in self.subfolders_to_use if d not in all_semantic_dirs]
            if missing:
                print(f"Avertissement: Dossiers spécifiés non trouvés dans {root_dir}: {missing}")
        else:
            semantic_dirs = all_semantic_dirs
            print(f"Trouvé {len(semantic_dirs)} dossiers sémantiques (tous seront utilisés), recherche de '0_real' et '1_fake'...")

        image_extensions = ('.png', '.jpg', '.jpeg', '.webp')
        
        for sem_dir in semantic_dirs:
            sem_path = os.path.join(root_dir, sem_dir)
            for label_name, label_idx in self.label_map.items():
                data_path = os.path.join(sem_path, label_name)
                if os.path.exists(data_path):
                    for root, _, files in os.walk(data_path):
                        for file in files:
                            if file.lower().endswith(image_extensions):
                                self.samples.append((os.path.join(root, file), label_idx))
        
        print(f"Total de {len(self.samples)} échantillons d'entraînement chargés.")
        counts = defaultdict(int)
        for _, label in self.samples:
            counts[label] += 1
        print(f"Distribution des échantillons: {dict(counts)} (0=fake, 1=real)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Avertissement: Échec du chargement de l'image {img_path}, {e}. Remplacement par un index aléatoire.")
            new_idx = random.randint(0, len(self) - 1)
            return self[new_idx]
            
        if self.transform:
            img = self.transform(img)
        
        return img, label

class SpecificEvalDataset(Dataset):
    """
    (De eval_UniversalFakeDetect.py)
    Charge les dossiers real et fake spécifiés pour l'évaluation.
    - real_dir -> label 1 (real/nature)
    - fake_dir -> label 0 (fake/ai)
    """
    def __init__(self, real_dir, fake_dir, transform=None):
        self.transform = transform
        self.samples = []
        label_map = {real_dir: 1, fake_dir: 0}

        print(f"    > Chargement Real: {real_dir} (Label 1)")
        print(f"    > Chargement Fake: {fake_dir} (Label 0)")

        image_extensions = ('.png', '.jpg', '.jpeg', '.JPEG', '.webp', '.tif')

        for data_path, label in label_map.items():
            if not os.path.exists(data_path):
                print(f"Avertissement: Chemin non trouvé, ignoré: {data_path}")
                continue

            path_files = []
            for ext in image_extensions:
                path_files.extend(glob.glob(os.path.join(data_path, '**', f"*{ext}"), recursive=True))

            for f in path_files:
                self.samples.append((f, label))

        counts = defaultdict(int)
        for _, label in self.samples:
            counts[label] += 1
        print(f"    > Chargement terminé: {len(self.samples)} échantillons. Distribution: {dict(counts)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Avertissement: Échec du chargement de l'image {img_path}, {e}. Remplacement par un index aléatoire.")
            new_idx = random.randint(0, len(self) - 1)
            return self[new_idx]

        if self.transform:
            img = self.transform(img)

        return img, label

class ConditionalResize(object):
    """
    (De eval_UniversalFakeDetect.py)
    Transform serializable
    """
    def __init__(self, size):
        self.size = size
        self.resize_op = transforms.Resize(size)

    def __call__(self, img):
        if min(img.size) < self.size:
            return self.resize_op(img)
        return img

# --- Modèle Principal (de clip_dual) ---

class DualStreamModel(nn.Module):
    def __init__(self, stream1_model_name, stream2_model_name, noiser, lora_rank, clip_input_size=224):
        super(DualStreamModel, self).__init__()

        print(f"Initialisation du Modèle Dual-Stream:")
        
        # --- Stream 1: CLIP + LoRA ---
        print(f"  Stream 1: Chargement {stream1_model_name}...")
        try:
            clip_model = CLIPModel.from_pretrained(stream1_model_name)
            self.clip_backbone = clip_model.vision_model
            self.clip_feat_dim = clip_model.config.vision_config.hidden_size 
        except Exception as e:
            print(f"ERREUR : Échec du chargement du modèle CLIP depuis Hugging Face : {e}")
            raise e

        print(f"  Stream 1: Application de LoRA avec rang {lora_rank}...")
        for layer in self.clip_backbone.encoder.layers:
            attn = layer.self_attn
            attn.q_proj = replace_with_lora(attn.q_proj, lora_rank)
            attn.k_proj = replace_with_lora(attn.k_proj, lora_rank)
            attn.v_proj = replace_with_lora(attn.v_proj, lora_rank)
            attn.out_proj = replace_with_lora(attn.out_proj, lora_rank)
        
        lora.mark_only_lora_as_trainable(self.clip_backbone, bias='lora_only')
        print(f"  Stream 1: Backbone CLIP (dim={self.clip_feat_dim}) et LoRA prêts.")


        # --- Stream 2: Residual + EffNet ---
        print(f"  Stream 2: Chargement {stream2_model_name}...")
        try:
            self.effnet_backbone = EfficientNet.from_pretrained(stream2_model_name, num_classes=1) 
            self.effnet_feat_dim = self.effnet_backbone._fc.in_features
            self.effnet_backbone._fc = nn.Identity() 
        except Exception as e:
            print(f"ERREUR : Échec du chargement d'EfficientNet : {e}")
            raise e
        print(f"  Stream 2: Backbone EfficientNet (dim={self.effnet_feat_dim}) prêt.")

        # --- Noiser (pour Stream 2) ---
        print(f"  Stream 2: Rattachement du Noiser...")
        self.noiser = noiser
        self.noiser.eval()
        for param in self.noiser.parameters():
            param.requires_grad = False
        print(f"  Stream 2: Noiser rattaché et gelé.")


        # --- Transformations de pré-traitement (au sein du modèle) ---
        self.crop224 = transforms.CenterCrop(clip_input_size)
        self.normalize_clip = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073], 
            std=[0.26862954, 0.26130258, 0.27577711]
        )

        # --- Tête de Fusion ---
        self.fusion_dim = self.clip_feat_dim + self.effnet_feat_dim
        print(f"  Fusion: CLS={self.clip_feat_dim}, EFF={self.effnet_feat_dim}, Total={self.fusion_dim}")
        self.fusion_head = nn.Sequential(
            nn.Linear(self.fusion_dim, 256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(256, 1)
        )        
        for param in self.fusion_head.parameters():
            param.requires_grad = True

        print("Modèle Dual-Stream initialisé.")

    def forward(self, x):
        # x est (B, 3, 256, 256), normalisé VQGAN (-1 à 1)

        # --- Stream 1: CLIP ---
        x_0_1 = denorm(x)
        x_224 = self.crop224(x_0_1)
        x_clip = self.normalize_clip(x_224)
        feat1 = self.clip_backbone(x_clip)['pooler_output']

        # --- Stream 2: Residual ---
        with torch.no_grad(): 
            residual = self.noiser(x)
        residual_fp = residual - residual.floor()
        
        feat2 = self.effnet_backbone.extract_features(residual_fp)
        feat2 = self.effnet_backbone._avg_pooling(feat2)
        feat2 = feat2.flatten(start_dim=1)
        
        # --- Fusion ---
        combined_feat = torch.cat([feat1, feat2], dim=1)
        output = self.fusion_head(combined_feat)
        
        return output

# --- Fin du modèle ---


def get_parser():
    parser = argparse.ArgumentParser()
    
    # === Arguments fusionnés ===

    # De progan_baseline (données d'entraînement)
    parser.add_argument("--train_dir", type=str, default="/CNNDetection/progan_train")
    parser.add_argument("--train_subfolders", type=str, nargs='+', default=['car','cat','chair','horse','tvmonitor','train','sofa','sheep','pottedplant','person','motorbike','dog','diningtable','cow','bus','bottle','boat','bird','bicycle','airplane'])   
    # parser.add_argument("--train_subfolders", type=str, nargs='+', default=['car','cat','chair','horse'])   

    # De clip_dual (paramètres d'entraînement)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    # De clip_dual (logs et éval)
    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument("--save_model_freq", type=int, default=5000)
    parser.add_argument("--eval_freq", type=int, default=5000)
    parser.add_argument("--eval_batch", type=int, default=1000, help="Nombre max de lots par dataset d'évaluation (pour accélérer)")

    # De clip_dual (Modèles et Checkpoints)
    parser.add_argument("--stream1_model", type=str, default="openai/clip-vit-large-patch14", help="Backbone pour Stream 1 (CLIP)")
    parser.add_argument("--stream2_model", type=str, default="efficientnet-b0", help="Backbone pour Stream 2 (Résidu)")
    parser.add_argument("--lora_rank", type=int, default=4, help="Rang pour LoRA sur Stream 1")
    
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--noiser_resume", type=str, default="/media2/dys/01-projects/22_freqaug/fp_extractor/runs/denosier_250824225428_bs8_ga16_lr0.0003/models/checkpoint-400000.pth", help="Checkpoint Denoiser pré-entraîné (pour Stream 2)")

    # De clip_dual (Logique d'augmentation AE)
    parser.add_argument("--encoder_module", type=str, default="decoders.encoder2", help="Module encodeur AE")
    parser.add_argument("--decoder_module", type=str, default="decoders.decoder24", help="Module décodeur AE")
    parser.add_argument("--ae_resume", type=str, default='/media2/dys/01-projects/22_freqaug/coolboy2/runs/nosier_250708204953_decoder24_bs4_ga32_both_1.0_0.25_lr4e-05/models/checkpoint-1000000.pth', help="Checkpoint AE pré-entraîné")
    parser.add_argument("--aug_target", type=str, default='all', choices=['none', 'ai', 'nature', 'all'], help="Cible d'augmentation")
    parser.add_argument("--aug_prob", type=float, default=0.5, help="Probabilité d'augmentation sur la cible")
    
    return parser


# --- Fonction d'évaluation (de clip_dual, améliorée avec AUC/AP de progan_baseline) ---
def evaluate(model, val_loader, device, max_batches=None, dataset_name=""):
    """
    Évalue le DualStreamModel.
    Le modèle gère en interne tout le pré-traitement.
    Renvoie (acc, auc, ap).
    """
    model.eval()
    correct = 0
    total = 0
    batch_count = 0
    
    all_labels = []
    all_probs = []

    with torch.no_grad():
        pbar_desc = f"  > Évaluation {dataset_name}"
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
            print(f"  (Avertissement: {dataset_name} n'a qu'une seule classe, AUC/AP non calculés)")
            
    except Exception as e:
        print(f"  (Avertissement: Échec du calcul AUC/AP pour {dataset_name}: {e})")

    return acc, auc, ap

def main(params):
    # Définir les graines
    torch.manual_seed(params.seed)
    np.random.seed(params.seed)
    random.seed(params.seed)

    # --- Configuration du répertoire d'expérimentation ---
    start_step = 1
    best_acc = 0 # Sera basé sur l'ACC moyen

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
        
        best_model_path = os.path.join(model_dir, "best_model.pth")
        if os.path.exists(best_model_path):
            try:
                best_model_cp = torch.load(best_model_path, map_location=device)
                best_acc = best_model_cp.get("val_acc", 0) # val_acc est l'ACC moyen
                print(f"Chargé depuis {best_model_path}, best_acc (moyen) défini à {best_acc:.2f}%")
            except Exception as e:
                print(f"Avertissement: Impossible de charger best_acc: {e}. best_acc commence à 0.")
        else:
            print("best_model.pth non trouvé. best_acc commence à 0.")
            
    else:
        # --- Mode Nouvel Entraînement ---
        print("Aucun 'resume' valide fourni, démarrage d'un nouvel entraînement.")
        now = datetime.now()
        dt_string = now.strftime("%y%m%d%H%M%S")
        
        exp_name_parts = [
            f"dualstream_{dt_string}",
            f"s1_{params.stream1_model.split('/')[-1]}-lora{params.lora_rank}",
            f"s2_{params.stream2_model}",
            f"bs{params.batch_size}",
            f"ga{params.grad_accum_steps}",
            f"lr{params.lr}",
            f"aug_{params.aug_target}{params.aug_prob}", # <<< Ajout de la logique d'aug
            "train_progan" 
        ]
        exp_name = "_".join(exp_name_parts)
        
        exp_path = os.path.join("runs", exp_name)
        model_dir = os.path.join(exp_path, "models")
        os.makedirs(model_dir, exist_ok=True)

        with open(os.path.join(exp_path, 'args.txt'), 'w') as f:
            for arg in vars(params):
                f.write(f"{arg}: {getattr(params, arg)}\n")
            f.write(f"Commande: {' '.join(sys.argv)}\n")
            
    # --- Transformations de données ---
    normalize_vqgan = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    
    # Transform d'entraînement (de progan_baseline)
    train_transform = transforms.Compose([
        lambda img: transforms.Resize(params.img_size)(img) if min(img.size) < params.img_size else img,
        transforms.RandomCrop(params.img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize_vqgan
    ])
    
    # Transform de validation (de eval_UniversalFakeDetect)
    val_transform = transforms.Compose([
        ConditionalResize(size=params.img_size),
        transforms.CenterCrop(params.img_size),
        transforms.ToTensor(),
        normalize_vqgan
    ])

    # --- DataLoaders ---
    
    # 1. Données d'entraînement (de progan_baseline)
    print("=== Chargement des données d'entraînement ===")
    train_dataset = CustomTrainDataset(
        params.train_dir, 
        transform=train_transform,
        subfolders_to_use=params.train_subfolders
    )
    train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, persistent_workers=True, num_workers=8, pin_memory=True, drop_last=True)
    
    # Labels: 0 = ai/fake, 1 = nature/real
    # CELA CORRESPOND aux hypothèses de clip_dual (ai=0, nature=1)
    ai_label = 0
    nature_label = 1
    print(f"Index label AI: {ai_label}, Index label Nature: {nature_label}")

    # 2. Données d'évaluation (de eval_UniversalFakeDetect)
    print("\n=== Chargement des données d'évaluation ===")
    
    # FIXME: METTEZ À JOUR CES CHEMINS AVEC VOS CHEMINS LINUX
    EVAL_DATASETS = {
        'biggan': {
            'real_path': '/CNNDetection/CNN_synth_testset/biggan/0_real',
            'fake_path': '/CNNDetection/CNN_synth_testset/biggan/1_fake'
        },
        'crn': {
            'real_path': '/CNNDetection/CNN_synth_testset/crn/0_real',
            'fake_path': '/CNNDetection/CNN_synth_testset/crn/1_fake'
        },
        'cyclegan_mixall': {
            'real_path': '/CNNDetection/CNN_synth_testset/cyclegan_mixall/0_real',
            'fake_path': '/CNNDetection/CNN_synth_testset/cyclegan_mixall/1_fake'
        },
        'deepfake': {
            'real_path': '/CNNDetection/CNN_synth_testset/deepfake/0_real',
            'fake_path': '/CNNDetection/CNN_synth_testset/deepfake/1_fake'
        },
        'progan': {
            'real_path': '/CNNDetection/CNN_synth_testset/progan_mixall/0_real',
            'fake_path': '/CNNDetection/CNN_synth_testset/progan_mixall/1_fake'
        },
        'gaugan': {
            'real_path': '/CNNDetection/CNN_synth_testset/gaugan/0_real',
            'fake_path': '/CNNDetection/CNN_synth_testset/gaugan/1_fake'
        },
        'imle': {
            'real_path': '/CNNDetection/CNN_synth_testset/imle/0_real',
            'fake_path': '/CNNDetection/CNN_synth_testset/imle/1_fake'
        },
        'progan': {
            'real_path': '/CNNDetection/CNN_synth_testset/progan_mixall/0_real',
            'fake_path': '/CNNDetection/CNN_synth_testset/progan_mixall/1_fake'
        },
        'san': {
            'real_path': '/CNNDetection/CNN_synth_testset/san/0_real',
            'fake_path': '/CNNDetection/CNN_synth_testset/san/1_fake'
        },
        'seeingdark': {
            'real_path': '/CNNDetection/CNN_synth_testset/seeingdark/0_real',
            'fake_path': '/CNNDetection/CNN_synth_testset/seeingdark/1_fake'
        },
        'stargan': {
            'real_path': '/CNNDetection/CNN_synth_testset/stargan/0_real',
            'fake_path': '/CNNDetection/CNN_synth_testset/stargan/1_fake'
        },
        'stylegan_mixall': {
            'real_path': '/CNNDetection/CNN_synth_testset/stylegan_mixall/0_real',
            'fake_path': '/CNNDetection/CNN_synth_testset/stylegan_mixall/1_fake'
        },
        'dalle': {
            'real_path': '/diffusion_datasets/imagenet/0_real',
            'fake_path': '/diffusion_datasets/dalle/1_fake'
        },
        'glide_50_27': {
            'real_path': '/diffusion_datasets/imagenet/0_real',
            'fake_path': '/diffusion_datasets/glide_50_27/1_fake'
        },
        'glide_100_10': {
            'real_path': '/diffusion_datasets/imagenet/0_real',
            'fake_path': '/diffusion_datasets/glide_100_10/1_fake'
        },
        'glide_100_27': {
            'real_path': '/diffusion_datasets/imagenet/0_real',
            'fake_path': '/diffusion_datasets/glide_100_27/1_fake'
        },
        'guided': {
            'real_path': '/diffusion_datasets/laion/0_real',
            'fake_path': '/diffusion_datasets/guided/1_fake'
        },
        'ldm_100': {
            'real_path': '/diffusion_datasets/imagenet/0_real',
            'fake_path': '/diffusion_datasets/ldm_100/1_fake'
        },
        'ldm_200': {
            'real_path': '/diffusion_datasets/imagenet/0_real',
            'fake_path': '/diffusion_datasets/ldm_200/1_fake'
        },
        'ldm_200_cfg': {
            'real_path': '/diffusion_datasets/imagenet/0_real',
            'fake_path': '/diffusion_datasets/ldm_200_cfg/1_fake'
        }
    }
    # FIXME: FIN DE LA SECTION À METTRE À JOUR

    eval_loaders = {}
    for dataset_name, paths in EVAL_DATASETS.items():
        print(f"--- Chargement de l'évaluation: {dataset_name} ---")
        dataset = SpecificEvalDataset(paths['real_path'], paths['fake_path'], transform=val_transform)
        if len(dataset) == 0:
            print(f"  > Avertissement: {dataset_name} n'a chargé aucun échantillon, il sera ignoré.")
            continue
        loader = DataLoader(
            dataset,
            batch_size=params.batch_size * 2, 
            shuffle=True, 
            num_workers=8,
            persistent_workers=False,
            pin_memory=True
        )
        eval_loaders[dataset_name] = loader
    print(f"{len(eval_loaders)} datasets d'évaluation chargés avec succès.")


    # --- Chargement des modèles pré-requis ---
    
    # 1. AE (pour l'augmentation - de clip_dual)
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

    # 2. Noiser (pour Stream 2 - de clip_dual)
    print(f"\nChargement du Noiser (DenoisingFCNWithSkip)")
    noiser = DenoisingFCNWithSkip().to(device)
    
    if params.noiser_resume and os.path.isfile(params.noiser_resume):
        print(f"Chargement du modèle Noiser pré-entraîné: {params.noiser_resume}")
        checkpoint = torch.load(params.noiser_resume, map_location=device)
        
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
            try:
                noiser.load_state_dict(noiser_state_dict, strict=False)
            except Exception as e2:
                print(f"Échec même avec strict=False: {e2}. Le Stream 2 utilisera un noiser non entraîné !")
            
    else:
        print(f"Avertissement: Checkpoint Noiser {params.noiser_resume} non trouvé. Le Stream 2 utilisera un noiser non entraîné !")
    
    # --- Initialisation du Modèle Principal ---
    
    model = DualStreamModel(
        stream1_model_name=params.stream1_model,
        stream2_model_name=params.stream2_model,
        noiser=noiser,
        lora_rank=params.lora_rank
    ).to(device)

    # --- Optimiseur ---
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
    
    # --- Logique de Reprise ---
    if params.resume and os.path.isfile(params.resume):
        print(f"Chargement du checkpoint du modèle dual-stream: {params.resume}")
        clf_checkpoint = torch.load(params.resume, map_location=device)
        
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
                print(f"Avertissement: Échec du chargement de l'état de l'optimiseur: {e}.")

        if "step" in clf_checkpoint:
            start_step = clf_checkpoint["step"] + 1
            print(f"Reprise de l'entraînement à partir de l'étape {start_step}.")
    else:
        print("Démarrage d'un nouvel entraînement.")


    # --- Boucle d'entraînement ---
    criterion_unbalanced = nn.BCEWithLogitsLoss()

    start_time = time.time()
    data_iter = iter(train_loader)

    print("\n=== Démarrage de l'entraînement (Logique de clip_dual / Données de ProGAN) ===")
    
    for step in range(start_step, params.steps + 1):
        model.train() 
        
        try:
            imgs, original_labels = next(data_iter)
        except:
            data_iter = iter(train_loader)
            imgs, original_labels = next(data_iter)

        imgs, original_labels = imgs.to(device, non_blocking=True), original_labels.to(device, non_blocking=True)
        
        # === DÉBUT DE LA LOGIQUE DE TRAIN_CLIP_DUAL ===
        
        augmented_imgs = imgs.clone()
        is_augmented = torch.zeros_like(original_labels, dtype=torch.bool, device=device)

        # 1. Augmentation de données (AE)
        if encoder and decoder and params.aug_target != 'none' and params.aug_prob > 0:
            with torch.no_grad():
                aug_mask = torch.zeros_like(original_labels, dtype=torch.bool, device=device)
                prob_mask = torch.rand(imgs.size(0), device=device) < params.aug_prob

                # Les labels de CustomTrainDataset (ai=0, nature=1) correspondent
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

        # 2. Logique des Nouveaux Labels
        # Positif (1): Nature Original (non-augmenté)
        # Négatif (0): AI Original, AI Augmenté, Nature Augmenté
        final_labels = torch.zeros_like(original_labels, device=device)
        final_labels[(original_labels == nature_label) & (~is_augmented)] = 1
        
        # 3. Forward Pass (Dual-Stream)
        outputs = model(augmented_imgs).squeeze()
        
        # 4. Calcul de la Perte
        loss = criterion_unbalanced(outputs, final_labels.float())
        
        # === FIN DE LA LOGIQUE DE TRAIN_CLIP_DUAL ===

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
                # L'ACC ici est basé sur la *logique d'entraînement* (par rapport aux final_labels)
                acc = (predicted == final_labels).float().mean().item()
            
            duration = time.time() - start_time
            log_msg = f"{dt_string} [{step:07d}] Loss: {loss.item() * params.grad_accum_steps:.4f} Acc(train_logic): {acc*100:.2f} Time: {format_time(duration)}"
            print(log_msg)
            
            with open(os.path.join(exp_path, 'logs.txt'), 'a') as f:
                f.write(log_msg + "\n")

        # --- Évaluation (logique de progan_baseline / eval_UniversalFakeDetect) ---
        if step % params.eval_freq == 0:
            if not eval_loaders:
                print("Avertissement: Aucun dataset d'évaluation chargé, évaluation ignorée.")
                continue

            print(f"\n--- Évaluation Multi-Dataset (Étape {step}) ---")
            
            all_eval_accs = []
            all_eval_aucs = []
            all_eval_aps = []

            with open(os.path.join(exp_path, 'eval_results.txt'), 'a') as f:
                f.write(f"--- Étape {step} Multi-Dataset ---\n")
                
            for dataset_name, eval_loader in eval_loaders.items():
                # L'évaluation utilise la tâche binaire standard (Fake=0, Real=1)
                eval_acc, eval_auc, eval_ap = evaluate(model, eval_loader, device, params.eval_batch, dataset_name)
                
                log_line = f"  {dataset_name:15s}: ACC {eval_acc:6.2f}% | AUC {eval_auc:6.2f}% | AP {eval_ap:6.2f}%"
                print(log_line)
                with open(os.path.join(exp_path, 'eval_results.txt'), 'a') as f:
                    f.write(log_line + "\n")
                    
                all_eval_accs.append(eval_acc)
                all_eval_aucs.append(eval_auc if eval_auc > 0 else 0) 
                all_eval_aps.append(eval_ap if eval_ap > 0 else 0) 

            # Calculer les moyennes
            avg_acc = np.mean(all_eval_accs)
            avg_auc = np.mean(all_eval_aucs)
            avg_ap = np.mean(all_eval_aps)
            
            avg_log = f"\n--- Moyenne (Étape {step}): ACC {avg_acc:.2f}% | AUC {avg_auc:.2f}% | AP {avg_ap:.2f}% ---"
            print(avg_log)
            with open(os.path.join(exp_path, 'eval_results.txt'), 'a') as f:
                f.write(avg_log + "\n\n")

            if avg_acc > best_acc:
                best_acc = avg_acc
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "val_acc": avg_acc,
                    "val_auc": avg_auc,
                    "val_ap": avg_ap
                }, os.path.join(model_dir, "best_model.pth"))
                print(f"*** Nouveau meilleur modèle sauvegardé (Acc Moyen: {avg_acc:.2f}%) ***")

            # Remettre le modèle en mode entraînement
            model.train() 

        # --- Sauvegarde du Checkpoint ---
        if step % params.save_model_freq == 0:
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step
            }, os.path.join(model_dir, f"checkpoint-{step:06d}.pth"))

    print(f"Entraînement terminé, meilleure précision moyenne de validation: {best_acc:.2f}%")

if __name__ == "__main__":   
    parser = get_parser()
    params = parser.parse_args()
        
    main(params)