import os
import glob
import random
from collections import defaultdict
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

# --- Transforms ---
class ConditionalResize(object):
    """
    Serializable transform
    """
    def __init__(self, size):
        self.size = size
        self.resize_op = transforms.Resize(size)

    def __call__(self, img):
        if min(img.size) < self.size:
            return self.resize_op(img)
        return img

# --- Dataset Classes ---
class CustomTrainDataset(Dataset):
    """
    Loads custom training data (e.g., progan_train).
    Labels: 0=fake/ai, 1=real/nature
    """
    def __init__(self, root_dir, transform=None, subfolders_to_use=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.subfolders_to_use = subfolders_to_use
        
        self.label_map = {'0_real': 1, '1_fake': 0}
        
        print(f"=== Loading custom training data: {root_dir} ===")
        
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Training directory not found: {root_dir}")
            
        all_semantic_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        
        if self.subfolders_to_use:
            print(f"Total of {len(all_semantic_dirs)} semantic folders found.")
            semantic_dirs = [d for d in all_semantic_dirs if d in self.subfolders_to_use]
            print(f"Using the {len(semantic_dirs)} specified folders: {semantic_dirs}")
            
            missing = [d for d in self.subfolders_to_use if d not in all_semantic_dirs]
            if missing:
                print(f"Warning: Specified folders not found in {root_dir}: {missing}")
        else:
            semantic_dirs = all_semantic_dirs
            print(f"Found {len(semantic_dirs)} semantic folders (all will be used), searching for '0_real' and '1_fake'...")

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
        
        print(f"Total of {len(self.samples)} training samples loaded.")
        counts = defaultdict(int)
        for _, label in self.samples:
            counts[label] += 1
        print(f"Sample distribution: {dict(counts)} (0=fake, 1=real)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Failed to load image {img_path}, {e}. Replacing with a random index.")
            new_idx = random.randint(0, len(self) - 1)
            return self[new_idx]
            
        if self.transform:
            img = self.transform(img)
        
        return img, label

class SpecificEvalDataset(Dataset):
    """
    Loads specified real and fake folders for evaluation.
    - real_dir -> label 1 (real/nature)
    - fake_dir -> label 0 (fake/ai)
    """
    def __init__(self, real_dir, fake_dir, transform=None):
        self.transform = transform
        self.samples = []
        label_map = {real_dir: 1, fake_dir: 0}

        print(f"    > Loading Real: {real_dir} (Label 1)")
        print(f"    > Loading Fake: {fake_dir} (Label 0)")

        image_extensions = ('.png', '.jpg', '.jpeg', '.JPEG', '.webp', '.tif')

        for data_path, label in label_map.items():
            if not os.path.exists(data_path):
                print(f"Warning: Path not found, skipped: {data_path}")
                continue

            path_files = []
            for ext in image_extensions:
                path_files.extend(glob.glob(os.path.join(data_path, '**', f"*{ext}"), recursive=True))

            for f in path_files:
                self.samples.append((f, label))

        counts = defaultdict(int)
        for _, label in self.samples:
            counts[label] += 1
        print(f"    > Loading completed: {len(self.samples)} samples. Distribution: {dict(counts)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Failed to load image {img_path}, {e}. Replacing with a random index.")
            new_idx = random.randint(0, len(self) - 1)
            return self[new_idx]

        if self.transform:
            img = self.transform(img)

        return img, label

# --- Helper Functions for Loading ---
def load_genimg_eval_datasets(eval_datasets_root, val_transform):
    """
    Loads evaluation datasets for GenImage.
    """
    eval_loaders = {}
    eval_info = {}
    
    if not os.path.exists(eval_datasets_root):
        print(f"Warning: Evaluation datasets root directory not found: {eval_datasets_root}")
        return eval_loaders, eval_info
    
    print(f"\n=== Loading evaluation datasets ===")
    print(f"Root directory: {eval_datasets_root}")
    
    dataset_folders = [item for item in os.listdir(eval_datasets_root) if os.path.isdir(os.path.join(eval_datasets_root, item))]
    dataset_folders.sort()
    print(f"Found {len(dataset_folders)} dataset folders: {dataset_folders}")
    
    for dataset_name in dataset_folders:
        dataset_path = os.path.join(eval_datasets_root, dataset_name)
        print(f"\n--- Processing dataset: {dataset_name} ---")
        
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
            print(f"  Chosen path: {selected_path} (Type: {selected_path_type})")
            
            try:
                subfolders = [f for f in os.listdir(selected_path) if os.path.isdir(os.path.join(selected_path, f))]
                print(f"  Subfolders: {subfolders}")
                
                if 'ai' in subfolders and 'nature' in subfolders:
                    dataset = ImageFolder(selected_path, transform=val_transform)
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
                    
                    print(f"  ✓ Loaded successfully")
                    print(f"     Class mapping: {class_to_idx}")
                    print(f"     Sample counts: {class_name_counts}")
                else:
                    print(f"  ✗ Skipped - 'ai' and 'nature' folders not found (Found: {subfolders})")
            except Exception as e:
                print(f"  ✗ Loading failed: {str(e)}")
        else:
            print(f"  ✗ Skipped - Valid 'val' path not found")
    
    print(f"\n=== Evaluation datasets loading completed ===")
    print(f"{len(eval_loaders)} datasets loaded successfully: {list(eval_loaders.keys())}")
    
    return eval_loaders, eval_info

def get_ufd_eval_loaders(batch_size, val_transform):
    """
    Returns the dataloaders for the Universal Fake Detect evaluation task.
    """
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
    
    eval_loaders = {}
    for dataset_name, paths in EVAL_DATASETS.items():
        print(f"--- Loading evaluation: {dataset_name} ---")
        dataset = SpecificEvalDataset(paths['real_path'], paths['fake_path'], transform=val_transform)
        if len(dataset) == 0:
            print(f"  > Warning: {dataset_name} loaded no samples, it will be skipped.")
            continue
        loader = DataLoader(
            dataset,
            batch_size=batch_size * 2, 
            shuffle=True, 
            num_workers=8,
            persistent_workers=False,
            pin_memory=True
        )
        eval_loaders[dataset_name] = loader
    print(f"{len(eval_loaders)} evaluation datasets loaded successfully.")
    return eval_loaders