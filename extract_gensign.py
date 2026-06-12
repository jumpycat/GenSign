import argparse
import os
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from gensign_extractor import DenoisingFCNWithSkip


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default='denosier_250824225428-400000.pth')
    parser.add_argument("--img_size", type=int, default=256)

    return parser


def build_transform(img_size):
    tfms = []
    if img_size and img_size > 0:
        tfms.append(lambda img: transforms.Resize(img_size)(img) if min(img.size) < img_size else img)
        tfms.append(transforms.CenterCrop(img_size))
    tfms.append(transforms.ToTensor())
    tfms.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    return transforms.Compose(tfms)


def load_noiser(checkpoint_path, features, num_blocks):
    noiser = DenoisingFCNWithSkip().to(device)

    print(f"loading {checkpoint_path} ...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "noiser" in checkpoint:
        state_dict = checkpoint["noiser"]
    else:
        state_dict = checkpoint

    noiser.load_state_dict(state_dict)
    noiser.eval()
    return noiser


@torch.no_grad()
def extract_noise(noiser, x):
    pred_noise = noiser(x)
    noise_frac = pred_noise - torch.floor(pred_noise)
    return noise


def main(params):
    os.makedirs(params.output_dir, exist_ok=True)

    noiser = load_noiser(params.checkpoint, params.features, params.num_blocks)
    transform = build_transform(params.img_size)

    img_files = sorted(
        f for f in os.listdir(params.input_dir)
        if f.lower().endswith(IMG_EXTENSIONS)
    )


    for fname in img_files:
        img_path = os.path.join(params.input_dir, fname)
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            continue

        x = transform(img).unsqueeze(0).to(device)  # [1, 3, H, W]

        noise_frac = extract_noise(noiser, x)

        name, _ = os.path.splitext(fname)
        save_path = os.path.join(params.output_dir, f"{name}_noise.png")
        save_image(noise_frac.squeeze(0).cpu(), save_path)

        print(f"[OK] {fname} -> {save_path}")


if __name__ == "__main__":
    parser = get_parser()
    params = parser.parse_args()
    main(params)