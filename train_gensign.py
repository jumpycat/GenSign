import argparse
import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from datetime import datetime, timedelta
import importlib
from model import DenoisingFCNWithSkip
import cv2
from torch.utils.data import Dataset


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def denorm(x):
    # [-1,1] -> [0,1]
    return (x + 1.0) / 2.0

def format_time(seconds):
    delta = timedelta(seconds=seconds)
    days = delta.days
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{days}d {hours}h {minutes}m {seconds}s'



def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_dir", type=str, default="CoCo/coco_train/train_2017")
    parser.add_argument("--batch_size", type=int, default=8)

    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--steps", type=int, default=400010)
    parser.add_argument("--encoder_module", type=str, default="dair.encoder2")
    parser.add_argument("--decoder_module", type=str, default="dair.decoder24_with_graph.py")


    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument("--save_model_freq", type=int, default=100000)
    parser.add_argument("--save_image_freq", type=int, default=5000)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad_accum_steps", type=int, default=16)
    parser.add_argument("--resume", type=str, default=None)


    return parser

def main(params):
    torch.manual_seed(params.seed)
    np.random.seed(params.seed)
    random.seed(params.seed)

    now = datetime.now()
    dt_string = now.strftime("%y%m%d%H%M%S")
    exp_name = f"denosier_{dt_string}_bs{params.batch_size}_ga{params.grad_accum_steps}_lr{params.lr}"

    exp_path = os.path.join("runs", exp_name)
    model_dir = os.path.join(exp_path, "models")
    img_dir = os.path.join(exp_path, "imgs")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    with open(os.path.join(exp_path, 'args.txt'), 'w') as f:
        for arg in vars(params):
            f.write(f"{arg}: {getattr(params, arg)}\n")
        f.write(f"Command: {' '.join(sys.argv)}\n")

    normalize_vqgan = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalize (x - 0.5) / 0.5
    transform = transforms.Compose([
        lambda img: transforms.Resize(params.img_size)(img) if min(img.size) < params.img_size else img,
        transforms.RandomCrop(params.img_size),
        transforms.ToTensor(),
        normalize_vqgan
    ])
    dataset = ImageFolder(params.train_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=params.batch_size, shuffle=True, num_workers=16)


    decoder_module = importlib.import_module(params.decoder_module)
    Decoder_class = getattr(decoder_module, "Decoder")

    encoder_module = importlib.import_module(params.encoder_module)
    Encoder_class = getattr(encoder_module, "Encoder")

    encoder = Encoder_class().to(device)
    decoder = Decoder_class().to(device)
    noiser = DenoisingFCNWithSkip().to(device)

    encoder.eval()
    decoder.eval()
    noiser.train()

    img_mse = nn.MSELoss()

    optimizer = torch.optim.Adam(noiser.parameters(), lr=params.lr)


    if params.resume is not None and os.path.isfile(params.resume):
        checkpoint = torch.load(params.resume, map_location=device)
        encoder.load_state_dict(checkpoint["encoder"])
        decoder.load_state_dict(checkpoint["decoder"])
        print(f"done")


    start_time = time.time()
    data_iter = iter(loader)

    for step in range(1, params.steps + 1):
        try:
            imgs, _ = next(data_iter)
        except:
            data_iter = iter(loader)
            imgs, _ = next(data_iter)

        imgs = imgs.to(device)

        with torch.no_grad():
            latents = encoder(imgs)
            recon_imgs = decoder(latents)
            gt_noise = imgs - recon_imgs

        prd_noise = noiser(imgs)

        loss_mse = img_mse(prd_noise, gt_noise)

        loss = loss_mse / params.grad_accum_steps  # Normalize loss
        loss.backward()

        if step % params.grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        if step % params.log_freq == 0:
            duration = time.time() - start_time
            log_msg = f"{dt_string} [{step:07d}] "
            log_msg += f"MSE Loss: {loss_mse.item():.5f} "
            log_msg += f"Time: {format_time(duration)}"
            print(log_msg)

            with open(os.path.join(exp_path, 'logs.txt'), 'a') as f:
                f.write(log_msg + "\n")

        if step % params.save_image_freq == 0:
            with torch.no_grad():
                imgs_vis = imgs[:8]
                recons_vis = (prd_noise[:8] - prd_noise[:8].min()) / (prd_noise[:8].max() - prd_noise[:8].min() + 1e-8)

                grid = torch.cat([denorm(imgs_vis), recons_vis], dim=0)
                save_path = os.path.join(img_dir, f"recon_step{step:07d}.jpg")
                save_image(grid, save_path, nrow=4)

        if step % params.save_model_freq == 0:
            torch.save({
                "noiser": noiser.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "step": step
            }, os.path.join(model_dir, f"checkpoint-{step:06d}.pth"))

if __name__ == "__main__":
    parser = get_parser()
    params = parser.parse_args()
    main(params)