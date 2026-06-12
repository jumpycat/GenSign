import torch
import torch.nn as nn
import torch.nn.functional as F

class DenoisingFCNWithSkip(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=64, num_blocks=8):
        super(DenoisingFCNWithSkip, self).__init__()
        
        self.num_blocks = num_blocks
        
        self.input_conv = nn.Conv2d(in_channels, features, kernel_size=3, padding=1)
        self.input_gn = nn.GroupNorm(num_groups=16, num_channels=features)
        
        self.res_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.res_blocks.append(ResBlock(features, groups=16))
        
        self.output_conv = nn.Conv2d(features, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        out = F.gelu(self.input_gn(self.input_conv(x)))
        
        for block in self.res_blocks:
            out = block(out)
        
        predicted_noise = self.output_conv(out)
        
        return predicted_noise
    
    def denoise(self, noisy_image):
        with torch.no_grad():
            predicted_noise = self.forward(noisy_image)
            clean_image = noisy_image - predicted_noise
            clean_image = torch.clamp(clean_image, 0.0, 1.0)
        return clean_image


class ResBlock(nn.Module):
    def __init__(self, features, groups=16):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(num_groups=groups, num_channels=features)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(num_groups=groups, num_channels=features)
        
    def forward(self, x):
        residual = x
        out = F.gelu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += residual
        return F.gelu(out)


if __name__ == "__main__":
    model = DenoisingFCNWithSkip(in_channels=3, out_channels=3, features=128, num_blocks=8)
    
    print(f"{sum(p.numel() for p in model.parameters()):,}")
    
    batch_size = 4
    height, width = 256, 256
    channels = 3
    
    noisy_image = torch.randn(batch_size, channels, height, width)
    
    model.eval()
    predicted_noise = model(noisy_image)
    
    print(f"{noisy_image.shape}")
    print(f"{predicted_noise.shape}")
    
    clean_image = model.denoise(noisy_image)
    print(f"{clean_image.shape}")
