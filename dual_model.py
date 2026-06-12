import torch
import torch.nn as nn
from torchvision import transforms
from transformers import CLIPModel
import loralib as lora
from efficientnet_pytorch import EfficientNet

def denorm(x):
    """ Denormalizes from VQGAN [-1, 1] to [0, 1] """
    return (x + 1.0) / 2.0

def replace_with_lora(module, rank):
    """ Replaces an nn.Linear layer with a lora.Linear layer and copies weights. """
    lora_layer = lora.Linear(
        module.in_features,
        module.out_features,
        r=rank,
        lora_alpha=rank, 
        lora_dropout=0.1, # Can be adjusted
        bias=(module.bias is not None)
    )
    # Copy original weights and bias
    lora_layer.weight.data.copy_(module.weight.data)
    if module.bias is not None:
        lora_layer.bias.data.copy_(module.bias.data)
    return lora_layer

class DualStreamModel(nn.Module):
    def __init__(self, stream1_model_name, stream2_model_name, noiser, lora_rank, clip_input_size=224):
        """
        Initializes the dual-stream model.
        
        Args:
            stream1_model_name (str): Name of the CLIP model (e.g., 'openai/clip-vit-large-patch14').
            stream2_model_name (str): Name of the EfficientNet model (e.g., 'efficientnet-b0').
            noiser (nn.Module): Pre-trained instance of DenoisingFCNWithSkip.
            lora_rank (int): Rank to use for LoRA on the CLIP backbone.
            clip_input_size (int): Input size for the CLIP backbone (e.g., 224).
        """
        super(DualStreamModel, self).__init__()

        print(f"Initializing Dual-Stream Model:")
        
        # --- Stream 1: CLIP + LoRA ---
        print(f"  Stream 1: Loading {stream1_model_name}...")
        try:
            clip_model = CLIPModel.from_pretrained(stream1_model_name)
            self.clip_backbone = clip_model.vision_model
            # Get dim robustly
            self.clip_feat_dim = clip_model.config.vision_config.hidden_size 
        except Exception as e:
            print(f"ERROR: Failed to load CLIP model from Hugging Face: {e}")
            print("Please check your connection or HF_ENDPOINT.")
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
            # We initialize with num_classes=1 for Stream 2 compatibility, then override _fc
            self.effnet_backbone = EfficientNet.from_pretrained(stream2_model_name, num_classes=1)
            self.effnet_feat_dim = self.effnet_backbone._fc.in_features
            self.effnet_backbone._fc = nn.Identity() 
        except Exception as e:
            print(f"ERROR: Failed to load EfficientNet: {e}")
            raise e
        print(f"  Stream 2: EfficientNet backbone (dim={self.effnet_feat_dim}) ready.")

        # --- Noiser (for Stream 2) ---
        print(f"  Stream 2: Attaching Noiser...")
        self.noiser = noiser
        # Freeze the noiser - it's a fixed feature extractor
        self.noiser.eval()
        for param in self.noiser.parameters():
            param.requires_grad = False
        print(f"  Stream 2: Noiser attached and frozen.")

        # --- Pre-processing transforms (within the model) ---
        self.crop224 = transforms.CenterCrop(clip_input_size)
        # REQ 3: CLIP specific normalization
        self.normalize_clip = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073], 
            std=[0.26862954, 0.26130258, 0.27577711]
        )

        # --- Fusion Head (Shallow Classifier) ---
        self.fusion_dim = self.clip_feat_dim + self.effnet_feat_dim
        print(f"  Fusion: CLS={self.clip_feat_dim}, EFF={self.effnet_feat_dim}, Total={self.fusion_dim}")
        # REQ 6: Shallow classifier
        self.fusion_head = nn.Sequential(
            nn.Linear(self.fusion_dim, 256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(256, 1)
        )        
        # Ensure the fusion head is trainable
        for param in self.fusion_head.parameters():
            param.requires_grad = True

        print("Dual-Stream Model initialized.")

    def forward(self, x):
        # x is (B, 3, 256, 256), VQGAN normalized (-1 to 1)

        # --- Stream 1: CLIP ---
        # REQ 3: Denormalize VQGAN [-1, 1] -> [0, 1]
        x_0_1 = denorm(x)
        # REQ 1: Center crop to 224
        x_224 = self.crop224(x_0_1)
        # REQ 3: Apply CLIP normalization
        x_clip = self.normalize_clip(x_224)
        # REQ 5: Get feature vector
        feat1 = self.clip_backbone(x_clip)['pooler_output'] # (B, 1024)

        # --- Stream 2: Residual ---
        with torch.no_grad(): # Applied based on usage context
            residual = self.noiser(x)
        residual_fp = residual - residual.floor() # (B, 3, 256, 256)
        
        # REQ 5: Extract EfficientNet features
        feat2 = self.effnet_backbone.extract_features(residual_fp)
        feat2 = self.effnet_backbone._avg_pooling(feat2) # (B, 1280, 1, 1)
        feat2 = feat2.flatten(start_dim=1) # (B, 1280)
        
        # --- REQ 6: Fusion ---
        combined_feat = torch.cat([feat1, feat2], dim=1) # (B, 2304)
        output = self.fusion_head(combined_feat) # (B, 1)
        
        return output