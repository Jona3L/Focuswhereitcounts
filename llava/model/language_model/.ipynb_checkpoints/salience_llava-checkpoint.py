import torch
import torch.nn as nn
import torch.nn.functional as F

from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
import sys
sys.path.append("/scratch/jl9356/salience_llava")
from QAGNet_main.single_wrapper import SalienceSingleImageWrapper

import pdb

class MySaliencyLlamaForCausalLM(LlavaLlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self._sal_wrapper = None
        self.sal_projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(224 * 224, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4096),
        )

    def set_sal_wrapper(self, wrapper: SalienceSingleImageWrapper):
        """
        Explicitly attach a saliency wrapper. 
        This overrides any automatic creation in encode_images().
        """
        self._sal_wrapper = wrapper


    def encode_images(self, images):
        """
        1) Call LLaVA's normal encode_images to get patch embeddings.  
        2) Optionally fuse your saliency feature if desired.  
        """

        # First, run the standard LLaVA pipeline
        base_embeds = super().encode_images(images)
        # shape: [B, num_patches, hidden_dim], e.g. [B, 576, 4096]

        # If you haven't already attached a sal_wrapper, create a default
        if self._sal_wrapper is None:
            self._sal_wrapper = SalienceSingleImageWrapper(
                config_file="/scratch/jl9356/salience_llava/QAGNet_main/configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml"
            )

        # Option A: Always fuse saliency at both training & inference
        # Option B: Only fuse saliency at inference if self.training is False
        #   if self.training:
        #       return base_embeds

        fused_list = []
        batch_size, _, hidden_dim = base_embeds.shape
        device = base_embeds.device

        for i in range(batch_size):
            vis_feat = base_embeds[i]  # [num_patches, hidden_dim], e.g. [576, 4096]
            sal_map = self._sal_wrapper.get_next_saliency()

            if sal_map is None:
                # If no saliency found, fallback to a dummy
                sal_map = torch.zeros((1, 1, 224, 224), dtype=base_embeds.dtype, device=device)
            else:
                # Convert & resize
                sal_map = sal_map.float().to(device)
                sal_map = F.interpolate(sal_map.unsqueeze(0).unsqueeze(0),
                                        size=(224, 224),
                                        mode='nearest')
                sal_map = sal_map.to(base_embeds.dtype)

            # MLP from [1,1,224,224] -> [1,4096]
            sal_embed = self.sal_projector(sal_map)

            # Fuse: cat along patch dimension => shape [576+1, 4096] = [577, 4096]
            fused_feat = torch.cat([vis_feat, sal_embed], dim=0)
            fused_list.append(fused_feat)

        # Re-stack => [B, 577, 4096]
        fused_embeds = torch.stack(fused_list, dim=0)
        return fused_embeds
    

    def get_model(self):
        return super().get_model()

    
