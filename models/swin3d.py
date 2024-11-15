import torch
import torch.nn as nn
import torchvision.models.video as models
import numpy as np

class Swin3DClassifier(nn.Module):
    def __init__(self, num_classes=400, pretrained=True):
        super(Swin3DClassifier, self).__init__()
        
        if pretrained:
            # Use pretrained weights from Kinetics-400
            weights = models.Swin3D_T_Weights.DEFAULT
            self.model = models.swin3d_t(weights=weights)
            # Freeze all layers except head
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            # Initialize with random weights
            self.model = models.swin3d_t(weights=None)
        
        
        # import pdb; pdb.set_trace()
        # Modify patch embedding for 112x112 input
        # self.model.patch_embed.proj = nn.Conv3d(3, 96, kernel_size=(2, 4, 4), stride=(2, 4, 4))
        
        # # Adjust window sizes for each stage
        # window_size = np.array([4, 4, 4])  # Half the original window size
        # for stage in self.model.features:
        #     if isinstance(stage, nn.Sequential):
        #         for block in stage:
        #             if hasattr(block, 'attn'):
        #                 import pdb; pdb.set_trace()
        #                 block.attn.window_size = window_size
        #                 # Also need to adjust shift size
        #                 if hasattr(block.attn, 'shift_size'):
        #                     block.attn.shift_size = window_size // 2
        
        # Enable gradient checkpointing
        # self.model.encoder.use_checkpoint = True  # This can reduce memory usage significantly
        # import pdb; pdb.set_trace()
        
        # Modify the head to match your number of classes
        in_features = self.model.head.in_features
        self.model.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # Swin3D expects input shape: (B, C, T, H, W)
        return self.model(x)