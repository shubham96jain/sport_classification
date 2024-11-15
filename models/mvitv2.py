import torch
import torch.nn as nn
import torchvision.models.video as models

class MViTClassifier(nn.Module):
    def __init__(self, num_classes=400, pretrained=True):
        super(MViTClassifier, self).__init__()
        
        if pretrained:
            # Use pretrained weights from Kinetics-400
            weights = models.MViT_V2_S_Weights.DEFAULT
            self.model = models.mvit_v2_s(weights=weights)
            # Freeze all layers except head
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            # Initialize with random weights
            self.model = models.mvit_v2_s(weights=None)
        


        # Enable gradient checkpointing
        # self.model.use_checkpoint = True  # This can reduce memory usage significantly
        
        # Modify the head to match your number of classes
        # MViT has a different structure than R3D, accessing head directly
        in_features = self.model.head[1].in_features
        self.model.head[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # MViT expects input shape: (B, C, T, H, W)
        return self.model(x)