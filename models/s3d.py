import torch
import torch.nn as nn
import torchvision.models.video as models

class S3DClassifier(nn.Module):
    def __init__(self, num_classes=400, pretrained=True):
        super(S3DClassifier, self).__init__()
        
        if pretrained:
            # Use pretrained weights from Kinetics-400
            weights = models.S3D_Weights.DEFAULT
            self.model = models.s3d(weights=weights)
            # Freeze all layers except classifier
            for param in self.model.features.parameters():
                param.requires_grad = False
        else:
            # Initialize with random weights
            self.model = models.s3d(weights=None)
        # Modify the classifier
        # The original classifier is a Sequential with Dropout and Conv3d
        in_features = 1024  # This is the number of input channels to the final conv
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Conv3d(in_features, num_classes, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        )

    def forward(self, x):
        # S3D expects input shape: (B, C, T, H, W)
        return self.model(x)