import torch
import torch.nn as nn
import torchvision.models.video as models

class ResNet3DClassifier(nn.Module):
    def __init__(self, num_classes=400, pretrained=True):
        super(ResNet3DClassifier, self).__init__()
        self.model = models.r3d_18(pretrained=pretrained)

        if pretrained:
            # Freeze all layers except final fc
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Modify the last fully connected layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)