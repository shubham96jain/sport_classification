import torch
import torch.nn as nn
import torchvision.models.video as models

class S3DV2Classifier(nn.Module):
    def __init__(self, num_classes=400, pretrained=True, hidden_size=512, num_layers=2):
        super(S3DV2Classifier, self).__init__()
        
        if pretrained:
            weights = models.S3D_Weights.DEFAULT
            self.model = models.s3d(weights=weights)
            # Freeze feature extractor
            for param in self.model.features.parameters():
                param.requires_grad = False
        else:
            self.model = models.s3d(weights=None)
            
        # Remove original classifier
        self.features = self.model.features
        
        # LSTM configuration
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Add LSTM layer
        self.lstm = nn.LSTM(
            input_size=1024,  # S3D feature size
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0,
        )
        
        # New classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            # nn.Linear(hidden_size, hidden_size * 2)
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        # x shape: (batch, channels, time, height, width)
        batch_size = x.size(0)
        
        # Get S3D features
        features = self.features(x)  # (batch, 1024, T, H, W)
        
        # Global average pool spatial dimensions
        features = torch.mean(features, dim=[3, 4])  # (batch, 1024, T)
        
        # Prepare for LSTM
        features = features.permute(0, 2, 1)  # (batch, T, 1024)
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(features)  # (batch, T, hidden_size*2)
        
        # Use last time step output
        final_hidden = lstm_out[:, -1, :]  # (batch, hidden_size*2)
        
        # Classify
        output = self.classifier(final_hidden)
        
        return output