import torch
import argparse
from pathlib import Path
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image

from models.resnet3d import ResNet3DClassifier
from models.mvitv2 import MViTClassifier
from models.swin3d import Swin3DClassifier
from models.s3d import S3DClassifier
from models.s3dv2 import S3DV2Classifier
from config import Config

class InferenceConfig:
    MODEL_PATH = "/teamspace/studios/this_studio/video_classification/runs/training/resnet_3d_fcp_8_lr_1e-5/best_model.pth"
    MODEL = "resnet3d"
    DEVICE = Config.DEVICE
    NUM_CLASSES = Config.NUM_CLASSES
    FRAMES_PER_CLIP = 128
    IMAGE_SIZE = 112
    MEAN = [0.45, 0.45, 0.45]
    STD = [0.225, 0.225, 0.225]

def load_model(config):
    """Create and load the model based on configuration."""
    if config.MODEL == 'resnet3d':
        model = ResNet3DClassifier(num_classes=config.NUM_CLASSES)
    elif config.MODEL == 'mvitv2':
        model = MViTClassifier(num_classes=config.NUM_CLASSES)
    elif config.MODEL == 'swin3d':
        model = Swin3DClassifier(num_classes=config.NUM_CLASSES)
    elif config.MODEL == 's3d':
        model = S3DClassifier(num_classes=config.NUM_CLASSES)
    elif config.MODEL == 's3dv2':
        model = S3DV2Classifier(num_classes=config.NUM_CLASSES, 
                              hidden_size=512,
                              num_layers=2)
    else:
        raise ValueError(f"Unknown model: {config.MODEL}")
    
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=config.DEVICE))
    model = model.to(config.DEVICE)
    model.eval()
    return model

def preprocess_video(video_path, config):
    """Load and preprocess video for inference."""
    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    # Read video
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = transform(frame)
        frames.append(frame)
        frame_count += 1
    
    cap.release()
    
    # Handle videos shorter than FRAMES_PER_CLIP
    if frame_count < config.FRAMES_PER_CLIP:
        # Duplicate last frame until we reach FRAMES_PER_CLIP
        last_frame = frames[-1]
        while len(frames) < config.FRAMES_PER_CLIP:
            frames.append(last_frame)
    
    # Handle videos longer than FRAMES_PER_CLIP
    elif frame_count > config.FRAMES_PER_CLIP:
        # Sample frames uniformly
        indices = np.linspace(0, len(frames)-1, config.FRAMES_PER_CLIP, dtype=int)
        frames = [frames[i] for i in indices]
    
    # Stack frames into tensor
    frames = torch.stack(frames)  # Shape: [T, C, H, W]
    frames = frames.permute(1, 0, 2, 3)  # Shape: [C, T, H, W]
    return frames.unsqueeze(0)  # Add batch dimension

def inference(video_path, config):
    """Run inference on a single video."""
    # Load model
    model = load_model(config)
    
    # Preprocess video
    video_tensor = preprocess_video(video_path, config)
    video_tensor = video_tensor.to(config.DEVICE)
    
    # Run inference
    with torch.no_grad():
        outputs = model(video_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
    # Get prediction
    predicted_class = list(Config.CLASSES.keys())[predicted.item()]
    confidence = confidence.item() * 100
    
    return predicted_class, confidence

def main():
    parser = argparse.ArgumentParser(description='Run inference on a video')
    parser.add_argument('video_path', type=str, help='Path to input video file')
    parser.add_argument('--model_path', type=str, default=InferenceConfig.MODEL_PATH,
                      help='Path to trained model weights')
    parser.add_argument('--model', type=str, default=InferenceConfig.MODEL,
                      help='Model architecture to use')
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    InferenceConfig.MODEL_PATH = args.model_path
    InferenceConfig.MODEL = args.model
    
    # Run inference
    try:
        predicted_class, confidence = inference(args.video_path, InferenceConfig)
        print(f"\nResults for: {args.video_path}")
        print(f"Predicted Sport: {predicted_class}")
        print(f"Confidence: {confidence:.2f}%")
    except Exception as e:
        print(f"Error processing video: {str(e)}")

if __name__ == '__main__':
    main()