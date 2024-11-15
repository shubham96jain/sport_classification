import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from pathlib import Path
import json
from sklearn.metrics import classification_report, confusion_matrix
# import seaborn as sns
# import matplotlib.pyplot as plt
import numpy as np

from dataset import CustomVideoDataset
from models.resnet3d import ResNet3DClassifier
from models.mvitv2 import MViTClassifier
from models.swin3d import Swin3DClassifier
from models.s3d import S3DClassifier
from models.s3dv2 import S3DV2Classifier
from config import Config

class TestConfig:
    # Testing specific configurations
    BATCH_SIZE = 8
    NUM_WORKERS = 16
    MODEL_PATH = "/teamspace/studios/this_studio/video_classification/runs/training/resnet_3d_fcp_8_lr_1e-5/best_model.pth"  # Path to saved model
    SAVE_PREDICTIONS = False  # Whether to save predictions to file
    OUTPUT_DIR = "test_results"  # Directory to save results
    FRAMES_PER_CLIP = 128
    STEPS_BETWEEN_CLIPS = 128
    IMAGE_SIZE = 112
    DEVICE = Config.DEVICE
    MODEL = "resnet3d"
    NUM_CLASSES = Config.NUM_CLASSES
    OUTPUT_DIR = "results"

def create_model(config):
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
    
    # Load model weights
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=TestConfig.DEVICE))
    return model.to(config.DEVICE)

# def plot_confusion_matrix(cm, class_names, output_path):
#     """Plot and save confusion matrix."""
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=class_names,
#                 yticklabels=class_names)
#     plt.title('Confusion Matrix')
#     plt.ylabel('True Label')
#     plt.xlabel('Predicted Label')
#     plt.xticks(rotation=45)
#     plt.yticks(rotation=45)
#     plt.tight_layout()
#     plt.savefig(output_path)
#     plt.close()

def test(config):
    # Create output directory
    output_dir = Path(TestConfig.OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Create dataset and dataloader
    test_dataset = CustomVideoDataset(
        root=Config.DATA_PATH,
        mode="test",
        frames_per_clip=config.FRAMES_PER_CLIP,
        steps_between_clips=config.STEPS_BETWEEN_CLIPS,
        num_workers=config.NUM_WORKERS
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )

    # Load model
    model = create_model(config)
    model.eval()

    # Initialize lists to store predictions and ground truth
    all_preds = []
    all_labels = []
    all_paths = []
    
    print("Starting evaluation...")
    with torch.no_grad():
        for videos, labels, paths in tqdm(test_loader):
            videos = videos.to(config.DEVICE)
            outputs = model(videos)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_paths.extend(paths)

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate metrics
    class_names = list(Config.CLASSES.keys())
    classification_rep = classification_report(
        all_labels, 
        all_preds, 
        target_names=class_names,
        digits=4
    )
    conf_matrix = confusion_matrix(all_labels, all_preds)
    final_accuracy = 100 * np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix)

    # Save results
    results = {
        'classification_report': classification_rep,
        'confusion_matrix': conf_matrix.tolist(),
        'overall_accuracy': final_accuracy,
        'predictions': {
            'video_paths': all_paths,
            'predicted_labels': all_preds.tolist(),
            'true_labels': all_labels.tolist()
        }
    }

    # Save results to files
    with open(output_dir / 'test_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    with open(output_dir / 'classification_report.txt', 'w') as f:
        f.write(classification_rep)

    # Plot and save confusion matrix
    # plot_confusion_matrix(
    #     conf_matrix, 
    #     class_names,
    #     output_dir / 'confusion_matrix.png'
    # )

    print("\nTest Results:")
    print(classification_rep)
    print("\nResults saved to:", output_dir)

def main():
    parser = argparse.ArgumentParser(description='Test a trained video classification model')
    parser.add_argument('--model_path', type=str, default=TestConfig.MODEL_PATH,
                        help='Path to the trained model weights')
    parser.add_argument('--batch_size', type=int, default=TestConfig.BATCH_SIZE,
                        help='Batch size for testing')
    parser.add_argument('--output_dir', type=str, default=TestConfig.OUTPUT_DIR,
                        help='Directory to save test results')
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    TestConfig.MODEL_PATH = args.model_path
    TestConfig.BATCH_SIZE = args.batch_size
    TestConfig.OUTPUT_DIR = args.output_dir

    test(TestConfig)

if __name__ == '__main__':
    main()