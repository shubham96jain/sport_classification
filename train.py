import torch
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from dataset import CustomVideoDataset
from models.resnet3d import ResNet3DClassifier
from models.mvitv2 import MViTClassifier
from models.swin3d import Swin3DClassifier
from models.s3d import S3DClassifier
from models.s3dv2 import S3DV2Classifier
from config import Config
from utils import set_seed, untransform_video

if Config.DEVICE == 'cuda':
    scaler = torch.cuda.amp.GradScaler()


def train_one_epoch(model, dataloader, criterion, optimizer, device, writer, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    running_loss = 0

    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    for batch_idx, (videos, labels, video_paths) in pbar:
        videos, labels = videos.to(device), labels.to(device)
        optimizer.zero_grad()
        if Config.DEVICE == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(videos)
                loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        else:
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        running_loss = total_loss / (batch_idx + 1)  # Calculate running average
        # Update tqdm postfix with running statistics
        pbar.set_postfix({
            'loss': f'{running_loss:.4f}',
            'acc': f'{100.0 * correct / total:.2f}%'
        })

        # Store first batch data and predictions for visualization
        if batch_idx == 0 and writer is not None:
            _, predicted = outputs[:4].max(1)  # Get predictions for first 4 samples
            sample_data = {
                'videos': videos[:4].clone(),
                'labels': labels[:4].clone(),
                'predictions': predicted.clone(),
                'video_paths': video_paths[:4]
            }
            log_predictions(sample_data, writer, epoch, 'train')

    return total_loss / len(dataloader), 100. * correct / total

def validate(model, dataloader, criterion, device, writer, epoch):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Validation')
        for batch_idx, (videos, labels, video_paths) in pbar:
            videos, labels = videos.to(device), labels.to(device)
            
            outputs = model(videos)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            running_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix({
                'val_loss': f'{running_loss:.4f}',
                'val_acc': f'{100.0 * correct / total:.2f}%'
            })

            # Store first batch data and predictions for visualization
            if batch_idx == 0 and writer is not None:
                _, predicted = outputs[:4].max(1)  # Get predictions for first 4 samples
                sample_data = {
                    'videos': videos[:4].clone(),
                    'labels': labels[:4].clone(),
                    'predictions': predicted.clone(),
                    'video_paths': video_paths[:4]
                }
                log_predictions(sample_data, writer, epoch, 'val')

    return total_loss / len(dataloader), 100. * correct / total

def log_predictions(sample_data, writer, epoch, mode):
        # Log sample videos to tensorboard
    if sample_data is not None and writer is not None:
        # Convert predictions and true labels to class names
        pred_classes = [Config.NUM_TO_CLASSES[pred] for pred in sample_data['predictions'].cpu().numpy()]
        true_classes = [Config.NUM_TO_CLASSES[label] for label in sample_data['labels'].cpu().numpy()]

        videos = sample_data['videos'] # [C, T, H, W]
        videos = untransform_video(videos)
        
        # Log videos
        for i in range(min(4, len(pred_classes))):
            video = videos[i].cpu().permute(1, 0, 2, 3) # [T, C, H, W]
            writer.add_video(
                f'{mode}_sample_{i}_{sample_data["video_paths"][i]}/Pred:{pred_classes[i]}_True:{true_classes[i]}',
                video.unsqueeze(0), # add a batch dimension
                epoch,
                fps=4
            )

def main():

    set_seed(Config.SEED)
    # # Create dataset and dataloader
    train_dataset = CustomVideoDataset(root=Config.DATA_PATH, 
                                       mode="train",
                                       frames_per_clip=Config.FRAMES_PER_CLIP,
                                       steps_between_clips=Config.STEPS_BETWEEN_CLIPS,
                                       num_workers=Config.NUM_WORKERS, 
                                       seed=Config.SEED)

    val_dataset = CustomVideoDataset(root=Config.DATA_PATH, 
                                   mode="val",
                                   frames_per_clip=Config.FRAMES_PER_CLIP,
                                   steps_between_clips=Config.STEPS_BETWEEN_CLIPS,
                                   num_workers=Config.NUM_WORKERS, 
                                   seed=Config.SEED)
    
    if Config.SMALL_DATASET:
        # Create smaller datasets for testing
        n_train = 100  # Number of training samples you want
        n_val = 20     # Number of validation samples you want
        
        # Take random samples
        rng = np.random.RandomState(Config.SEED)
        train_indices = np.random.choice(len(train_dataset), n_train, replace=False)
        val_indices = np.random.choice(len(val_dataset), n_val, replace=False)
        
        # Create subset datasets
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS
    )

    print(train_loader)
    # Initialize model
    if Config.MODEL == 'resnet3d':
        model = ResNet3DClassifier(num_classes=Config.NUM_CLASSES, pretrained=Config.PRETRAINED)
    elif Config.MODEL == 'mvitv2':
        model = MViTClassifier(num_classes=Config.NUM_CLASSES, pretrained=Config.PRETRAINED)
    elif Config.MODEL == 'swin3d':
        model = Swin3DClassifier(num_classes=Config.NUM_CLASSES, pretrained=Config.PRETRAINED)
    elif Config.MODEL == 's3d':
        model = S3DClassifier(num_classes=Config.NUM_CLASSES, pretrained=Config.PRETRAINED)
    elif Config.MODEL == 's3dv2':
        model = S3DV2Classifier(num_classes=Config.NUM_CLASSES, 
        pretrained=Config.PRETRAINED,
        hidden_size=512,
        num_layers=2)
        
    model = model.to(Config.DEVICE)

    # Define loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = StepLR(optimizer, 
                      step_size=Config.LR_STEP_SIZE,  # e.g., 10 epochs
                      gamma=Config.LR_GAMMA)  

    # Initialize metrics dictionary to store history
    metrics = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    # Initialize tensorboard writer
    writer = SummaryWriter('runs/training')

    # Training loop
    best_val_acc = 0.0
    for epoch in range(Config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{Config.NUM_EPOCHS}")
        
        # Training phase
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, Config.DEVICE, writer, epoch
        )
        
        # Validation phase
        val_loss, val_acc = validate(
            model, val_loader, criterion, Config.DEVICE, writer, epoch
        )

        # Step the scheduler
        if isinstance(scheduler, StepLR):
            scheduler.step()

         # Log metrics to tensorboard
        writer.add_scalars('Loss', {
            'train': train_loss,
            'val': val_loss
        }, epoch)
        
        writer.add_scalars('Accuracy', {
            'train': train_acc,
            'val': val_acc
        }, epoch)
        
        # Print epoch results
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model (optional)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"New best model saved! Val Acc: {val_acc:.2f}%")

    writer.close()

if __name__ == "__main__":
    main()