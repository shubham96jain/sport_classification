import torch

class Config:

    SEED = 42
    # Dataset
    DATA_PATH = "data/kinetics_processed_224x224"
    SMALL_DATASET = False # Always False unless explicitly user wants to load a subset train and val sets
    NUM_CLASSES = 4
    FRAMES_PER_CLIP = 128 # 8 for resnet3d/2d+1 based models and 128 for s3d models
    STEPS_BETWEEN_CLIPS = 128 # Usually same as Frames per clip unless user wants overlap between clips
    IMAGE_SIZE = 224 # Configure image size. Height and width are kept same.

    # Training
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 15
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LR_STEP_SIZE = 1
    LR_GAMMA = 0.9
    NUM_WORKERS = 16
    
    # Model
    PRETRAINED = True
    MODEL = 's3dv2'# Available choices 's3d', 's3dv2', 'resnet3d', 'swin3d', 'mvitv2'

    #Classes
    CLASSES = {
    'baseball': 0,
    'basketball': 1,
    'cricket': 2,
    'soccer': 3
}
    NUM_TO_CLASSES = {
    0:'baseball',
    1:'basketball',
    2:'cricket',
    3:'soccer'
}