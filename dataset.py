
import random
from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
import torchvision.transforms as transforms
from torchvision.datasets import VisionDataset
from torchvision.datasets.video_utils import VideoClips
import numpy as np

from config import Config

class CustomVideoDataset(VisionDataset):
    def __init__(
        self,
        root: str,
        mode: str = "train",
        frames_per_clip: int = 8,
        steps_between_clips: int = 8,
        transform: Optional[Callable] = None,
        frame_rate: Optional[int] = None,
        _precomputed_metadata: Optional[dict] = None,
        num_workers: int = 4,
        output_format: str = "TCHW",
        seed: int = 42
    ):
        """Custom Video Dataset similar to Kinetics but for local files
        
        Args:
            root (str): Root directory path
            mode (str): 'train' or 'val' 
            frames_per_clip (int): Number of frames per clip
            steps_between_clips (int): Number of frames to skip between frames
            transform (callable, optional): Transform to be applied on video
            frame_rate (int, optional): Target frame rate. If None, keep original
            _precomputed_metadata (dict, optional): Precomputed metadata for VideoClips
            num_workers (int): Number of workers for VideoClips loading
            output_format (str): Format of output tensors ('TCHW' or 'THWC')
        """
        self.root = Path(root)
        self.mode = mode
        self.split_folder = self.root / mode
        self.frames_per_clip = frames_per_clip
        self.steps_between_clips = steps_between_clips
        self.frame_rate = frame_rate 
        self.num_workers = num_workers
        self.output_format = output_format
        self._precomputed_metadata = _precomputed_metadata
        self.seed = seed
        super().__init__(self.root, transform=transform)

        # Define main classes mapping
        self.classes = Config.CLASSES

        # Set seed for this instance
        random.seed(seed)
        np.random.seed(seed)
        
        # Get list of video files and their labels
        self.samples = []
        for class_path in self.split_folder.glob('*'):
            if class_path.is_dir() and class_path.name in self.classes.keys():
                for video_path in class_path.glob('*.mp4'):
                    self.samples.append((str(video_path), self.classes[class_path.name]))

        self.video_list = [x[0] for x in self.samples]

        self._load_or_create_video_clips()

        # Default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize(
                    mean=[0.45, 0.45, 0.45], 
                    std=[0.225, 0.225, 0.225]
                ),
                transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
                # transforms.RandomCrop(224) if mode == 'train' else transforms.CenterCrop(224),
                # transforms.RandomHorizontalFlip(p=0.5) if mode == 'train' else transforms.Lambda(lambda x: x)
            ])
        else:
            self.transform = transform
    
    def _load_or_create_video_clips(self):
        """Load VideoClips metadata from cache or create new"""
        # Create cache directory if it doesn't exist
        cache_dir = self.root / '.cache'
        cache_dir.mkdir(exist_ok=True)
        
        # Create a unique cache filename based on parameters
        cache_params = f"{self.mode}_{self.frames_per_clip}_{self.steps_between_clips}_{self.frame_rate}"
        cache_file = cache_dir / f"video_clips_metadata_{cache_params}.pt"
        
        if cache_file.exists():
            print(f"Loading VideoClips metadata from cache: {cache_file}")
            try:
                metadata = torch.load(cache_file)
                self.video_clips = VideoClips(
                    video_paths=self.video_list,
                    clip_length_in_frames=self.frames_per_clip,
                    frames_between_clips=self.steps_between_clips,
                    frame_rate=self.frame_rate,
                    _precomputed_metadata=metadata,
                    num_workers=self.num_workers,
                    output_format=self.output_format,
                )
                print("Successfully loaded cached metadata!")
                return
            except Exception as e:
                print(f"Failed to load cache: {str(e)}")
        
        print("Creating new VideoClips object and caching metadata...")
        # Create new VideoClips object
        self.video_clips = VideoClips(
            video_paths=self.video_list,
            clip_length_in_frames=self.frames_per_clip,
            frames_between_clips=self.steps_between_clips,
            frame_rate=self.frame_rate,
            _precomputed_metadata=self._precomputed_metadata,
            num_workers=self.num_workers,
            output_format=self.output_format,
        )
        
        # Cache the metadata
        try:
            metadata = self.video_clips.metadata
            torch.save(metadata, cache_file)
            print(f"Cached metadata to: {cache_file}")
        except Exception as e:
            print(f"Failed to cache metadata: {str(e)}")

    def _load_video(self, idx):

            video, audio, info, video_idx = self.video_clips.get_clip(idx)
            return video, video_idx

    def __len__(self) -> int:
        return self.video_clips.num_clips()

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Args:
            idx (int): Index
            
        Returns:
            tuple: (video, label) where video is a tensor of shape (C, T, H, W)
            and label is the class index
        """
        # Get video clip
        video, video_idx = self._load_video(idx)
        label = self.samples[video_idx][1]

        # Apply transforms
        if self.transform is not None:
            video = self.transform(video)

        # Reshape to (C, T, H, W) - ideo comes as (T, C, H, W)
        video = video.permute(1, 0, 2, 3)

        # return video, label, self.video_list[video_idx]
        return video, label, self.samples[video_idx][0]

    def get_classes(self):
        """Returns the class mapping dictionary"""
        return self.classes