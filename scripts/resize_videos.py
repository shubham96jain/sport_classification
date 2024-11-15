import subprocess
from pathlib import Path
import shutil
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

INPUT_ROOT = "data"
OUTPUT_ROOT = "data/kinetics_processed_224x224"

def get_video_dimensions(video_path):
    """Get video dimensions using ffprobe"""
    try:
        cmd = [
            'ffprobe', 
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height',
            '-of', 'csv=p=0',
            str(video_path)
        ]
        output = subprocess.check_output(cmd).decode()
        width, height = map(int, output.strip().split(','))
        return width, height
    except Exception as e:
        print(f"Error reading dimensions for {video_path}: {e}")
        return None, None

def resize_video(src_path, dst_path, target_size=(112, 112)):
    """
    Resize video if dimensions are larger than target size using ffmpeg
    Returns: True if video was resized, False if copied as-is
    """
    try:
        # Get original dimensions
        orig_width, orig_height = get_video_dimensions(src_path)
        if orig_width is None:
            return False
            
        # Check if resize is needed
        target_height, target_width = target_size
        if orig_height <= target_height and orig_width <= target_width:
            # Copy file as-is if it's smaller than target size
            shutil.copy2(src_path, dst_path)
            return False
        
        # Create output directory if it doesn't exist
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        
        # FFmpeg command for resizing
        cmd = [
            'ffmpeg',
            '-i', str(src_path),  # Input file
            '-vf', f'scale={target_width}:{target_height}',  # Scale video
            '-c:v', 'libx264',  # Video codec
            '-preset', 'medium',  # Encoding preset
            '-crf', '23',  # Quality (lower = better, 18-28 is good)
            '-c:a', 'copy',  # Copy audio stream
            '-y',  # Overwrite output file if exists
            str(dst_path)  # Output file
        ]
        
        # Run FFmpeg
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
        
    except Exception as e:
        print(f"Error processing {src_path}: {e}")
        return False


def process_videos(input_root, output_root, target_size=(112, 112), num_workers=4):
    """
    Process all videos maintaining directory structure
    """
    input_root = Path(input_root)
    output_root = Path(output_root)
    
    # Collect all video files
    video_files = []
    for split in ['train', 'val', 'test']:
        split_path = input_root / 'kinetics_processed' / split
        if not split_path.exists():
            continue
            
        for class_path in split_path.glob('*'):
            if not class_path.is_dir():
                continue
                
            for video_path in class_path.glob('*.mp4'):
                # Create corresponding output path
                rel_path = video_path.relative_to(split_path)
                output_path = output_root / split / rel_path
                video_files.append((video_path, output_path))
    
    # Create output directories
    for _, output_path in video_files:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Process videos in parallel
    total_videos = len(video_files)
    resized_count = 0
    copied_count = 0
    failed_count = 0
    
    print(f"Found {total_videos} videos to process")
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for src_path, dst_path in video_files:
            future = executor.submit(resize_video, src_path, dst_path, target_size)
            futures.append((future, src_path))
        
        # Monitor progress
        for future, src_path in tqdm(futures, total=len(futures), desc="Processing videos"):
            try:
                was_resized = future.result()
                if was_resized:
                    resized_count += 1
                else:
                    copied_count += 1
            except Exception as e:
                print(f"Failed to process {src_path}: {e}")
                failed_count += 1
    
    # Print statistics
    print("\nProcessing completed!")
    print(f"Total videos: {total_videos}")
    print(f"Resized: {resized_count}")
    print(f"Copied as-is: {copied_count}")
    print(f"Failed: {failed_count}")

if __name__ == "__main__":
    # Example usage
    input_root = Path(INPUT_ROOT)
    output_root = Path(OUTPUT_ROOT)
    target_size = (224, 224)
    
    process_videos(
        input_root=input_root,
        output_root=output_root,
        target_size=target_size,
        num_workers=4  # Adjust based on your CPU
    )