import pickle
import os
import shutil
from pathlib import Path

SOURCE_BASE = "data/kinetics"  # Adjust this to your source videos path
TARGET_BASE = "data/kinetics_processed"
VIDEOS_PKL_PATH = "sports_videos.pkl"
CLASS_LIST = ['baseball', 'basketball', 'soccer', 'cricket']
SPLITS = ['train', 'val', 'test']

def create_directory_structure(base_path):
    """Create the directory structure if it doesn't exist."""
    for split in SPLITS:
        for sport in CLASS_LIST:
            path = os.path.join(base_path, split, sport)
            Path(path).mkdir(parents=True, exist_ok=True)

def format_video_filename(youtube_id, start_time, end_time):
    """Convert youtube_id and timestamps to the video filename format"""
    return f"{youtube_id}_{start_time:06d}_{end_time:06d}.mp4"

def organize_videos():
    
    # Load the pickle file
    with open(VIDEOS_PKL_PATH, 'rb') as f:
        video_data = pickle.load(f)
    
    # Create directory structure
    create_directory_structure(TARGET_BASE)
    
    # Statistics counters
    stats = {split: {sport: 0 for sport in CLASS_LIST} 
            for split in SPLITS}
    errors = []
    
    # Process each split
    for split in video_data:
        print(f"\nProcessing {split} split...")
        
        # Process each sport category
        for sport in video_data[split]:
            print(f"Processing {sport}...")
            
            # Process each video in the category
            for video_id in video_data[split][sport]:
                video_file = format_video_filename(video_id, video_data[split][sport][video_id]['time_start'],  video_data[split][sport][video_id]['time_end'])
                source_file = os.path.join(SOURCE_BASE, split, video_file)
                target_file = os.path.join(TARGET_BASE, split, sport, video_file)
                
                try:
                    if os.path.exists(source_file):
                        # Copy the file if it doesn't exist in target
                        if not os.path.exists(target_file):
                            shutil.copy2(source_file, target_file)
                            stats[split][sport] += 1
                    else:
                        errors.append(f"File not found: {source_file}")
                except Exception as e:
                    errors.append(f"Error processing {video_id}: {str(e)}")
    
    # Print statistics
    print("\nCopy Statistics:")
    total_copied = 0
    for split in stats:
        split_total = sum(stats[split].values())
        total_copied += split_total
        print(f"\n{split.upper()}:")
        for sport, count in stats[split].items():
            print(f"{sport}: {count} videos")
        print(f"Total {split}: {split_total} videos")
    
    print(f"\nTotal videos copied: {total_copied}")
    
    # Print errors if any
    if errors:
        print("\nErrors encountered:")
        for error in errors[:10]:  # Print first 10 errors
            print(error)
        if len(errors) > 10:
            print(f"...and {len(errors) - 10} more errors")

if __name__ == "__main__":
    # Ask for confirmation before proceeding
    print("This script will organize videos into sport categories.")
    print("Make sure you have enough disk space for the copied files.")
    response = input("Do you want to proceed? (yes/no): ")
    
    if response.lower() == 'yes':
        organize_videos()
    else:
        print("Operation cancelled.")