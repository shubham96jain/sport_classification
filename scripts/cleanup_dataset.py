import os
import csv

# List of allowed classes
ALLOWED_CLASSES = [
    "catching or throwing baseball",
    "dribbling basketball",
    "dunking basketball",
    "hitting baseball",
    "juggling soccer ball",
    "kicking soccer ball",
    "playing basketball",
    "playing cricket",
    "shooting basketball",
    "shooting goal (soccer)"
]

def format_video_filename(youtube_id, start_time, end_time):
    """Convert youtube_id and timestamps to the video filename format"""
    return f"{youtube_id}_{start_time:06d}_{end_time:06d}.mp4"

def clean_validation_folder(csv_path, val_folder):
    # Create a set to store valid video IDs
    valid_video_ids = set()
    
    # Read the CSV file and collect valid video IDs
    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        header = next(csvreader)  # Skip header
        for row in csvreader:
            if row:  # Check if row is not empty
                action_class = row[0].strip('"')
                # import pdb; pdb.set_trace()
                time_start = int(row[2])
                time_end = int(row[3])
                if action_class in ALLOWED_CLASSES:
                    video_id = row[1]  # Get video ID from second column
                    video_filename = format_video_filename(video_id, time_start, time_end)
                    valid_video_ids.add(video_filename)
                # else:
                #     print(action_class)
    print(valid_video_ids)
    # Count statistics
    total_files = 0
    deleted_files = 0
    
    # Iterate through files in validation folder and delete unwanted ones
    for filename in os.listdir(val_folder):
        total_files += 1
        video_id = filename.split('.')[0]  # Remove file extension to get video ID
        # import pdb; pdb.set_trace()
        if filename not in valid_video_ids:
            file_path = os.path.join(val_folder, filename)
            try:
                os.remove(file_path)
                deleted_files += 1
                print(f"Deleted: {filename}")
            except OSError as e:
                print(f"Error deleting {filename}: {e}")
    
    # Print summary
    print("\nCleaning complete!")
    print(f"Total files processed: {total_files}")
    print(f"Files deleted: {deleted_files}")
    print(f"Files remaining: {total_files - deleted_files}")

if __name__ == "__main__":

    # Paths to your CSV file and validation folder
    val_csv_path = "data/kinetics-dataset/k400/annotations/val.csv"
    val_folder = "data/kinetics-dataset/k400/val"  # Adjust this path as needed

    train_csv_path = "data/kinetics-dataset/k400/annotations/train.csv"
    train_folder = "data/kinetics-dataset/k400/train"

    test_csv_path = "data/kinetics-dataset/k400/annotations/test.csv"
    test_folder = "data/kinetics-dataset/k400/test"
    
    # Confirm before proceeding
    print("This will delete files that don't belong to the specified sports classes.")
    response = input("Do you want to proceed? (yes/no): ")
    
    if response.lower() == 'yes':
        clean_validation_folder(val_csv_path, val_folder)
        clean_validation_folder(train_csv_path, train_folder)
        clean_validation_folder(test_csv_path, test_folder)
    else:
        print("Operation cancelled.")