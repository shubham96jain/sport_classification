import csv
import pickle
from collections import defaultdict
import os

from class_mapping import CLASSES

ANNO_PATH = 'data/kinetics/annotations/'
SELECTED_LABELS_PATH = 'kinetics_selected_labels.txt'
VIDEOS_PKL_PATH = 'sports_videos.pkl'

# Create reverse mapping from specific action to main sport
ACTION_TO_SPORT = {}
for sport, actions in CLASSES.items():
    for action in actions:
        ACTION_TO_SPORT[action] = sport

# Read selected labels
with open(SELECTED_LABELS_PATH, 'r') as f:
    selected_labels = [line.strip() for line in f if line.strip()]

# Create a nested dictionary structure
video_data = defaultdict(dict)

files = ['val.csv', 'train.csv', 'test.csv']
# Read the CSV file
for filename in files:
    split = filename.split('.')[0]
    split_data = {}
    with open(os.path.join(ANNO_PATH, filename), 'r') as f:
        csv_reader = csv.reader(f)
        header = next(csv_reader)  # Skip header
        
        for row in csv_reader:
            if not row:  # Skip empty rows
                continue
            
            label = row[0].strip('"')  # Remove quotes if present
        
            # Check if this label is one we want to keep
            if label in selected_labels:
                youtube_id = row[1]
                time_start = int(row[2])
                time_end = int(row[3])

                # Get the main sport category for this action
                sport_category = ACTION_TO_SPORT[label]
                
                if sport_category not in split_data:
                    split_data[sport_category] = {}

                if youtube_id not in split_data[sport_category]:
                    split_data[sport_category][youtube_id] = {
                        'time_start': time_start,
                        'time_end': time_end,
                        'original_action': label  # Optionally store the original action
                    }
    video_data[split] = split_data

# Convert defaultdict to regular dict
video_data = dict(video_data)

# Save to pickle file
with open(VIDEOS_PKL_PATH, 'wb') as f:
    pickle.dump(video_data, f)

# Print some statistics
print("\nDataset statistics:")
for split in video_data:
    print(f"{split}:")
    for label in video_data[split]:
        print(f"{label}: {len(video_data[split][label])} videos")