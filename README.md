Sport Classifier

To setup this codebase, following are the steps:

1. Download and extract the Kinectics-400 dataset. Refer to this link: https://github.com/cvdfoundation/kinetics-dataset

2. Clone this repository by running the following command:
```
git clone https://github.com/shubham96jain/sport_classification.git
```

3. Install the dependencies by running the following command:
```
pip install -r requirements.txt
```

4. Setup the paths for the files in ```scripts/preprocess_annotations_csv.py``` and then run the following command to preprocess the annotations csv file:
```
python scripts/preprocess_annotations_csv.py
```

This creates a pickle file ```sports_videos.pkl``` in the root directory and print dataset statistics.

5. Setup the paths for the files in ```scripts/organize_dataset.py``` and then run the following command to organize the videos into the directory structure:
```
python scripts/organize_dataset.py
```
This will create the train, val and test folders with the videos organized in them based on the sport category.

6. Optionally, you can resize the videos to 224x224 by running the following command:
```
python scripts/resize_videos.py
```

Now the dataset is ready to be used for training.

7. Setup the config.py file in ```configs/``` and then run the following command to start training:
```
python train.py
```
This will start the training process. This will also create a cache file for dataset to speed up the consecutive runs with same dataset parameters. The training code also generates TensorBoard logs in ```runs/``` directory which the code creates automatically.

