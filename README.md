# Deep-learning-Grade-Detection-Project
Project by Onkar Trivedi, Aishwary Jagetia &amp; Animesh Nema 

Dataset collected and made available by MIT AgeLab. Acquired from the following link: http://lexfridman.com/carsync/

The CNN can be run by the following steps:

Parsing the main datset: Main_Dataset_Parsing.py (converts the main video_front file into individual frames)

Generating a corresponding segmented dataset: Road_segmentation.py

Running the final CNN model: DLmodel_final_CNN.py

Visualizing the CNN model: CNNmodel_Visualization.py

Generating a video to visualize the model results: Slope_Visualization_video.py


[NOTE: The model.h5 & model.json files are provided, generated from the final CNN model. road_dataset_newest.csv & roadseg_dataset_newest.csv contain the respective csv files with the required IMU pitch & GPS altitude data.]
