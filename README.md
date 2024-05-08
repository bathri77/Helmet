# Real-time-Detection-System-of-Bike-Riders-without-Helmet
INTRODUCTION TO THE PROJECT-

This project detects whether a person is wearing helmet or not in real time. This project involves training of custom yolo model to detect two classes, namely - helmet and no_helemt
The camera starts and the real time pictures are clicked and sent into the trained model to detect whether a person is wearing helmet or not . If it identifies that the person is not
wearing helmet then an alarm is raised as a warning to that person. 

ABOUT THE DATASET- 

A dataset containing pictures of people wearing helmet and people not wearing helmet is used. The dataset is labelled using open source labelling tool. The labels conatins the 
coordinates of the detected object. 

PREREQUISITE INSTALLATION- 

open-cv, numpy, pygame

STEPS OF THE PROJECT-

1. DATA COLLECTION- Pictures of people wearing helmet and not wearing helmet are collected by clicking images from high resolution camera and from internet.

2. DATA LABELLING- The dataset collected is labelled using open source labelling tool to generate a text file containing coordinates of the object to be detected along with their
classes. Both the label files and images are transferred to folder named obj and zipped it.

3. GENERATING PATHS OF THE IMAGES FILES- Made a test and train text files containing paths of images to be trained.

4. PREPARATION OF YOLO CFG FILE- Downloaded yolo-v4-tiny.cfg and edited the file by changing number of classes and number of filters (formula used- (classes_num + 5)*3). 

5. PREPARATION OF NAMES AND DATA FILE- Made obj.names file containing names of the classes to be detected namely- helmet and no_helmet. Made obj.data file in which threre are 
informations like number of classes, paths of backup folder, test.txt, train.txt and obj.names file.

6. PREPARATION OF FINAL FOLDER CONTAINING FILES NEEDDED FOR TRAINING- .cfg file, .names file, .data file, obj.zip, test.txt and train.txt are transferred to a folder and uploaded 
on drive. 

7. TRAINED YOLO CUSTOM MODEL ON GOOGLE COLAB- Trained yolo custom model by cloning https://github.com/AlexeyAB/darknet and using the saved folder in step6. The code of training can
be found in training.pynb file.

8. DOWNLOADED WEIGHTS FILE- Downloaded weights file of trained model to further use it for inference.

9. WRITING FINAL CODE FOR RUNNING INFERENCE- Written final code to run inferece on real time pictures using .cfg , .names and .weights file. The code is in helmet3.py. An alarm
file is added. The code when runs clicks pictures of person continuously and send them to trained model to detect the helmet or no_helmet class. If no helmet is found then
a warning alarm is raised andit raises continuously till no_helmet detected.

STEPS TO RUN INFERENCE-

1. Install the prerequisites mentioned above.
2. Change the paths of yolov4-tiny-obj.cfg, yolov4-tiny-obj_last.weights, obj.names and alarm.wav file in helmet3.py.
3. Open command prompt at this path and write- python helmet3.py
4. The webcam will start and it will start detecting continously unless interrupted using enter key. 

FUTURE SCOPE OF THE PROJECT-

At present this project can be used for bike riders implemented using appropriate hardware. The model is a bit slower on real time if used in mobile devices like raspberry pi.
This can be made faster by converting into other faster formats like open vino format(.xml and .bin) or tensorflowlite. But converting this into these forms may lead to less 
accuracy. So there is always a speed-accuracy trade-off.
