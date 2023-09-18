## with inspiration from: 
        # Author: Evan Juras
        # Date: 10/27/19
        # Description: 
        # This program uses a TensorFlow Lite model to perform object detection on a live webcam
        # feed. It draws boxes and scores around the objects of interest in each frame from the
        # webcam. To improve FPS, the webcam object runs in a separate thread from the main program.
        # This script will work with either a Picamera or regular USB webcam.
        #
        # This code is based off the TensorFlow Lite image classification example at:
        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
        #
        # I added my own method of drawing boxes and labels using OpenCV.
        # 
        # Modified by: Shawn Hymel
        # Date: 09/22/20
        # Description:
        # Added ability to resize cv2 window and added center dot coordinates of each detected object.
        # Objects and center coordinates are printed to console.
## Version 2.0
## Author: Noah Vandal
## Date: 9/19/2023
## Use case: more simpler, more functional version 

import os
import cv2
from tflite_runtime.interpreter import Interpreter
from datetime import datetime
import numpy as np
from firebase_admin import credentials, firestore, initialize_app


# for image augmentation
global isFloatingModel
global input_mean
global input_std
input_mean = 127.5
input_std = 127.5


class LoadModel():
    def __init__(self, 
        modelName='ModelDir',
        weights='detect.tflite',
        labels='labelmap.txt',
        imW = 640,
        imH = 480,
        ):
        self.modelName = modelName
        self.weightPath = weights
        self.labelPath = labels
        self.imgWidth = imW
        self.imgHeight = imH

        ## run load
        self.load()

    def load(self):
        global isFloatingModel
        # Get path to current working directory
        CWD_PATH = os.getcwd()

        # Path to .tflite file, which contains the model that is used for object detection
        PATH_TO_CKPT = os.path.join(CWD_PATH,self.modelName,self.weightPath)

        # Path to label map file
        PATH_TO_LABELS = os.path.join(CWD_PATH,self.modelName,self.labelPath)

        # Load the label map
        with open(PATH_TO_LABELS, 'r') as f:
            labels = [line.strip() for line in f.readlines()]

        ## fix for labels from coco training
        if labels[0] == '???':
            del(labels[0])

        interpreter = Interpreter(model_path=PATH_TO_CKPT)

        interpreter.allocate_tensors()

        # Get model details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        height = input_details[0]['shape'][1]
        width = input_details[0]['shape'][2]

        isFloatingModel = (input_details[0]['dtype'] == np.float32)

        return interpreter, input_details, output_details, labels, height, width


def name_to_color(name):
    # Generate a color based on the hash of the class name
    # Taking the hash modulo 256 to fit it into an 8-bit color channel
    color = tuple([(hash(name) >> i) & 0xFF for i in (0, 8, 16)])
    return color

def plot_paths(frame, class_paths):
    '''
    Plot tracks of particular class in a frame
    '''
    for class_name, points in class_paths.items():
        color = name_to_color(class_name)
        for i in range(1, len(points)):
            cv2.line(frame, points[i - 1], points[i], color, 3)
    return frame

class SendToFirestoreDB():
    def __init__(self):
        cred = credentials.Certificate("rpicvod-firebase-adminsdk-x27x0-02a6d3581d.json")
        initialize_app(cred)

        self.loadDb()

    def loadDb(self):
        db = firestore.client()

        return db


def processImage(frame,interpreter, output_details,labels, imH, imW):
    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
    #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    objects= {}
    for i in range(len(scores)):
        if ((scores[i] > 0.5) and (scores[i] <= 1.0)):

            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
            
            # Draw label
            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

            # Draw circle in center
            xcenter = xmin + (int(round((xmax - xmin) / 2)))
            ycenter = ymin + (int(round((ymax - ymin) / 2)))
            cv2.circle(frame, (xcenter, ycenter), 5, (0,0,255), thickness=-1)

            objectDetected = {
                'name': labels[int(classes[i])],
                'location': (xcenter,ycenter),
                'upperLeft': (xmin, ymin),
                'time': datetime.now().isoformat()
            }
            objects[f'object_{i}'] = objectDetected # dictionary of dictionaries
    
    return frame, objects


def augmentImageBeforeInference(frame, height, width):
    global isFloatingModel
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    # frame_resized = resize_aspect_fit(frame_rgb, (width, height))

    input_data = np.expand_dims(frame_resized, axis=0)
    # print(frame_rgb.shape, frame_resized.shape)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if isFloatingModel:
        input_data = (np.float32(input_data) - input_mean) / input_std
    
    return input_data

def main():
    ## initialize model
    model = LoadModel()
    interpreter, input_details, output_details, labels, height, width = model.load()

    ## initialize database origin
    database = SendToFirestoreDB()
    db = database.loadDb()

    # Initialize frame rate calculation
    frame_rate_calc = 1
    freq = cv2.getTickFrequency()

    # initialize frame source
    src = cv2.VideoCapture(0)

    # for storing tracking data
    classPaths = {}

    # size of cam dims
    imH, imW = 480, 640

    startTime = str(datetime.now())
    frameNumber = 0
    while True:
        print('Frame', frameNumber)
        t1 = cv2.getTickCount()
        ret, frame = src.read() ## grab frame
        inputData = augmentImageBeforeInference(frame, height, width)

        ## inference frame
        interpreter.set_tensor(input_details[0]['index'],inputData)
        interpreter.invoke()

        ## put data on image
        outputFrame, frameData = processImage(frame, interpreter, output_details, labels, imH, imW)

        ## push data to db; new document for each frame
        db_push = db.collection(startTime).document(f"Frame_{frameNumber}")
        db_push.set(frameData)

        # Draw framerate in corner of frame
        cv2.putText(outputFrame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
        outputFrame = plot_paths(outputFrame, classPaths)

        ## display image
        cv2.imshow('Object detector', outputFrame)

        # Calculate framerate
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc= 1/time1

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

        # update framecount
        frameNumber += 1
    
    ## when terminated, release from memory
    src.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

