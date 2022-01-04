# Import libraries
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import load_model
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import math
from sklearn.model_selection import train_test_split
import time
import csv

# Import proprocessed feature sets from files
print("Importing test feature sets...")
Time = time.time()

X_Test_a = np.load("saved_data/X_Test_a.npy")
X_Test_a_r = np.load("saved_data/X_Test_a_r.npy")
X_Test_c = np.load("saved_data/X_Test_c.npy")
X_Test_c_r = np.load("saved_data/X_Test_c_r.npy")

Time = round(time.time() - Time, 1)
print("Done (Time taken: ", Time, "sec)")

# Load label IDs from csv file for -a
with open("saved_data/LabelIDs-a.csv", mode="r") as file:
    r = csv.reader(file)
    LabelIDs_a = {rows[1]:rows[0] for rows in r}

# Load event IDs from csv file for _a
with open("saved_data/EventIDs-a.csv", mode="r") as file:
    r = csv.reader(file)
    EventIDs_a = {rows[1]:rows[0] for rows in r}

# Load label IDs from csv file for -c
with open("saved_data/LabelIDs-c.csv", mode="r") as file:
    r = csv.reader(file)
    LabelIDs_c = {rows[1]:rows[0] for rows in r}

# Load event IDs from csv file for -c
with open("saved_data/EventIDs-c.csv", mode="r") as file:
    r = csv.reader(file)
    EventIDs_c = {rows[1]:rows[0] for rows in r}

# Retrieve the models
Model_a = load_model("saved_models/PLA_Learning_CNN-Model-a.h5")
Model_c = load_model("saved_models/PLA_Learning_CNN-Model-c.h5")

## Predict feature sets from the test data, print activity names, and show plot

print("======================================================================================")
print("Event log")
print("======================================================================================")
print("Activity name                                           Activity type        Timestamp")
print("——————————————————————————————————————————————————————————————————————————————————————")

# Initialize the figure for the plot
Figure, (Subplot_a, Subplot_c) = plt.subplots(2, 1)

# Create subplots for each -a and -c
Subplot_a.set_title("#Placeholder\nfor\ntight_layout", loc = "left")
Frame_a = Subplot_a.imshow(X_Test_a[1], animated = True, aspect = "auto")
Subplot_c.set_title("\n#Placeholder\nfor\ntight_layout", loc = "left")
Frame_c = Subplot_c.imshow(X_Test_c[40], animated = True, aspect = "auto")

Figure.tight_layout()

# Set the time interval at which the program goes through the event log
Interval = 0.25 * 1000 # 1,000 milliseconds in a second

# Global variable for stepping through periods
i = 0

def UpdateFigure_a(FrameNumber):
    global i
    # Predict the label based on the feature set
    y_Prediction = np.argmax(Model_a.predict(np.array([X_Test_a_r[i]])), axis=-1)
    y_Prediction = str(y_Prediction[0]) # Convert array to string value
    LabelID_a = LabelIDs_a[y_Prediction]
    y_PredictionProb = Model_a.predict(np.array([X_Test_a_r[i]]))
    # Determine if there is a new activity in the 'window'
    FirstPeriod = [el[0] for el in X_Test_a[i]]
    if any(FirstPeriod):
        # Print the activity name in terminal
        EventID_a = np.argmax(FirstPeriod, axis = None, out = None)
        print("%-55s" %(EventIDs_a[str(EventID_a)]), "%-20s" %("Alarm"), "%-10s" %(i))
    # Print the prediction, updating the probability
    if np.amax(y_PredictionProb) > 0.80:
        Subplot_a.set_title("ALARMS\n" + str(LabelID_a) + "\n" + str(int(np.amax(y_PredictionProb)*100)) + "%", loc = "left")
    else:
        Subplot_a.set_title("ALARMS\n\n", loc = "left")
    # Set the frame to the current feature set
    Frame_a.set_array(X_Test_a[i])
    # Return the frame to the animation function
    return Frame_a,

def UpdateFigure_c(FrameNumber):
    global i
    # Predict the label based on the feature set
    y_Prediction = np.argmax(Model_c.predict(np.array([X_Test_c_r[i]])), axis=-1)
    y_Prediction = str(y_Prediction[0]) # Convert array to string value
    LabelID_c = LabelIDs_c[y_Prediction]
    y_PredictionProb = Model_c.predict(np.array([X_Test_c_r[i]]))
    # Determine if there is a new activity in the 'window'
    FirstPeriod = [el[0] for el in X_Test_c[i]]
    if any(FirstPeriod):
        # Print the activity name in terminal
        EventID_c = np.argmax(FirstPeriod, axis = None, out = None)
        print("%-55s" %(EventIDs_c[str(EventID_c)]), "%-20s" %("Command"), "%-10s" %(i))
    # Print the prediction, updating the probability
    if np.amax(y_PredictionProb) > 0.80:
        Subplot_c.set_title("\nCOMMANDS\n" + str(LabelID_c) + "\n" + str(int(np.amax(y_PredictionProb)*100)) + "%", loc = "left")
    else:
        Subplot_c.set_title("\nCOMMANDS\n\n", loc = "left")
    # Set the frame to the current feature set
    Frame_c.set_array(X_Test_c[i])
    # Return the frame to the animation function
    return Frame_c,

def UpdateFigures(FrameNumber):
    global i
    a = UpdateFigure_a(FrameNumber)
    c = UpdateFigure_c(FrameNumber)
    i += 1
    print("Time:", i, end = "\r")
    return a + c

Animation = animation.FuncAnimation(Figure, UpdateFigures, interval = Interval)

plt.show()
