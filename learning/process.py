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

# Import labelled dataset (output from 'mining' phase)
LabelledDataset_Original_a = pd.read_csv("../mining/output_sequence/outputsequence-a.csv")
LabelledDataset_Sorted_a = LabelledDataset_Original_a.sort_values(["CaseID", "Timestamp"])
LabelledDataset_Original_c = pd.read_csv("../mining/output_sequence/outputsequence-c.csv")
LabelledDataset_Sorted_c = LabelledDataset_Original_c.sort_values(["CaseID", "Timestamp"])

# Load event IDs from csv file for -a
with open("saved_data/EventIDs-a.csv", mode="r") as file:
    r = csv.reader(file)
    EventIDs_a = {rows[0]:rows[1] for rows in r}
print("EventIDs_a:\n", EventIDs_a)

# Load event IDs from csv file for -c
with open("saved_data/EventIDs-c.csv", mode="r") as file:
    r = csv.reader(file)
    EventIDs_c = {rows[0]:rows[1] for rows in r}
print("EventIDs_c:\n", EventIDs_c)

# Get X-Height and X_Width for -a and -c, to be used for "exploded" test array
with open("saved_data/x_height-x_width-a.csv", mode="r") as file:
    r = csv.reader(file)
    X_Dims_a = {rows[0]:rows[1] for rows in r}
X_Height_a = int(X_Dims_a["X_Height"])
X_Width_a = int(X_Dims_a["X_Width"])
with open("saved_data/x_height-x_width-c.csv", mode="r") as file:
    r = csv.reader(file)
    X_Dims_c = {rows[0]:rows[1] for rows in r}
X_Height_c = int(X_Dims_c["X_Height"])
X_Width_c = int(X_Dims_c["X_Width"])
# # Set the width of the "exploded" test array
X_Length = max(max(LabelledDataset_Sorted_a["Timestamp"]), max(LabelledDataset_Sorted_c["Timestamp"])) + 1

## Build X_Test_a and X_Test_c to reveal activities at their correct timestamp

# Initialize all zero array of size X_Height_a/_c and X_Width
# '_e' means 'exploded', '_z' means a single zero array
X_Test_a = np.zeros((X_Length, X_Height_a, X_Width_a))
X_Test_c = np.zeros((X_Length, X_Height_c, X_Width_c))

# Loop through each period of each feature set in X_Test_o and build each new period in X_Test_o_e
print("Building exploded feature sets...")
Time = time.time()

## Introduce events into the test array
# Go through each period i where there was an event in the event log
for i in LabelledDataset_Sorted_a["Timestamp"]:
    # Find the corresponding event ID
    for Event in LabelledDataset_Sorted_a.loc[LabelledDataset_Sorted_a["Timestamp"] == i]["EventID"].values:
        # Add a 1 in the first column for that period and for that event ID
        X_Test_a[i][int(EventIDs_a[Event])][0] = 1
# Go through each period i where there was an event in the event log
for i in LabelledDataset_Sorted_c["Timestamp"]:
    # Find the corresponding event ID
    for Event in LabelledDataset_Sorted_c.loc[LabelledDataset_Sorted_c["Timestamp"] == i]["EventID"].values:
        # Add a 1 in the first column for that period and for that event ID
        X_Test_c[i][int(EventIDs_c[Event])][0] = 1

## Migrate events through the test array
# Go through each timestamp i
for i in range(X_Length - 1):
    # Go through each event type j
    for j in range(X_Height_a):
        # Copy each "window" position k in j for i to k+1 in j for i+1
        for k in range(X_Width_a - 1):
            if X_Test_a[i][j][k] == 1:
                X_Test_a[i + 1][j][k + 1] = 1
    # Go through each event type j
    for j in range(X_Height_c):
        # Copy each "window" position k in j for i to k+1 in j for i+1
        for k in range(X_Width_c - 1):
            if X_Test_c[i][j][k] == 1:
                X_Test_c[i + 1][j][k + 1] = 1
    print(i, "/", X_Length - 2, end = "\r")

Time = round(time.time() - Time, 1)
print("\nDone (Time taken: ", Time, "sec)")

# Reshape feature set data to specify the depth
# "_r" means "reshaped"
X_Test_a_r = X_Test_a.reshape(X_Length, X_Height_a, X_Width_a, 1)
X_Test_c_r = X_Test_c.reshape(X_Length, X_Height_c, X_Width_c, 1)

# Save preprocessed dataset (X_Train, X_Test, y_Train, and y_Test) in npy file
try:
    np.save("saved_data/X_Test_a.npy", X_Test_a)
    np.save("saved_data/X_Test_a_r.npy", X_Test_a_r)
    np.save("saved_data/X_Test_c.npy", X_Test_c)
    np.save("saved_data/X_Test_c_r.npy", X_Test_c_r)
    print("Preprocessed feature sets written to files.")
except:
    print("ERROR: Unable to write preprocessed feature sets to files.")
