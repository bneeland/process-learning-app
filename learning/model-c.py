# Import libraries
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
import pandas as pd
import math
from sklearn.model_selection import train_test_split
import csv

# Import labelled dataset (output from 'mining' phase)
LabelledDataset_Original = pd.read_csv("../mining/output_sequence/outputsequence-c.csv")
LabelledDataset_Sorted = LabelledDataset_Original.sort_values(["CaseID", "Timestamp"])
NumEvents = len(LabelledDataset_Original)
# Print number of events
print("Number of events: ", NumEvents)

# Get unique CaseIDs into a list
CaseIDs = LabelledDataset_Sorted["CaseID"].unique().tolist()
NumCases = len(CaseIDs)
# Print number of cases
print("Number of cases:", NumCases)

# Get unique labels into a list
LabelIDs = LabelledDataset_Sorted["LabelID"].unique().tolist()
LabelIDs.sort()
NumLabelIDs = len(LabelIDs)
# Assign an index for each EventID
LabelIndex = list(range(0, NumLabelIDs))
LabelIDs = dict(zip(LabelIDs, LabelIndex))
# Save label IDs in csv file (labels can be edited there)
try:
    w = csv.writer(open("saved_data/LabelIDs-c.csv", "w"))
    for key, val in LabelIDs.items():
        w.writerow([key, val])
    print("Label IDs written to file.")
except:
    print("ERROR: Unable to write label IDs to file.")

# Get unique events into a list
EventIDs = LabelledDataset_Sorted["EventID"].unique().tolist()
EventIDs.sort()
NumEventsIDs = len(EventIDs)
# Assign an index for each EventID
EventIndex = list(range(0, NumEventsIDs))
EventIDs = dict(zip(EventIDs, EventIndex))
# Save event IDs in csv file
try:
    w = csv.writer(open("saved_data/EventIDs-c.csv", "w"))
    for key, val in EventIDs.items():
        w.writerow([key, val])
    print("Event IDs written to file.")
except:
    print("ERROR: Unable to write event IDs to file.")

### Build feature sets and labels

## Initialize feature sets with zeros
# Feature set length along axis 0 (height) is the number of unique event ids
X_Height = NumEventsIDs
# Feature set length along axis 1 (width) is the max time period among all cases
X_Width = 0
for CaseID in CaseIDs:
    Case = LabelledDataset_Sorted.loc[LabelledDataset_Sorted.CaseID == CaseID]
    X_HeightCurrent = Case.shape[0]
    if X_HeightCurrent > X_Height:
        X_Height = X_HeightCurrent
    X_WidthCurrent = Case["Timestamp"].max() - Case["Timestamp"].min()
    if X_WidthCurrent > X_Width:
        X_Width = X_WidthCurrent
X_Width = math.ceil(X_Width) + 1
X = np.zeros((NumCases, X_Height, X_Width))
## Populate feature sets
# Look for the place in each CaseID's feature set in X that is at location_
# EventID along axis 0 and 'relative' timestamp along axis 1
i = 0
for CaseID in CaseIDs:
    Case_df = LabelledDataset_Sorted.loc[LabelledDataset_Sorted.CaseID == CaseID]
    Case_np = Case_df.values
    Time_0 = Case_df["Timestamp"].min()
    for Event in Case_np:
        X[i][EventIDs[Event[0]]][int(round(Event[1] - Time_0))] = 1
    i += 1
print("Feature sets (X) created.")

# Initialize labels with zeros
y = np.zeros((NumCases))
# Populate labels
i = 0
for CaseID in CaseIDs:
    Case_df = LabelledDataset_Sorted.loc[LabelledDataset_Sorted.CaseID == CaseID]
    Case_np = Case_df.values
    Label = Case_np[0][3]
    LabelID = LabelIDs[Label]
    y[i] = LabelID
    i += 1
print("Labels (y) created.")

# Record X_Height and X_Width to csv file to build proper test array in cnn_predict.py
try:
    w = csv.writer(open("saved_data/x_height-x_width-c.csv", "w"))
    w.writerow(["X_Height", X_Height])
    w.writerow(["X_Width", X_Width])
    print("X_Height and X_Width written to file.")
except:
    print("ERROR: Unable to write label IDs to file.")

## Preprocess data

# Create train and test sets
X_Train_o, X_Test_o, y_Train_o, y_Test_o = train_test_split(X, y, test_size = 0.25)

# Reshape feature set data to specify the depth
X_Train = X_Train_o.reshape(X_Train_o.shape[0], X_Height, X_Width, 1)
X_Test = X_Test_o.reshape(X_Test_o.shape[0], X_Height, X_Width, 1)

# Convert data type
X_Train = X_Train.astype("float32")
X_Test = X_Test.astype("float32")

# Convert 1-dimensional label array to NumLabelIDs-dimensional label matrices
y_Train = np_utils.to_categorical(y_Train_o, NumLabelIDs)
y_Test = np_utils.to_categorical(y_Test_o, NumLabelIDs)

# Build the CNN model
Model = Sequential()
Model.add(Conv2D(32, kernel_size = (1, int(X_Train.shape[2]/2)), strides = (1, 1), input_shape = (X_Train.shape[1], X_Train.shape[2], 1), activation="relu"))
Model.add(MaxPooling2D(pool_size=(1, int(X_Train.shape[2]/5 + 1)), strides=(1, int(X_Train.shape[2]/5 + 1))))
Model.add(Flatten())
Model.add(Dense(128, activation = "sigmoid"))
Model.add(Dense(NumLabelIDs, activation = "softmax"))

Model.summary()

# Compile model
Model.compile(loss="categorical_crossentropy", optimizer="adam", metrics = ["accuracy"])

# Fit the model
Model.fit(X_Train, y_Train, validation_data = (X_Test, y_Test), epochs = 500, batch_size=10, callbacks = [EarlyStopping(monitor = "val_accuracy", min_delta = 0.00001, patience = 20, verbose = 1, mode = "auto")])

# Save the model
try:
    Model.save("saved_models/PLA_Learning_CNN-Model-c.h5")
    print("Model saved.")
except:
    print("ERROR: Unable to save model.")
