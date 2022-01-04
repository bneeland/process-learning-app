# Process Learning App
This is a convolutional neural network using Keras that classifies sets of events separated by time. These events are, in this case, industrial control system alarms and operator commands. The use case is for industrial operators to be able to use this program as a tool to automatically label sets of alarms with the root cause of the alarm set. In addition, the subsequent operator commands can be labelled automatically by the program to identify the root cause to which these commands conform, as a form of check on operator actions.

## How to use this program
Run `learning/process.py` to convert the raw input data found in `mining/output_sequence/` CSV files into numpy arrays (NPY files), which are saved in `learning/saved_data/`. The raw input data is fictional, but is based on real alarms and operator commands in an industrial plant.

Run `learning/model-a.py` to train the convolutional neural network on the alarm data, and  `learning/model-c.py` to train another CNN on the operator command data. The weights for each of these two CNNs is saved in `learning/saved_models/`.

Run `learning/predict.py` to use the saved trained CNN models on the data to label events (alarms and commands) that arise over time. The labels indicated the root cause of the events. A Matplotlib animation shows the events occurring over time, and the labels and probabilities predicted by the trained CNN.
