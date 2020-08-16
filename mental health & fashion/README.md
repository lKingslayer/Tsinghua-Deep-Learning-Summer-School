# dlss_exp
Experiments for deep learning summer school in Tsinghua University.

You should try to complete the tasks of two projects list below as many as possible.

## Project 1: Depression Detection

### Background
Depression is a major contributor to the overall global burden of diseases. Traditionally, doctors diagnose depressed people face to face via referring to clinical depression criteria. However, more than 70% of the patients would not consult doctors at early stages of depression, which leads to further deterioration of their conditions.

Meanwhile, people are increasingly relying on social media to disclose emotions and sharing their daily lives, thus social media have successfully been leveraged for helping detect physical and mental diseases.

### Target 
Make timely depression detection via harvesting social media data.

That is, classify the users into depressed group and non-depressed group.

### Data & Feature
As we have limited time, we have already extracted some features from the raw data crawled from Twitter and construct dataset (feature set) . The feature table can be shown in `data/depression/feature_table.xlsx`.

### Experiment
This experiment based on a high-level tensorflow API `tf.contrib`. This API is powerful and simple to use.
 
 [See API in Github](https://github.com/tensorflow/tensorflow/tree/r1.0/tensorflow/contrib).

The experiment contains following steps:

1. Split the data in depression_detection/data.csv into training set data file and testing set data file with a ratio about 3:2

2. Load the data from file, use functions in `tf.contrib.learn.datasets.base`

3. Normalize the data. We must normalize all the data at the same time, so we should combine the training set and testing set firstly, and split them apart after normalization.

4. Build linear classifier and dnn classifier with `tf.contrib.learn`. You should choose arguments yourself.

5. Fit and evaluate the model. You should compute accuracy, precision, recall and F1-measure in macro-average.

### Grading (Total: 50 pts)

1. Split data (5 pts)

2. Load data (5 pts)

3. Normalize data (15 pts)

4. Build linear classifier with `tf.contrib.learn` (5 pts)

5. Build DNN classifier with `tf.contrib.learn` (5 pts)

6. Fit model. (5 pts)

7. Make evaluation (10 pts)



## Project 2: Fashion Style

### Background
People often use aesthetic words to describe the clothing fashion styles, which rely heavily on some visual details such as collar shape, sleeve length, pattern design, etc.

Different collocations of top and bottom clothing can lead to different aesthetic styles.

Can we bridge the gap between visual details and fashion styles of clothing collocations automatically? Can we teach computers to appreciate clothing like human beings?

### Target 
Build a model to appreciate the fashion styles of clothing collocations automatically.

That is, map the visual features of clothing to fashion style words.

### Data & Feature
We have already randomly split the raw feature data into training data and test data. The feature table is also provided.

### Experiment
The experiment contains following steps:

1. Use original features to train a 5-layer autoencoder

2. Calculate middle layer representation of original features

3. Use new representation of features to train DNN regressor

### Grading (Total: 50 pts)

1. Create weights and biases of encoder's second layer (5 pts)

2. Create weights and biases of encoder's \*second\* layer (5 pts)

3. Write the computation of encoder's second layer and decoder's \*second\* layer. (5 pts)

4. Define the loss function (5 pts)

5. Define a gradient descent optimizer (5 pts)

6. Change train_features and test_features from numpy arrays to tensorflow constants (5 pts)

7. Calculate the middle layer representation of train/test features (5 pts)

8. Define DNN regressor (5 pts)

9. Train the regressor (5 pts)

10. Evaluate the loss on test set (5 pts)

