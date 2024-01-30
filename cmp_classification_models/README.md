# Comparing Classification Models
## 4/24/23

## Abstract
This assignment involved evaluating the different classification methods on the MNIST dataset. This dataset contains 70,000 images of 28x28 images of a single handwritten digit, with the images labeled for what number they represent. Each of the images therefore have 784 features (28x28).

Classification methods include a linear classifier, SVM, and the Decision Tree classifier.

## Table of Contents
•&emsp;[Introduction and Overview](#introduction-and-overview)

•&emsp;[Theoretical Background](#theoretical-background)

•&emsp;[Algorithm Implementation and Development](#algorithm-implementation-and-development)


&emsp;•&emsp;[Problem 1](#problem-1)
&emsp;•&emsp;[Problem 2](#problem-2)
&emsp;•&emsp;[Problem 3](#problem-3)
&emsp;•&emsp;[Problem 4](#problem-4)
&emsp;•&emsp;[Problem 5](#problem-5)
&emsp;•&emsp;[Problem 6](#problem-6)
&emsp;•&emsp;[Problem 7](#problem-7)
&emsp;•&emsp;[Problem 8](#problem-8)
&emsp;•&emsp;[Problem 9](#problem-9)
&emsp;•&emsp;[Problem 10](#problem-10)

•&emsp;[Computational Results](#computational-results)

&emsp;•&emsp;[Problem 1](#problem-1-1)
&emsp;•&emsp;[Problem 2](#problem-2-1)
&emsp;•&emsp;[Problem 3](#problem-3-1)
&emsp;•&emsp;[Problem 4](#problem-4-1)
&emsp;•&emsp;[Problem 5](#problem-5-1)
&emsp;•&emsp;[Problem 6](#problem-6-1)
&emsp;•&emsp;[Problem 7](#problem-7-1)
&emsp;•&emsp;[Problem 8](#problem-8-1)
&emsp;•&emsp;[Problem 9](#problem-9-1)
&emsp;•&emsp;[Problem 10](#problem-10-1)

•&emsp;[Summary and Conclusions](#summary-and-conclusions)

## Introduction and Overview
There are various methods to classifying images through supervised learning. As the MNIST dataset has been labeled for which digit a handwritten number represents, we can evaluate the accuracy of selected models on the MNIST dataset.

Classifiers applied were a linear classifier (LDA), a support vector machine (SVM), and a decision tree classifier. These were compared on how accurately they classified the ten digits from the MNIST dataset.

A sample of the MNIST dataset is given in Fig. 1 with their corresponding labels.

![MNIST](https://cdn.discordapp.com/attachments/1096628827762995220/1099215819491250186/image.png)

> Fig. 1. Sample images from the "MNIST" dataset

## Theoretical Background
The three classifiers involved in this assignment are the LDA, the SVD, and the decision tree classifier. The LDA splits between data linearly, while the SVD splits data by approximating a plane through all of the features (so in this case, a 784 dimension plane) that will split the data. By having a plane through this many dimensions, the SVM also has a buffer space for where the separation line could be, and wants to maximize the distance from the line to the edge of the buffer (finding the "most middle spot" the line could be between two different classes). The decision tree classifier does not perform optimization and instead selects the principle components that best split the data and iterates downward, finding the next best feature to split the data into different classes.

## Algorithm Implementation and Development
The procedure is discussed in this section. For the results, see [Computational Results](#computational-results).

Firstly, a sample subset was taken from the 70,000 image size dataset to speed computations. Each image and its features were assigned to one vector, X, and then transposed so that each column represneted a single image. Each images label, or target, was assigned to y, which was a 1D matrix containing the labels as type string.

```py
# Load the MNIST data
mnist = fetch_openml('mnist_784', parser="auto")
y = mnist.target
X = mnist.data / 255.0  # Scale the data to [0, 1]

# changing format of MNIST dataset so that each image is a column vector
X = X.to_numpy().T

# obtain random sample
rand_num = random.sample(range(X.shape[1]), 4000)
sample_x = X[:, rand_num]
sample_y = y[rand_num]
```
> Fig. 2. Retrieving images from the "yalefaces" dataset

### Problem 1
This task involved performing SVD on the dataset, which will factorize the matrix into matrices U, S, and V<sup>T</sup>. U is the unitary matrix of left side eigenvectors. S represents `np.diag(s)`, where s^2 are the eigenvalues. V<sup>T</sup> are the right side eigenvectors, which are transposed. 

```py
# get top six principle components
u, s, vt = np.linalg.svd(sample_x, full_matrices=False)
```

### Problem 2
The objective of this task is to determine the rank of the basis matrix and then plot the Singular Value Spectrum. This involved plotting the eigenvalue magnitude against the Principle Component Number. This was plotted to better visualize when the magnitude flattens. As seen here, the rank was found to be 29.

```py
# Determine the rank r
threshold = 0.1 * s[0]
r = np.sum(s > threshold)
```

For the images that are least correlated, 


### Problem 3
Here, the explanation for what U, S, and Vt represent from the `svd()` function was desired. The discussion is given below in [Results](#problem-3-1)

### Problem 4
The next task involved selecting three V-modes and plotting the projection of the dataset on the V mode vectors on a 3D scatterplot. The code for finding the projection is given below.

```py
# Projecting onto three V-modes, 0, 3, 5
proj_matrix = [0, 3, 5]

# 3d Scatterplot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot projection of each of the images corresponding to a digit
# onto the three V-modes 0, 3, 5
for i in range(10):
    mask = sample_y == i
    ax.scatter(vt[proj_matrix[0], mask], vt[proj_matrix[1], mask], vt[proj_matrix[2], mask], label=str(i), s=20)

# Labels and legend
ax.set_xlabel('V-mode ' + str(proj_matrix[0]))
ax.set_ylabel('V-mode ' + str(proj_matrix[1]))
ax.set_zlabel('V-mode ' + str(proj_matrix[2]))
ax.legend(title='Target Label', loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()
```

### Problem 5
The objective of this task is to pick two digits and attempt to split the training dataset using lines. This was done using a linear classifier, which was done with the `SGDClassifier()` function, which find lines that cut between the datapoints and attempt high accuracy in separating points of different labels. In other words, it optimally divides the data into classes.

```py
# create the linear model SGDclassifier
from sklearn.linear_model import SGDClassifier
linear_clf = SGDClassifier()

# for label = 0, 3
mask = np.logical_or((sample_y == 0), (sample_y == 3))
sample_x_clf = sample_x[:, mask].T
sample_y_clf = sample_y[mask]

# split train/test
X_train, X_test, y_train, y_test = train_test_split(sample_x_clf, sample_y_clf, test_size=0.2)

# Train the classifier using fit() function
linear_clf.fit(X_train, y_train)

# Evaluate the result
y_pred = linear_clf.predict(X_test)
# Calculate the accuracy of the classifier on the test set
accuracy = accuracy_score(y_test, y_pred)
```

The results of the classifier are also printed.


### Problem 6
The objective for this problem was to repeat the process in problem 5, but to now select 3 digits and attempt to classify between them. This was again done with the SVC module and fitting with lines (planes), but instead attempting to split between three classes within the data.

```py
# for label = 1, 4, 5
mask = np.logical_or((sample_y == 1), (sample_y == 4), (sample_y == 5))
sample_x_clf = sample_x[:, mask].T
sample_y_clf = sample_y[mask]

print("Number of samples with Label = 0 or 3:\nX:", sample_x_clf.shape, "\nY:", sample_y_clf.shape)

# split train/test
X_train, X_test, y_train, y_test = train_test_split(sample_x_clf, sample_y_clf, test_size=0.2)

# Train the classifier using fit() function
linear_clf.fit(X_train, y_train)

# Print the learned coeficients
print ("\nFirst 10 coefficients of the linear boundary are:", linear_clf.coef_[linear_clf.coef_ != 0][:10])
print ("\nThe point of intersection of the lines are:",linear_clf.intercept_)

# Evaluate the result
y_pred = linear_clf.predict(X_test)
# Calculate the accuracy of the classifier on the test set
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
```

### Problem 7
For this problem, the two digits that appear most difficult for the LDA to separate were to be found. A function to find the best and worst cases for each classifier when splitting between two digits was built for simplicity. This function is given below and was reused for problem 8 and 9.

```py
def find_worst_and_best(clf, sample_x, sample_y):
    worst = 100.0
    best = 0.0
    num1_w, num1_b, num2_w, num2_b = 0, 0, 0, 0

    for i in range(10):
        for j in range(i + 1, 10):
            # for varying labels
            mask = np.logical_or((sample_y == i), (sample_y == j))
            sample_x_clf = sample_x[:, mask].T
            sample_y_clf = sample_y[mask]

            # split train/test
            X_train, X_test, y_train, y_test = train_test_split(sample_x_clf, sample_y_clf, test_size=0.2)

            # Train the classifier on the training set
            clf.fit(X_train, y_train)

            # Predict the labels of the test set
            y_pred = clf.predict(X_test)

            # Calculate the accuracy of the classifier on the test set
            accuracy = accuracy_score(y_test, y_pred)
            if (min(accuracy, worst) == accuracy):
                worst = accuracy
                num1_w, num2_w = i, j
            elif (max(accuracy, best) == accuracy):
                best = accuracy
                num1_b, num2_b = i, j

    return (num1_w, num1_b, num2_b, num2_w, worst, best)
```


### Problem 8
Here, the output for problem 7's function `find_worst_and_best()` output the best case scenario as well and was therefore satisfied in problem 7.

### Problem 9
For this problem, the process in problem 7 was to be repeated but with SVMs and decision tree classifiers. Because of this, the same function was used and the outputs were printed.


# Computational Results
## Problem 1
Doing the SVD analysis on the digit images, we obtained the vectors U, S, and Vt.
The dimensions for these vectors are quite larger and not easily readable, therefore the vector sizes are given.

`U.shape: (784, 784) S.shape: (784,) V^t.shape: (784, 4000)`

For U, if the number of features represented as m, then U is mxm basis features for the images. For S, these are the diagonal values for the importance of each of the features. For V^t, the weights for each of the features are given for all 4000 images in the sample image set.


### Problem 2
Following the SVD analysis, finding the rank, or the basis or number of principle component vectors to represent the data, is found by evaluating the eigenvalue magnitude.

This is then visualized through the Singular Value Spectrum (Fig. 2), where the number of principle components increases, but the point to when the eignvalue magnitude flattens and approaches 0 is the threshold for the number of principle components to include.

![Alt text](https://cdn.discordapp.com/attachments/1096628827762995220/1099221249118437441/image.png)

> Fig. 2. Singular Value Spectrum

### Problem 3

The SVD will factorize a 2D matrix <i>A</i> into three pieces, the first being <i>U</i>, the left singular vectors that span the column space of A, <i>Σ</i>, the singular values which are also denoted as S and are the root of the eigenvalue magnitude, and <i>V<sup> T </sup></i>, where V contains the right singular vectors which span the row space of A, and are the weights that are applied to U.
For these images, U contains the significant features for each of the images within each column (one column corresponds to each image). S contains the magnitude or importance (principle component vectors) of each of those vectors, and V is the coefficients applied to reach the original image.


### Problem 4
Following the expectations of the prompt, the dataset is projected onto 3 V-modes in a 3D scatterplot (or 3 columns of the V matrix). I selected v-mode 0, 3, and 5. Each data point is colored based on the target label from the dataset. The scatterplot is given in Fig. 3.

![Alt text](https://media.discordapp.net/attachments/1096628827762995220/1099223088962490458/image.png?width%3D458%26height%3D386)

> Fig. 3. Scatterplot for Projection onto 3 V-modes


### Problem 5
I selected labels 0 and 3 to be split with the LDA.

Number of samples with Label = 0 or 3:
X: (816, 784) 
Y: (816,)
Split: (652, 784) (164, 784)

With the LDA, the first 10 coefficients of the linear boundary are: [-0.800048   -2.18679787 -1.0400624  -0.400024   -0.01777884  0.40446871
  1.04450711  2.08456952  1.21785085  0.8800528 ]

The point of intersection of the lines are: [24.8613047]
Accuracy: 0.9817073170731707

This accuracy is satisfactory. The next problem requested splitting between 3 labels.

### Problem 6
I selected labels 1, 4, and 5 to be split with the LDA.

Number of samples with Label = 1, 4 or 5:
X: (847, 784) 
Y: (847,)
Split: (677, 784) (170, 784)

With the LDA, the first 10 coefficients of the linear boundary are: [-1.10505959 -0.14393633  0.03250175  0.02321554 -0.32037442 -1.17934931
 -0.60360398  1.37900294 -0.2460847  -0.29715888]

The point of intersection of the lines are: [9.42804633]
Accuracy: 1.0

This accuracy is exemplary, especially when considering that it required more classes to split between.


By plotting the six SVD modes, it is clear that the first mode captures the most shared features of a face, and the later modes capture lesser features of the face. 

### Problem 7
Here, the worst cases for the LDA across combinations of all 10 digits were to be found. The results are:

Most difficult to separate (minimum accuracy) is at indices: 3 5
with an accuracy of: 0.9230769230769231 

This means that the LDA struggled most to classify between images of numbers 3 and 5.

### Problem 8
Here, the best cases for the LDA across combinations of all 10 digits were to be found. The results are:

Easiest to separate (maximum accuracy) is at indices: 6 5
with an accuracy of: 1.0

This means that the LDA did best to classify between images of numbers 6 and 5, and since the accuracy is 1.0, means that it classified every test datapoint correctly.

### Problem 9
Here, the SVM and Decision Tree classifiers were to be evaluated similarly to the LDA. Therefore, the same operations were performed and the results are in [Problem 10](#problem-10-1) since all three classifiers results are discusses and compared.

Additionally, the decision tree classifier was plotted and is in the file included in this repository, "tree_1.dot" as the image is quite large and detailed. The root and its subbranches are displayed below in Fig. 4.

![Alt text](https://media.discordapp.net/attachments/1096628827762995220/1099225422606123058/image.png?width%3D458%26height%3D179)

> Fig. 4. Root and subbranches of the decision tree classifier

Here, feature 657, which can be identified as pixel 657, is the most significant vector to split the class = 0 (or label is 0) upon. Then the next split on the left depends on feature 653, which splits a class = 5 label. To the right, the next split is on feature 406, splitting class 0 further. This tree continues and shows which features are best splitting the classes and which classes are under that split.

### Problem 10
Comparisons between all three classifiers are given:

For the LDA Classifier:
Most difficult to separate (minimum accuracy) is at indices: 3 5
with an accuracy of: 0.9230769230769231 

Easiest to separate (maximum accuracy) is at indices: 6 5
with an accuracy of: 1.0

For the SVM Classifier:
Most difficult to separate (minimum accuracy) is at indices: 5 8
with an accuracy of: 0.9256756756756757 

Easiest to separate (maximum accuracy) is at indices: 6 8
with an accuracy of: 1.0

For the Decision Tree Classifier:
Most difficult to separate (minimum accuracy) is at indices: 4 7
with an accuracy of: 0.9090909090909091 

Easiest to separate (maximum accuracy) is at indices: 1 7
with an accuracy of: 1.0

Overall, all of the classifiers did best separating different classes, though 6 appears across both SVM and Decision Tree classifiers, indicating that it is more easily distinguishable. Additionally, both LDA and SVMs struggled to distinguish images with the number 5, as this was found in its pairing for least accuracy.
The SVM Classifier had the greatest overall accuracy at its best and worst case and the Decision Tree Classifier had the worst overall accuracy at its worst case.


## Summary and Conclusions
Analyzing the differences between the classifiers, each one performs best splitting between different classes. However, the SVM is computationally heavy, while the decision tree classifier gives a good representation of how to interpret the model, but has so far performed the worst of the three.

Further tuning through hyperparameterization may be required for the classes to be better representative of the MNIST datasets, however they have satisfactory accuracy is splitting classes so far.

