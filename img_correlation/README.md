
# Analyzing Correlation Between Images
## 4/18/23

## Abstract
The second assignment for EE 399, Introduction to Machine Learning for Science and Engineering, uses the "yalefaces" dataset of over 2000 images of faces under different lighting and finds the correlation and most significant feature spaces between the faces.

## Table of Contents
•&emsp;[Introduction and Overview](#introduction-and-overview)

•&emsp;[Theoretical Background](#theoretical-background)

•&emsp;[Algorithm Implementation and Development](#algorithm-implementation-and-development)


&emsp;•&emsp;[Problem A](#problem-a)
&emsp;•&emsp;[Problem B](#problem-b)
&emsp;•&emsp;[Problem C](#problem-c)
&emsp;•&emsp;[Problem D](#problem-d)
&emsp;•&emsp;[Problem E](#problem-e)
&emsp;•&emsp;[Problem F](#problem-f)
&emsp;•&emsp;[Problem G](#problem-g)

•&emsp;[Computational Results](#computational-results)

&emsp;•&emsp;[Problem A](#problem-a-1)
&emsp;•&emsp;[Problem B](#problem-b-1)
&emsp;•&emsp;[Problem C](#problem-c-1)
&emsp;•&emsp;[Problem D](#problem-d-1)
&emsp;•&emsp;[Problem E](#problem-e-1)
&emsp;•&emsp;[Problem F](#problem-f-1)
&emsp;•&emsp;[Problem G](#problem-g-1)

•&emsp;[Summary and Conclusions](#summary-and-conclusions)

## Introduction and Overview
Finding the correlation matrix or applying the SVD are methods of determining the relationship between given images, such as within the "yalefaces" dataset.

A correlation matrix gives the relative relationship between images through the dot product of the vectorization of the images, while the SVD determines the principle component vectors of the given data, the feature space which best defines a face.

Using these, the most and least correlated images, as well as the most shared features of a face, can be illustrated.

A sample of the "yalefaces' dataset is shown in Fig. 1.

<p><img src="https://media.discordapp.net/attachments/1096628827762995220/1097380398985642134/image.png" width=400></p>

> Fig. 1. Sample faces from the "yalefaces" dataset

## Theoretical Background
To find the correlation between two objects, they can be vectorized and then input into the dot product.

With the dot product, two vectors that are the same have a result of 1, and two vectors that are orthogonal have a result of 0. In other words, if two images were to be vectorized, then the correlation between the two images can be represented by the dot product of their vectors.

When applying the dot product to a set of images, the result is a 2D array, where each index is the correlation between the two corresponding images for that index. Larger values in this correlation matrix represent images that were evaluated by the dot product to share more features, while smaller values indicate less shared features.

To find the shared features among the faces in the dataset, the SVD, or Singular Value Decomposition, can be applied. The SVD will factorize a 2D matrix <i>A</i> into three pieces, the first being <i>U</i>, the left singular vectors, <i>Σ</i>, the singular values, and <i>V<sup> T </sup></i>, where V contains the right singular vectors.

The SVD can be applied using the `np.svd` function, which returns `U`, the left singular vectors as above, `S`, the singular values or the root of the eigenvalues (Σ in the SVD function), and the right singular vectors as `V`. Plotting the vectors for the largest found eigenvalues will visualize the feature space for the most prominant elements of each of the given images, which is done below.

## Algorithm Implementation and Development
The procedure is discussed in this section. For the results, see [Computational Results](#computational-results).

The "yalefaces" dataset was used for all problems, and the images were stored in matrix `X`, resulting in 2414 images of 1024 elements each, or a 1024x2414 array (Fig. 2).

```py
# load data and save images into X
results=loadmat('yalefaces.mat')
X=results['X']
```
> Fig. 2. Retrieving images from the "yalefaces" dataset

### Problem A
The objective of this task is to compute a 100x100 correlation matrix, `C`, between the first 100 images in the given dataset.
This correlation is found by applying the dot product between the 100 images, which was accomplished with the `np.dot` function. As a result, the elements of `C` is the correlation between the image of its x-index, and the image of its y-index. This correlation matrix is then plotted using matplotlib's pcolor function.

```py
# get first 100 images
hundr = X[:, :100]
C = np.ndarray((100, 100))

# correlation matrix C, dot product between first 100 images
C = np.dot(hundr.T, hundr)

# plot the correlation matrix
fig, ax = plt.subplots()
ab = ax.pcolor(range(0, 100), range(0, 100), C, vmin = np.min(C), vmax=np.max(C))
ax.set_title("Correlation Between First 100 Images")
ax.set_xlabel("Selected Image 1")
ax.set_ylabel("Selected Image 2")
fig.colorbar(ab, ax=ax, label="Correlation")
```

### Problem B
For this task, the two images that are most and least correlated are to be found and visualized. This was accomplished by retrieving the index values for the largest value of correlation in `C`. These index numbers are the image numbers.

Initially, just retrieving the most correlated images returns two of the same image, which makes sense, but to find correlation between two differing images, the next greatest correlation should be found.

```py
# get the most correlated images
most = np.argwhere(C == np.max(C))[0]

# get the least correlated images
least = np.argwhere(C == np.min(C))[0]

# notice that images are the same, repeat process
# but with the next greatest/least correlation when indices are different
most = np.argwhere(C == np.sort(C.flatten())[-3])[0]
least = np.argwhere(C == np.sort(C.flatten())[1])[0]
```

For the images that are least correlated, 


### Problem C
Here, the process from problem A is repeated, but with the given images (1-based):
`[1, 313, 512, 5, 2400, 113, 1024, 87, 314, 2005]`

The code is relatively similar to problem A, but with a 10x10 matrix `C` with only these images selected.

```py
# array to store correlation between the ten images
C = np.ndarray((10, 10))
# find the correlation matrix
C = np.dot(get_img.T, get_img)

# plot the correlation matrix
fig, ax = plt.subplots()
corr = ax.pcolor(range(0, 10), range(0, 10), C, vmin=np.min(C), vmax=np.max(C))
ax.set_title("Correlation Between Given Ten Images")
ax.set_xlabel("Selected Image 1")
ax.set_ylabel("Selected Image 2")
fig.colorbar(corr, ax=ax, label="Correlation")

# get the most and least correlated images (that are not the same image)
most = np.argwhere(C == np.sort(C.flatten())[-3])[0]
least = np.argwhere(C == np.sort(C.flatten())[1])[0]
```

### Problem D
For this task, matrix `Y` is created with the equation in Fig. 3. 

![Generating Y](https://cdn.discordapp.com/attachments/1096628827762995220/1097393110025248768/image.png)
> Fig. 3. Creating Matrix Y

Using `scipy.sparse.linalg.eigs` function, we can find the six largest eigenvalues and their corresponding eigenvectors across all images in the dataset. This is shown below.

```py
Y = np.matmul(X, X.T)

# eigenvalues stored in w
# eigenvectors stored in v as an array,
    # with the eigenvector corresponding to eigenvalue [i] is v[:, i]
# LM - find k=6 eigenvalues with the Largest Magnitude
w, v = scipy.sparse.linalg.eigs(Y, k=6, which="LM")
```

### Problem E
Here, the SVD for the dataset. The first six modes, or principle component directions is found and saved under `dir`.
The SVD altogether returns the eigenvectors as rows or as columns and the root of the eigenvalues in diagonal matrix S.

U are eigenvectors of <i>X X<sup>T</sup></i>, V are the eigenvectors of <i>X<sup>T</sup> X</i>.
S is the list of values in the diagonal matrix `(S = np.diag(s))`, where the eigenvalues are found with s<sup>2</sup>.

Since the rows of matrix U correspond to the rows of X, our data matrix, and the columns are the ordered eigenvectors of X X<sup>T</sup>, we can retrieve the first six principle directions with the first six columns of matrix U.

```py
u, s, vh = np.linalg.svd(X)
dir = u[:, :6]
```

### Problem F
Following the results of D and E, we can see how the two methods of finding and returning the eigenvectors for the largest magnitude eigenvalues compare on the first eigenvector (the largest magnitude eigenvalue).
These results were visualized with print statements, which are shown under [Computational Results for Problem F](#problem-f-1).

The norm of difference of the absolute values between the eigenvectors from Problem D (`v1`) and Problem E (`u1`) using the SVD was found using the code below.
```
# norm of difference of the absolute values 
norm = np.linalg.norm(np.abs(v1) - np.abs(u1))
```

### Problem G
Finally, the percentage of variance captured by each of the first six SVD modes was determined. This proportion was determined by dividing the eigenvalue by the sum of the eigenvalues. The variance of the data along an eigenvector is the eigenvalue corresponding to that eigenvector. Eigenvalues were found through the matrix `S`, where each value in the diagonal gives the eigenvalue through s<sup>2</sup>.

```py
# finding eigenvalues
var = s[:6] ** 2 / dir.length()
# determining percentage of variance
prop_var = var/np.sum(var) * 100
```

# Computational Results
## Problem A
Plotting the correlation matrix `C`, where the values of the indices is the image selected + 1 (the plot is index-0 but the images are 1-based), the correlation between the first 100 images in the dataset can be visualized. There appears to be a greater correlation to the top right, where the values are larger (Fig. 4).
![Correlation Between First 100 Images](https://media.discordapp.net/attachments/1096628827762995220/1097380756982083644/image.png?width%3D520%26height%3D418)
> Fig. 4. Correlation Between First 100 Images

### Problem B
Following the visualization of the correlation matrix, the most and least correlated images can be found. As expected, the most correlated images appear to be from the top left, at index `[86, 88]`, which corresponds to images `87` and `89`. These images are plotted in Fig. 5.

![Most Correlated Faces](https://media.discordapp.net/attachments/1096628827762995220/1097380774736560248/image.png)
> Fig. 5. Most Correlated Faces

The least correlated images are difficult to determine just from the correlation matrix plot, as there are several dark blue regions indicating very low correlation. However, the least correlated images were determined to be at index `[54, 64]`, corresponding to images `55` and `65`. These images are also plotted, given in Fig. 6. below.

![Least Correlated Faces](https://media.discordapp.net/attachments/1096628827762995220/1097380793397030922/image.png)
> Fig. 6. Least Correlated Faces

### Problem C
As Problem C repeats Problem A, the correlation matrix is given in Fig. 7. Again, the most images with the greatest correlation appear to be in the upper right.

![Correlation Between Given Ten Images](https://cdn.discordapp.com/attachments/1096628827762995220/1097400797689548870/image.png)
> Fig. 7. Correlation Between Given Ten Images

For verification, the images with the greatest and least magnitude of correlation were plotted in Fig. 8 and Fig. 9.

It is determined that the most correlated faces are from `8`, and `9`, which is at index `[7, 8]` in the correlation matrix, matching the location of the greatest correlation, which is the yellow region.

![Most Correlated Faces](https://media.discordapp.net/attachments/1096628827762995220/1097380824963366953/image.png)
> Fig. 8. Most Correlated Faces

The least correlated faces are from `2`, and `3`, which is at index `[1, 2]` in the correlation matrix, matching a the dark blue region where the correlation value is at its least.

![Least Correlated Faces](https://media.discordapp.net/attachments/1096628827762995220/1097380835793047623/image.png)
> Fig. 9. Least Correlated Faces


### Problem D
Following the expectations of the prompt, the first six eigenvectors with the largest magnitude eigenvalue and printed below. By specifying `k=6` and `LM` in the `scipy.sparse.linalg.eigs` function, the six eigenvalues of Largest Magnitude are returned and can be printed directly from the returned array.

```py
Eigenvalue: 234020.45485388613
Eigenvector: [-0.02384327 -0.02576146 -0.02728448 ... -0.02082937 -0.0193902
 -0.0166019 ]
Eigenvalue: 49038.3153005924
Eigenvector: [ 0.04535378  0.04567536  0.04474528 ... -0.03737158 -0.03557383
 -0.02965746]
Eigenvalue: 8236.539897013148
Eigenvector: [0.05653196 0.04709124 0.0362807  ... 0.06455006 0.06196898 0.05241684]
Eigenvalue: 6024.871457930171
Eigenvector: [ 0.04441826  0.05057969  0.05522219 ... -0.01006919 -0.00355905
  0.00040934]
Eigenvalue: 2051.496432691047
Eigenvector: [-0.03378603 -0.01791442 -0.00462854 ...  0.06172201  0.05796353
  0.05757412]
Eigenvalue: 1901.0791148236622
Eigenvector: [0.02207542 0.03378819 0.04487476 ... 0.03025485 0.02850199 0.00941028]
```


### Problem E

```
[[-0.02384327  0.04535378  0.05653196 -0.04441826  0.03378603 -0.02207542]
 [-0.02576146  0.04567536  0.04709124 -0.05057969  0.01791442 -0.03378819]
 [-0.02728448  0.04474528  0.0362807  -0.05522219  0.00462854 -0.04487476]
 ...
 [-0.02082937 -0.03737158  0.06455006  0.01006919 -0.06172201 -0.03025485]
 [-0.0193902  -0.03557383  0.06196898  0.00355905 -0.05796353 -0.02850199]
 [-0.0166019  -0.02965746  0.05241684 -0.00040934 -0.05757412 -0.00941028]]
```

### Problem F
Although not required, visualizing the vectors before finding the norm is helpful to assess results. `v1`, the first eigenvector from Problem D and `u1`, the first mode from Problem E are printed below. However, since the array is truncated due to its large size, comparing each element between the two arrays at increasing precision can show how close they are in value. This is shown in the print statements `"Elements match to the nth place?"`, which will output either `True` or `False`.

Finally the norm of difference of absolute values between the two vectors is given, which appears to be incredibly small.

```
v1 from D:
 [-0.02384327 -0.02576146 -0.02728448 ... -0.02082937 -0.0193902
 -0.0166019 ]
u1 from E:
 [-0.02384327 -0.02576146 -0.02728448 ... -0.02082937 -0.0193902
 -0.0166019 ]
Elements match to the 10th place? True
Elements match to the 11th place? True
Elements match to the 12th place? True
Elements match to the 13th place? True
Elements match to the 14th place? False
Elements match to the 15th place? False

Norm of difference of absolute values: 1.6670647475993934e-15
```

### Problem G
The percentage of variance captured by each of the first six SVD modes is printed below. The modes are 1-based.
```
Mode 1: 77.677%
Mode 2: 16.277%
Mode 3: 2.734%
Mode 4: 2.0%
Mode 5: 0.681%
Mode 6: 0.631%
```

Since the 1st Mode captures the a majority percentage of the variance in the dataset, then it is most representative of all the faces as it captures the most defining feature space for a face, as determined by the dataset. Through the later modes, the percentage of variance is much less and decreases rapidly, meaning that they represent lesser components of the face across the dataset. This is not only seen through the percentage of variance printed for each mode, but also by observing the plot (Fig. 10.)

![First 6 SVD Modes](https://media.discordapp.net/attachments/1096628827762995220/1097408017143693322/image.png)
> Fig. 10. First six SVD Modes

By plotting the six SVD modes, it is clear that the first mode captures the most shared features of a face, and the later modes capture lesser features of the face. 

## Summary and Conclusions
Analyzing the relationship between the images of the "yalefaces" dataset was done through finding a correlation matrix of the first 100 images, of select images, and through applying the SVD on all images and extracting the first six SVD modes (or the most representative feature spaces).

Results were visualized by plotting the correlation matrix, illustrating the images with greatest and least correlation, and evaluating the SVD by calculating the percentage of variance captured by each mode, as well as visually verifying results in Fig. 10.

