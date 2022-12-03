#!/usr/bin/env python
# coding: utf-8

# ### Let's import numpy, PIL and some sklearn modules for the popular olivetti dataset and splitting training data.

# In[1]:


import numpy as np
from PIL import Image
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from helpers import EigenfaceHelpers, negative_vector


# ### Define constants.

# In[2]:


# Define constants
IMG_SHAPE = (64, 64)


# ### Download the dataset, reshape the images into vectors and split it into pieces for training.
# - We fetch the olivetti dataset via sklearn
# - Olivetti.images is a collection of vectors, raveled 64x64 sized images
# - olivetti.target contains the id's of the people in the X array in the corresponding indices
# - We give the indices to the train_test_split to track which person is in which index after the function shuffles them, this will later be used to determine whether the algorithm predicted the correct person

# In[3]:


# Download Olivetti faces dataset
olivetti = fetch_olivetti_faces()
X = olivetti.images
y = olivetti.target

# Print info on shapes and reshape where necessary
X = X.reshape((400, 4096))
indices = np.arange(len(X))
Xtrain, Xtest, ytrain, ytest, idx_train, idx_test = train_test_split(X, y, indices)


# ### Construct the average face from the training set.
# - Add all training vectors together and divide the sum by the number of images.

# In[4]:


training_set = Xtrain
# Average face using numpy
avg_face = training_set.mean(axis=0)


# ### Derive normalized faces
# - Subtract the average face from each of the faces in the training set

# In[5]:


# Let's create the matrix A by subtracting the average face from each face in the training set
normalized_faces = []
neg_avg_face = negative_vector(avg_face)
sub = None
for v in training_set:
    sub = np.subtract(v, avg_face)
    normalized_faces.append(sub)

# Convert normalized faces array to a matrix
normalized_faces_matrix = np.asmatrix(normalized_faces)


# ### Form the covariance matrix
# - Transpose the matrix of normalized faces
# - Multiply the normalized faces matrix with its transposition

# In[6]:


# Form the covariance matrix
normalized_faces_t = np.array(normalized_faces).transpose()

# cov_matrix = (normalized_faces_matrix)(normalized_faces_t)
cov_matrix = np.cov(np.array(normalized_faces))


# ### Calculate the eigenvalues and eigenvectors for the coavariance matrix
# - In order to determine the strongest eigenfaces, we select the eigenvectors with the highest corresponding eigenvalues
# - Pair the eigenvalues/eigenvectors
# - Sort the pairs based on the highest eigenvalues

# In[7]:


# Calculate the eigenvectors of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
eig_pairs = [(eigenvalues[index], eigenvectors[:, index]) for index in range(len(eigenvalues))]

eig_pairs.sort(reverse=True)
eigvalues_sort = [eig_pairs[index][0] for index in range(len(eigenvalues))]
eigvectors_sort = [eig_pairs[index][1] for index in range(len(eigenvalues))]


# ### Select the 20 best eigenvectors

# In[8]:


# Choose the 10 eigenvectors with the highest eigenvalues as the eigenfaces
eigenfaces = np.array(eigvectors_sort[:20]).transpose()


# ### Create reduced eigenface space and calculate the weights for the projected vectors
# - Project the eigenfaces to the training_sets transposition by performing a dot product between the two
# - A weight is calculated by performing a dot product between each normalized face and the projections

# In[9]:


proj_data = np.dot(training_set.transpose(), eigenfaces)
proj_data = proj_data.transpose()

# Calculate weights for eigenfaces
w = np.array([np.dot(proj_data, i) for i in np.array(normalized_faces)])


# ### Calculate distance between the weights of each eigenface and the test image
# - Create the normalized unknown face
# - Calculate the weights of the normalized unknown weights in respect to the projections
# - Create the difference vector, which is the weights of the eigenfaces subracted from the weights of the test image
# - Find the index of the lowest difference

# In[10]:


correct_ids = []
predicted_ids = []

# Get images from dataset and convert them to vectors
test_index = 20
for test_index in range(len(Xtest)):
    unknown_face_vector = Xtest[test_index]
    mean_unknown_face = np.subtract(unknown_face_vector, avg_face)
    w_unknown = np.dot(proj_data, mean_unknown_face)
    difference_vector = w - w_unknown
    norms = np.linalg.norm(difference_vector, axis=1)
    index = np.argmin(norms)

    # Store the correct ids and the predicted ids in corresponding indices
    correct_ids.append(y[idx_test[test_index]])
    predicted_ids.append(y[idx_train[index]])


# ### Print results

# In[11]:


from sklearn.metrics import classification_report
print(classification_report(correct_ids, predicted_ids, zero_division=0))

