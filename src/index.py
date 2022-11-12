import PIL.Image
import numpy as np
from PIL import Image
import glob

IMG_SHAPE = (250, 250)


def img_to_vector(path_to_img):
    # Load the image
    img = Image.open(path_to_img).convert('L')

    # Convert the image into n²*1 array
    arr = np.array(img)
    flat_array = arr.ravel()

    return flat_array


def vector_to_img(vector, shape):
    array = vector.reshape(shape)
    img = Image.fromarray(array)
    return img


def sum_of_vectors(arr: []):
    sum_vector = np.zeros(len(arr[0]))
    for i in range(0, len(arr[0])):
        for v in arr:
            sum_vector[i] += v[i]

    return sum_vector


def scalar_multiply_vector(scalar, v):
    arr = np.array([])
    for x in v:
        arr = np.append(arr, x * scalar)
    return arr


def negative_vector(v):
    new_v = []
    for x in v:
        new_v.append(-x)
    return np.array(new_v)


all_imgs = []

# Get images from dataset and convert them to vectors
for folder in glob.iglob('../dataset/*'):
    for img in glob.iglob(folder + '/*'):
        all_imgs.append(img_to_vector(img))

# Create an np.array from the vectors
training_set = np.array(np.array(all_imgs))

# Manual calculation of the average face:
# avg_face = sum_of_vectors(training_set) * 1 / len(training_set)

avg_face = training_set.mean(axis=0)

# Let's create the matrix A by subtracting the average face from each face in the training set
A = []
neg_avg_face = negative_vector(avg_face)
for v in training_set:
    A.append(sum_of_vectors([v, neg_avg_face]))

# Convert A to a matrix
A_m = np.asmatrix(A)

A_t = A_m.transpose()

# Form the covariance matrix
cov_matrix = np.matmul(A_m, A_t)
