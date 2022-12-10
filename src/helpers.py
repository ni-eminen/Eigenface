from collections import Counter

import numpy as np
from scipy.spatial.distance import cityblock, hamming
from PIL import Image


# None of these are needed any longer...

class EigenfaceHelpers:
    def __init__(self, img_shape):
        self.img_shape = img_shape

    def crop_image(self, img):
        width, height = img.size

        new_width = self.img_shape[0]
        new_height = self.img_shape[1]
        left = (width - new_width) / 2 - .5
        top = (height - new_height) / 2 - .5
        right = (width + new_width) / 2 - .5
        bottom = (height + new_height) / 2 - .5

        # Crop the center of the image
        im = img.crop((left, top, right, bottom))

        return im

    def img_to_vector(self, path_to_img):
        # Load the image
        img = Image.open(path_to_img).convert('L')
        img = self.crop_image(img)

        # Convert the image into n²*1 array
        arr = np.array(img)
        flat_array = arr.ravel()

        return flat_array

    def vector_to_img(self, vector):
        array = vector.reshape(self.img_shape)
        img = Image.fromarray(array)
        return img


def sum_of_vectors(arr: []):
    sum_vector = np.zeros(len(arr[0]))
    for i in range(0, len(arr[0])):
        for vector in arr:
            sum_vector[i] += vector[i]

    return sum_vector


def scalar_multiply_vector(scalar, vector):
    arr = np.array([])
    for x in vector:
        arr = np.append(arr, x * scalar)
    return arr


def negative_vector(vector):
    new_v = []
    for x in vector:
        new_v.append(-x)
    return np.array(new_v)


def euclidean_distance(a, b):
    difference_vector = a - b
    norms = np.linalg.norm(difference_vector, axis=1)
    return norms


def manhattan(a, b):
    distance_matrix = np.array([])
    for v in a:
        ham = cityblock(v, b)
        distance_matrix = np.append(distance_matrix, ham)
    return distance_matrix

def hamming_distance(a, b):
    distance_matrix = np.array([])
    for v in a:
        ham = hamming(v, b)
        distance_matrix = np.append(distance_matrix, ham)
    return distance_matrix



def predict(Xtest, targets, idx_train, idx_test, avg_face, proj_data, w, distance_func=manhattan, type="",
            sample_size=3, threshold=2):
    predicted_ids = []
    correct_ids = []

    for test_index in range(len(Xtest)):
        unknown_face_vector = Xtest[test_index]
        mean_unknown_face = np.subtract(unknown_face_vector, avg_face)
        w_unknown = np.dot(proj_data, mean_unknown_face)
        difference_vector = distance_func(w, w_unknown)

        if type == "multi":
            index = multi_id_prediction(difference_vector, sample_size, threshold, idx_train, targets)
        else:
            index = np.argmin(difference_vector)

        # Store the correct ids and the predicted ids in corresponding indices
        correct_ids.append(index_to_id(test_index, idx_test, targets))
        predicted_ids.append(index_to_id(index, idx_train, targets))

    return correct_ids, predicted_ids


def index_to_id(idx, idx_train, targets):
    return targets[idx_train[idx]]


def multi_id_prediction(norms: np.array, sample_size: int, threshold: int, idx_train, targets):
    indices = np.argpartition(norms, sample_size)[:sample_size]  # multi_id_prediction(norms, 0, multi_n)
    represented_ids = [index_to_id(i, idx_train, targets) for i in indices]
    most_common_id = n_of_the_same(represented_ids, threshold)

    # Return the most common ids index
    for i, x in enumerate(represented_ids):
        if x == most_common_id:
            return indices[i]

    return 0


def n_of_the_same(arr: list, n: int):
    # Check if the array is empty
    if len(arr) == 0:
        return None

    threshold_passing_items = []

    # Use a Counter to count the number of times each item appears in the array
    count = Counter(arr)

    # Return the first item that appears at least n times in the array
    for item, c in count.items():
        if c >= n:
            threshold_passing_items.append((item, c))

    if len(threshold_passing_items) == 0:
        return None
    else:
        most_common = sorted(threshold_passing_items, key=lambda x: x[1], reverse=True)
        return most_common[0][0]
