"""This file contains helper functions for the eigenfaces method. Additionally a class EigenfaceHelpers that
    contains similar, contextual methods."""

from collections import Counter

import numpy as np
from scipy.spatial.distance import cityblock, hamming
from PIL import Image


# None of these are needed any longer...

class EigenfaceHelpers:
    """The EigenfaceHelpers class contains functions that help in transforming vectors, cropping images etc."""
    def __init__(self, img_shape):
        """
        Initialize the EigenfaceHelpers
        :rtype: object
        """
        self.img_shape = img_shape

    def crop_image(self, img: Image):
        """Crop the center of an image.

        This function crops the center of the image to the specified width and height. The width and height
        of the cropped image are specified by the `img_shape` attribute of the object.

        Parameters
        ----------
        img : Image
            The image to be cropped.

        Returns
        -------
        im : Image
            The cropped image.
        """
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
        """
        Converts an image in to a 1d vector
        :param path_to_img: Path to the image
        :return: 1d np.array
        """
        # Load the image
        img = Image.open(path_to_img).convert('L')
        img = self.crop_image(img)

        # Convert the image into n²*1 array
        arr = np.array(img)
        flat_array = arr.ravel()

        return flat_array

    def vector_to_img(self, vector):
        """
        Converts a vector into a PIL image
        :param vector: Image in vector format
        :return: The image of type PIL.Image
        """
        array = vector.reshape(self.img_shape)
        img = Image.fromarray(array)
        return img


def sum_of_vectors(arr: []):
    """
    Sums an array of vector into one vector
    :param arr: Array of vectors. The vectors can be of any 1d iterable type
    :return: The sum of all the vectors
    """
    sum_vector = np.zeros(len(arr[0]))
    for i in range(0, len(arr[0])):
        for vector in arr:
            sum_vector[i] += vector[i]

    return sum_vector


def scalar_multiply_vector(scalar, vector):
    """
    Scales a vector by a scalar
    :param scalar: Scalar
    :param vector: Vector
    :return: The vector scaled with the scalar
    """
    arr = np.array([])
    for x in vector:
        arr = np.append(arr, x * scalar)
    return arr


def negative_vector(vector):
    """
    Negates a vector.
    :param vector: A 1d array
    :return: The 1d array with all its elements scaled by -1.
    """
    new_v = []
    for x in vector:
        new_v.append(-x)
    return np.array(new_v)


def euclidean_distance(a: np.array, v: np.array):
    """
    Returns an array of euclidean distance between a vector and a matrix's component vectors.
    :param a: An array of vectors
    :param v: A vector
    :return: An array of Euclidean distances between the array elements and the vector b
    """
    if a.shape[1] != v.shape[0]:
        print(a.shape, v.shape)
        return None
    difference_vector = a - v
    norms = np.linalg.norm(difference_vector, axis=1)
    return norms


def manhattan_distance(a: np.array, v: np.array):
    """
    Returns and array of manhattan distances between the array b and the arrays in matrix a.
    :param a: 2d array
    :param v: 1d array
    :return: 1d array of Manhattan distances between corresponding elements in a with the vector v
    """
    if np.array(a).shape[1] != np.array(v).shape[0]:
        return None

    distance_matrix = np.array([])
    for v_a in a:
        ham = cityblock(v_a, v)
        distance_matrix = np.append(distance_matrix, ham)
    return distance_matrix


def hamming_distance(a: np.array, v: np.array):
    """
    Returns and array of Hamming distances between the array b and the arrays in matrix a.
    :param a: 2d array
    :param v: 1d array
    :return: 1d array of Hamming distances between corresponding elements in a with the vector v
    """
    if np.array(a).shape[1] != np.array(v).shape[0]:
        return None

    distance_matrix = np.array([])
    for v_a in a:
        ham = hamming(v_a, v)
        distance_matrix = np.append(distance_matrix, ham)
    return distance_matrix


def predict(xtest, targets, idx_train, avg_face, proj_data, w, distance_func=manhattan_distance, type_="",
            sample_size=3, threshold=2):
    """
    Predicts ids for each of the images in Xtest
    :param xtest: An array of vectors representing images
    :param targets: The target ids
    :param idx_train: The indexes that correspond to the correct ids within the targets array
    :param avg_face: The average face that has been calculated using the dataset
    :param proj_data: The projection of the training images onto the eigenfaces
    :param w: The weights for the projections
    :param distance_func: Preferred distance measurement for measuring the distance between the test face weights and the projections' weights
    :param type_: The type of evaluation. "KNN" for K-nearest number evaluation of the face.
    :param sample_size: If "KNN" type has been selected, this is the amount of nearest neighbours that are evaluated
    :param threshold: The amount of identical ids required to identify a face
    :return: An array of predicted ids. Each index corresponds the index in the Xtest array
    """
    predicted_ids = []
    for test_index in range(len(xtest)):
        unknown_face_vector = xtest[test_index]
        mean_unknown_face = np.subtract(unknown_face_vector, avg_face)
        w_unknown = np.dot(proj_data, mean_unknown_face)
        difference_vector = distance_func(w, w_unknown)

        if type_ == "KNN":
            index = KNN_prediction(difference_vector, sample_size, threshold, idx_train, targets)
        else:
            index = np.argmin(difference_vector)

        # Store the correct ids and the predicted ids in corresponding indices
        predicted_ids.append(index_to_id(index, idx_train, targets))

    return predicted_ids


def index_to_id(idx, idx_train, targets):
    """
    Finds the target id an index in idx_train represents
    :param idx: The index for the element in idx_train that's id is to be retrieved
    :param idx_train: A list of indexes that correspond to the ids of the test set
    :param targets: The targets of the prediction
    :return: Returns the id that represents the image in idx_train[idx]
    """
    return targets[idx_train[idx]]


def KNN_prediction(norms: np.array, sample_size: int, threshold: int, idx_train, targets):
    """
    K-nearest neighbour for predictions.
    :param norms: The norms that determine which face is the closest to the test face
    :param sample_size: The amount of neighbours to be evaluated
    :param threshold: How many neighbours must be of the same id for the image to be identified
    :param idx_train: An array of indexes that can be used to find the id of a given index in idx_train
    :param targets: The targets for the prediction
    :return: Returns the most common element in the nearest neighbours that passes the threshold. If no id passes the threshold, returns 0
    """
    # Get indices of the smallest values in norms array
    indices = np.argpartition(norms, sample_size)[:sample_size]

    # Convert the indices to the person ids they represent
    represented_ids = [index_to_id(i, idx_train, targets) for i in indices]

    most_common_id = n_of_the_same(represented_ids, threshold)

    # Return the most common ids index
    for i, x in enumerate(represented_ids):
        if x == most_common_id:
            return indices[i]

    return 0


def n_of_the_same(arr: list, n: int):
    """
    Returns an element if it appears n time within an array
    :param arr: An iterable
    :param n: The threshold for the times an item has to appear
    :return: The item that appears the most times, passing the threshold. Otherwise None.
    """
    # Check if the array is empty
    if len(arr) == 0:
        return None

    threshold_passing_items = []

    # Use a Counter to count the number of times each item appears in the array
    count = Counter(arr)

    # Create an array of the items that appear more than or equal to the times mentioned in the threshold
    for item, c in count.items():
        if c >= n:
            threshold_passing_items.append((item, c))

    # Return the item that appears most commonly
    if len(threshold_passing_items) == 0:
        return None
    else:
        most_common = sorted(threshold_passing_items, key=lambda x: x[1], reverse=True)
        return most_common[0][0]
