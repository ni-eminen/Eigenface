import numpy as np
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

def two_of_the_same(arr: list):
    if len(arr) == 0:
        return None
    while(len(arr)>0):
        item = arr.pop()
        if item in arr:
            return item
    return None

def predictions(Xtest, y, idx_train, idx_test, avg_face, proj_data, w, type: str):
    predicted_ids = []
    correct_ids = []

    for test_index in range(len(Xtest)):
        unknown_face_vector = Xtest[test_index]
        mean_unknown_face = np.subtract(unknown_face_vector, avg_face)
        w_unknown = np.dot(proj_data, mean_unknown_face)
        difference_vector = w - w_unknown
        norms = np.linalg.norm(difference_vector, axis=1)

        if type == "multi":
            index = multi_id_prediction(norms, 10000)
        else:
            index = np.argmin(norms)

        # Store the correct ids and the predicted ids in corresponding indices
        correct_ids.append(y[idx_test[test_index]])
        predicted_ids.append(y[idx_train[index]])

    return correct_ids, predicted_ids

def multi_id_prediction(norms, not_recognized: int, ):
    index = []
    for i in range(0, 2):
        index.append(np.argmin(norms))

    predicted_index = two_of_the_same(index)
    if predicted_index is None:
        return not_recognized
    else:
        return predicted_index