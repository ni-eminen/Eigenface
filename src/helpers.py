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

    @staticmethod
    def sum_of_vectors(arr: []):
        sum_vector = np.zeros(len(arr[0]))
        for i in range(0, len(arr[0])):
            for vector in arr:
                sum_vector[i] += vector[i]

        return sum_vector

    @staticmethod
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
