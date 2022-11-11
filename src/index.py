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
    array = vector.reshape(shape) #np.asarray(vector).reshape(shape)
    img = Image.fromarray(array)
    return img

all_imgs = []

# Get images from dataset and convert them to vectors
for folder in glob.iglob('../dataset/*'):
    for img in glob.iglob(folder + '/*'):
        all_imgs.append(img_to_vector(img))

# Create an np.array from the vectors
np_images = np.array(all_imgs)
avg_face = np_images.mean(axis=0)

vector_to_img(avg_face, IMG_SHAPE).show()

# Let's calculate the average face
#avg_multiplier = 1/len(np_images)
#sum_of_faces = sum(np_images)
#avg_face = sum_of_faces/len(np_images)
#vector_to_img(np.array(avg_face), IMG_SHAPE).show()