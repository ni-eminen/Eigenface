import numpy as np
from PIL import Image
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from helpers import EigenfaceHelpers, negative_vector


class Eigenfaces:
    def __init__(self, DATASET_SIZE=400):
        self.DATASET_SIZE = DATASET_SIZE
        self.IMG_SHAPE = (64, 64)

    def train_on_olivetti_dataset(self):
        olivetti = fetch_olivetti_faces()

        # Create a larger dataset (with repetition, but it is not relevant in this context)
        for i in range(0, 5):
            olivetti.images = np.append(olivetti.images, olivetti.images, axis=0)
            olivetti.target = np.append(olivetti.target, olivetti.target, axis=0)
        X = olivetti.images[:self.DATASET_SIZE]
        y = olivetti.target[:self.DATASET_SIZE]
        X = X.reshape((self.DATASET_SIZE, 4096))
        indices = np.arange(len(X))
        Xtrain, Xtest, ytrain, ytest, idx_train, idx_test = train_test_split(X, y, indices)
        training_set = Xtrain
        avg_face = training_set.mean(axis=0)
        normalized_faces = []
        neg_avg_face = negative_vector(avg_face)
        sub = None
        for v in training_set:
            sub = np.subtract(v, avg_face)
            normalized_faces.append(sub)
        normalized_faces_matrix = np.asmatrix(normalized_faces)
        normalized_faces_t = np.array(normalized_faces).transpose()
        cov_matrix = np.cov(np.array(normalized_faces))
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        eig_pairs = [(eigenvalues[index], eigenvectors[:, index]) for index in range(len(eigenvalues))]
        eig_pairs.sort(reverse=True)
        eigvalues_sort = [eig_pairs[index][0] for index in range(len(eigenvalues))]
        eigvectors_sort = [eig_pairs[index][1] for index in range(len(eigenvalues))]
        eigenfaces = np.array(eigvectors_sort[:20]).transpose()
        proj_data = np.dot(training_set.transpose(), eigenfaces)
        proj_data = proj_data.transpose()
        w = np.array([np.dot(proj_data, i) for i in np.array(normalized_faces)])