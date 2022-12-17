# Program structure
Eigenfaces program structure is very simple, with the main algorithm in a [Jupyter notebook](https://jupyter.org/) located in [eigenfaces.ipynb](https://github.com/ni-eminen/eigenface/blob/master/src/eigenfaces.ipynb). A .py file has been generated off the notebook and is located here [eigenfaces.py](https://github.com/ni-eminen/eigenface/blob/master/src/eigenfaces.py).

The algorithm uses some helper functions and methods that are located in [helpers.py](https://github.com/ni-eminen/eigenface/blob/master/src/helpers.py). For functions that need contextual information, a class named [EigenfaceHelpers](https://github.com/ni-eminen/eigenface/blob/44b492de480c874266d4333b446b2bf9aec4646f/src/helpers.py#L10) has been created. Stand-alone functions are located in the same file.

# Time complexity
Time complexity of the training is O(n³). The most computationally demanding task of the eigenfaces algorithm is to evaluate the eigenvalues and eigenvectors, which bumps the time complexity to the third power.

# Performance
The performance of the different distance methods and the K-nearest neighbour method is linear. The training phase is not affected by the preferred prediction evaluation method.

# What could be improved
There are some things left to be improved. For example, the predict function could be implemented slightly more performantly and with some more clarity in terms of readability.

# Sources
[M.üge Çarıkçı, Figen Özen, A Face Recognition System Based on Eigenfaces Method](https://www.sciencedirect.com/science/article/pii/S2212017312000242?ref=pdf_download&fr=RR-2&rr=764c913e48f4376d)