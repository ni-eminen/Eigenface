# Test document - Eigenfaces

## Unit testing
### Coverage report
[![codecov](https://codecov.io/github/ni-eminen/eigenface/branch/master/graph/badge.svg?token=OE1J0JCOY3)](https://codecov.io/github/ni-eminen/eigenface)
![GHA workflow badge](https://github.com/ni-eminen/eigenface/workflows/CI/badge.svg)

### What was tested
The tests concentrate on validating most of the helper functions. The main algorithm isn't itself being tested, as the vast majority of operation are being done with numpy.

### Vector operations
All the handmade vector operations are tested to ensure correct results on common operations.

### Image processing
Image processing is being use for larger images. This includes cropping and modifying the color composition of the dataset images, if they are larger than 64x64.

### Distances
Common machine learning distance measures such as the hamming, euclidean and cityblock distances are being tested. This program implements handmade distance calculations between matrices and vectors.

### KNN predictions
In an attempt to improve results a k-nearest neighbour algorithm was implemented. It's simple use case is tested as it is an integral part of the algorithm.

## Running the tests

Install the dependencies for the project
    
    poetry install

Run pytest
    
    pytest

To generate an HTML document of the test coverage, run

    coverage html

and open the generated index.html file in your browser.