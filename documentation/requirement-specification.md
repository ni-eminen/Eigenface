# Requirement specification - Eigenface face recognition

### Programming languages
## This project: 
1. Python

## Languages I can use for evaluating other people's work:
1. Python
1. Java
1. C#
1. Javascript/Typescript
1. C++
1. Haskell

### Algorithms and data structures required for this project:
Face recognition algorithm utilizing the Eigenfaces method
Principal Component Analysis

![eigenfaces process flow](https://github.com/ni-eminen/eigenface/blob/master/documentation/imgs/eigenfaces-algorithm.png)
<br/>source: [M.üge Çarıkçı, Figen Özen, A Face Recognition System Based on Eigenfaces Method](https://www.sciencedirect.com/science/article/pii/S2212017312000242?ref=pdf_download&fr=RR-2&rr=764c913e48f4376d)

### The problem
The problem of face recognition has been bubbling since the 1960's, and the eigenfaces method for face recognition was presented in 1991 by M. Turk and A. Pentland. The problem, specifically, is classifying people's faces based on training data of said people.

I chose this topic because it was an interesting application of PCA (principal component analysis) and a good introduction to supervised machine learning.

### Inputs
The program receives as an input an image of a face which it then classifies as one of the people from the training set, or as unknown when no match is found.

### Performance
Training the algorithm must work in exponential time, recognizing a new face must be in polynomial time.

Sources:
[1]: https://www.sciencedirect.com/science/article/pii/S2212017312000242?ref=pdf_download&fr=RR-2&rr=764c913e48f4376d
[2]: https://en.wikipedia.org/wiki/Eigenface#History
