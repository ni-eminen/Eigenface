# Week 6 - Eigenfaces

## Progress
- Made multi id evaluation possible. Now there is an option to set a sample size and a threshold for the closest matches. If there is more equivalent ID matches than the threshold in the set of closes matches, that ID will be predicted as the result.
- I also implemented a possibility to use different kinds of distance measures to determine the similarity of the unknown image with the eigenvectors. Hamming, Manhattan and Euclidean distances were implemented.

## Key takeaways:
- Multi id detection doesn't work reliably
- Manhattan distance is by far the most effective distance measure for (at the very least) this eigenfaces implementation.

## What I've found difficult
- Comprehending matrix operations


## What's next
- Statistical analysis of the different methods (distances and multi-id prediction)
- Testing all the helpers
- Proper test documentation