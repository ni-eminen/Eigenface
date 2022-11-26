# Week 2 - Eigenfaces

## Progress
- Tested all the helper functions, although this week I had to render them useless in order to gain performance. It made no sense to do them by hand in the first place, but I treated it as more of a learning experiment.
- Changed the dataset to use the popular olivette dataset because the original dataset seemed to give very bad results. Basically the algorithm doesn't seem to work unless your images are a perfect fit for it.
- Completed all the phases of the algorithm. Learned a lot about vector and matrix operations such as norms, dot products etc. that were needed to create the eigenfaces and the eigenspace.
- Learned to use Sklearn library to display results and split the training data
- I moved the whole project to the ipynb file, because it is a lot more convenient for this kind of project

## Key takeaways:
- Sklearn, linear algebra operations, how eigenfaces literally are formed
- What kind of dataset is suitable for eigenfaces and what kind of images you need of people to identify them
- Numpy functionality

## What I've found difficult
- At first I spent countless hours figuring out why my algorithm is not working at all. After that I tried changing the dataset up, and the algorithm started working. Basically the only images that the earlier dataset was able to recognize were images that were a part of that dataset.


## What's next
- Next I will compare different kinds of ways to find the correct face (alternatives to euclidian distance) and see if basing the prediction on multiple matches instead of only the closest match would improve the precision from .85~.