1. How to Compile and Run Code

knn.py
_____________

- In the command line terminal navigate to the directory where the files exists
	Run knn.py as follows:

	$ python knn.py {k} {distance}

e.g python knn.py 3 euclidean
e.g python knn.py 5 manhattan

knn-cv.py
_____________

- In the command line terminal navigate to the directory where the files exists
	Run knn-cv.py as follows:

	$ python knn-cv.py {k} {distance}

e.g python knn-cv.py 3 euclidean
e.g python knn-cv.py 5 manhattan


2. Printing out confusion matrix

- I implemented a method that prints out the confusion matrix but the columns and 
	rows are not labeled. However they follow the following ocnvention.

				   Predicted 
                 1     2     7
        	1 100.0   0.0   0.0
Actual      2  10.0  85.0   5.0
            7   5.0   0.0  95.0

Note that for tie breaking I choose the first element with the highest votes which is basically a random way of dealing with tie breaking.