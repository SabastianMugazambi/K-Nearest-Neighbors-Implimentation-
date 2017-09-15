
import numpy as np
import matplotlib.pyplot as plt
import pylab
import sys
import math


def euclidean(test_image,train_data,train_labels,k):
	
	cal_dis = list()
	for y in range(0, np.shape(train_data)[0]):
		cal_dis.append(math.sqrt(np.sum(np.subtract(test_image,train_data[y])**2)))

	voters = np.argsort(cal_dis,axis=-1,kind="mergesort",order=None)
	votes = list()

	for i in range(0,k):
		votes.append(train_labels[voters[i]])

	#handle ties please
	counts = np.bincount(votes)
	pred_label = np.argmax(counts)
	return np.argmax(counts)

def manhattan(test_image,train_data,train_labels,k):
	
	cal_dis = list()
	for y in range(0, np.shape(train_data)[0]):
		cal_dis.append(np.sum(np.absolute(np.subtract(test_image,train_data[y]))))

	voters = np.argsort(cal_dis,axis=-1,kind="mergesort",order=None)
	votes = list()

	for i in range(0,k):
		votes.append(train_labels[voters[i]])

	#handle ties please
	counts = np.bincount(votes)
	pred_label = np.argmax(counts)
	return np.argmax(counts)

def create_conf_matrix(actual, predicted, labels):

	cm = np.zeros((len(labels), len(labels)))
	for a, p in zip(actual, predicted):
		if a==1:
			a=0
		if a==2:
			a=1
		if a==7:
			a=2

		if p==1:
			p=0
		if p==2:
			p=1
		if p==7:
			p=2
		cm[a][p] += 1
	return cm

def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels]+[5]) # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print "    " + empty_cell,
    for label in labels: 
        print "%{0}s".format(columnwidth) % label,
    print
    # Print rows
    for i, label1 in enumerate(labels):
        print "    %{0}s".format(columnwidth) % label1,
        for j in range(len(labels)): 
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print cell,
        print

def getAccuracy(test_data,test_labels,train_data,train_labels,k,mode):

	if mode == "euclidean":

		pred_labels = list()
		for x in range (0, np.shape(test_data)[0]):
			pred_labels.append(euclidean(test_data[x],train_data,train_labels,k))

		return pred_labels

	else:

		pred_labels = list()
		for x in range (0, np.shape(test_data)[0]):
			pred_labels.append(manhattan(test_data[x],train_data,train_labels,k))

		return pred_labels



def main():
    train_data = np.loadtxt("train_data.txt",dtype=int,delimiter=",")
    train_labels = np.loadtxt("train_labels.txt",dtype=int,delimiter="\n")

    if len(sys.argv) < 3:
    	print("Not enough args")
    	exit()
    else:

		k = int(sys.argv[1])
		mode = sys.argv[2]
		print "k : ",k
		print "Distance : ",mode

		perm_index = np.random.permutation(np.shape(train_data)[0])

		perm_data = train_data[perm_index]
		perm_labels = train_labels[perm_index]
		cv_data = np.split(perm_data, 5, axis=0)
		cv_labels = np.split(perm_labels, 5, axis=0)

		accuracies = list()
		confusion = np.zeros((3,3))

		for i in range (0,len(cv_data)):
			print "Working on CV Fold : ", i, "..."
			test_data = cv_data[i]
			test_labels = cv_labels[i]

			new_cv_data = np.delete(cv_data, i, 0)
			new_cv_labels = np.delete(cv_labels,i, 0)

			train_data = np.concatenate((new_cv_data[0], new_cv_data[1],new_cv_data[2],new_cv_data[3]), axis=0)
			train_labels = np.concatenate((new_cv_labels[0], new_cv_labels[1],new_cv_labels[2],new_cv_labels[3]), axis=0)

			pred_labels = getAccuracy(test_data,test_labels,train_data,train_labels,k,mode)

			C = create_conf_matrix(test_labels, pred_labels,("1","2","7"))
			confusion = np.add(confusion,C)
			accuracies.append((test_labels == pred_labels).sum() / float(len(test_labels)))

			#print "Sub - Confusion Matrix:"
			#print_cm(C,("1","2","7"))

		print "CV Grand Confusion Matrix not Averaged"
		print_cm(confusion,("1","2","7"))
		print "CV Combined Accuracy (Averaged)", np.mean(accuracies)






if __name__ == '__main__':
    main()