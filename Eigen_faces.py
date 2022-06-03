import cv2
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import classification_report , roc_curve , roc_auc_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from yellowbrick import ROCAUC
# reading the data
data = []
target = []
for person in range(1,40):
    if (person == 14):
        pass
    else:
        try:
            path =  ".\CroppedYale\yaleB0{}".format(person)
            files = os.listdir(path)
        except:
            path = ".\CroppedYale\yaleB{}".format(person)
            files =  os.listdir(path)
        finally:
            for file in  files:
                image = cv2.imread(path+'\{}'.format(file),-1)
                width ,height = image.shape
                image = image.flatten()
                data.append(image)
                target.append(person)
target = np.asarray(target)
data = np.asarray(data)

# compute zero mean data
zero_mean_data = data - np.mean(data,axis=0)

# split data to training and testing
x,x_test , y , y_test = train_test_split(zero_mean_data,target,stratify=target,shuffle=True,test_size=0.2, random_state=42)
x= np.transpose(x)
x_test = np.transpose(x_test)
# get the Principle components, which are the eigen vectors of the Covariance matrix
t0 = time.time()
cov_mat = np.dot(np.transpose(x),x)
print(time.time()-t0)
t0 = time.time()
eig_values , eig_vectors = np.linalg.eig(cov_mat)
eig_vectors = np.dot(x,eig_vectors)
print(time.time()-t0)
# select the variance to keep
var = 0
var_to_keep = 0.99
n_components = 1
sum_value = np.sum(eig_values)
while(var<var_to_keep):
    cumsum = np.cumsum(eig_values[:n_components])[-1] 
    var = cumsum/sum_value
    n_components = n_components+1
skip = 100 # skip the first 100 vectors, since they tend to be generic and do not help us to distinguish between classes 
selected_val , selected_pc = eig_values[skip:n_components],eig_vectors[:,skip:n_components] # select up to k components 
# not: skip and var_to_keep values were determined empirically to give good results.
display_vector = np.transpose(selected_pc[:,0])
display_vector = display_vector.reshape(width,height)

# vis2 = cv2.CreateMat(width, height, cv2.CV_32FC3)
# vis0 = cv2.fromarray(display_vector)
projected_train = np.transpose(np.dot(np.transpose(selected_pc),x))
projected_test = np.transpose(np.dot(np.transpose(selected_pc),x_test))
clf = SVC().fit(projected_train,y)
y_pred = clf.predict(projected_test)
print(classification_report(y_test, y_pred))

clf = SVC()
visualizer = ROCAUC(clf,per_class=False)

visualizer.fit(projected_train, y)        # Fit the training data to the visualizer
visualizer.score(projected_test, y_test)        # Evaluate the model on the test data
visualizer.show()
