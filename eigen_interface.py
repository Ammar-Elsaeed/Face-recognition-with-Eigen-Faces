from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QLabel
from PyQt5.QtGui import QIcon, QPixmap
import sys
from PyQt5.uic import loadUi , loadUiType
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QDialog, QFileDialog 
from PyQt5.QtGui import QIcon, QPixmap , QImage
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

# mainwindow = loadUi("Eigen_Faces.ui")
MAIN_WINDOW,_=loadUiType("Eigen_Faces.ui")
class MainWindow(QtWidgets.QMainWindow,MAIN_WINDOW):
    def __init__(self):
        super(MainWindow, self).__init__()
        # self.ui = Ui_MainWindow()
        self.setupUi(self)
        self.TrainButton.clicked.connect(self.train)
        self.Test_Button.clicked.connect(self.test)
        self.Detect_Faces_Button.clicked.connect(self.detect_faces)
        self.actionOpen_File.triggered.connect(self.display_img)
        self.actionClose_File.triggered.connect(self.hide_img)
        self.actionOpen_DataSet.triggered.connect(self.load_dataset)
        # self.button.clicked.connect(self.addimage)

    def display_img(self):
        self.hide_img()
        fileName, _ = QFileDialog.getOpenFileName(self,"Open an image", "D:\Folders\forth year\Term-2\CV\assignment-5-cv-2022-sbe-404-team_02")
        qpixmap = QPixmap(fileName)
        qpixmap = qpixmap.scaled(self.input_img.width(),self.input_img.height())
        self.input_img.setPixmap(qpixmap)
        qpixmap = QPixmap()
        self.output_img.setPixmap(qpixmap)
        return
    def train(self):
        self.hide_img()
        t0 = time.time()
        qpixmap = QPixmap()
        self.output_img.setPixmap(qpixmap)
        # split data to training and testing
        x,x_test , y , y_test = train_test_split(self.zero_mean_data,self.target,stratify=self.target,shuffle=True,test_size=0.2, random_state=42)
        x= np.transpose(x)
        x_test = np.transpose(x_test)
        # get the Principle components, which are the eigen vectors of the Covariance matrix
        
        cov_mat = np.dot(np.transpose(x),x)
        
        
        eig_values , eig_vectors = np.linalg.eig(cov_mat)
        eig_vectors = np.dot(x,eig_vectors)
        
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
        selected_val , self.selected_pc = eig_values[skip:n_components],eig_vectors[:,skip:n_components] # select up to k components 
        # not: skip and var_to_keep values were determined empirically to give good results.
        display_vector = self.selected_pc[:,0]
        minimum_value = np.min(display_vector)
        display_vector = (display_vector+abs(minimum_value))
        display_vector = (display_vector/np.max(display_vector) * 255)
        display_vector = display_vector.reshape(self.myheight,self.mywidth)
        
        cv2.imwrite("displayed eigenvector.jpg", display_vector)
        displayed_vector_from_image =  cv2.imread("displayed eigenvector.jpg",0)
        qImg = QImage(displayed_vector_from_image.data, self.myheight, self.mywidth,QImage.Format_Grayscale8)
        self.displayed_vector_pixmap = QtGui.QPixmap(qImg)
        self.displayed_vector_pixmap = self.displayed_vector_pixmap.scaled(self.input_img.width(),self.input_img.height())
        self.output_img.setPixmap(self.displayed_vector_pixmap)
        projected_train = np.transpose(np.dot(np.transpose(self.selected_pc),x))
        projected_test = np.transpose(np.dot(np.transpose(self.selected_pc),x_test))
        self.clf = SVC().fit(projected_train,y)
        y_pred = self.clf.predict(projected_test)
        print(classification_report(y_test, y_pred))

        clf = SVC()
        visualizer = ROCAUC(clf,per_class=False)

        visualizer.fit(projected_train, y)        # Fit the training data to the visualizer
        visualizer.score(projected_test, y_test)        # Evaluate the model on the test data
        visualizer.show()
        print(time.time()-t0)
        return
    def test(self):
        self.hide_img()
        fileName, _ = QFileDialog.getOpenFileName(self,"Open a test image", "D:\Folders\forth year\Term-2\CV\assignment-5-cv-2022-sbe-404-team_02")
        qpixmap = QPixmap(fileName)
        qpixmap = qpixmap.scaled(self.input_img.width(),self.input_img.height())
        self.input_img.setPixmap(qpixmap)
        qpixmap = QPixmap()
        self.output_img.setPixmap(qpixmap)
        image = cv2.imread(fileName,-1)
        image = np.asarray(image.flatten())
        
        projected_test_image = np.transpose(np.dot(np.transpose(self.selected_pc),image))
        projected_test_image = np.transpose(projected_test_image.reshape(-1, 1))
        prediction = self.clf.predict(projected_test_image)
        print(prediction)
        return
    def detect_faces(self):
        self.hide_img()
        fileName, _ = QFileDialog.getOpenFileName(self,"Open an image", "D:\Folders\forth year\Term-2\CV\assignment-5-cv-2022-sbe-404-team_02")
        qpixmap = QPixmap(fileName)
        qpixmap = qpixmap.scaled(self.input_img.width(),self.input_img.height())
        self.input_img.setPixmap(qpixmap)
        cascPath = "haarcascade_frontalface_default.xml"
        # Create the haar cascade
        faceCascade = cv2.CascadeClassifier(cascPath)

        # Read the image
        image = cv2.imread(fileName)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
            # flags = cv2.CV_HAAR_SCALE_IMAGE
        )

        print("Found {0} faces!".format(len(faces)))

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        height, width, channel = image.shape
        bytesPerLine = 3 * width
        qImg = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
        self.pix = QtGui.QPixmap(qImg)
        self.pix = self.pix.scaled(self.output_img.width(),self.output_img.height())
        self.output_img.setPixmap(self.pix)
        return
    def hide_img(self):
        qpixmap = QPixmap()
        self.output_img.setPixmap(qpixmap)
        self.input_img.setPixmap(qpixmap)
        return    
    def load_dataset(self):
        self.hide_img()
        fileName = QFileDialog.getExistingDirectory(self,"Open a dataset", "D:\Folders\forth year\Term-2\CV\assignment-5-cv-2022-sbe-404-team_02")
        data =[]
        target = []
        for person in range(1,40):
            if (person == 14):
                pass
            else:
                try:
                    path =  fileName+"\yaleB0{}".format(person)
                    files = os.listdir(path)
                except:
                    path = fileName+"\yaleB{}".format(person)
                    files =  os.listdir(path)
                finally:
                    for file in  files:
                        image = cv2.imread(path+'\{}'.format(file),-1)
                        self.mywidth , self.myheight = image.shape
                        image = image.flatten()
                        data.append(image)
                        target.append(person)
        qpixmap = QPixmap(".\CroppedYale\yaleB01\yaleB01_P00A+000E+00.pgm")
        qpixmap = qpixmap.scaled(self.input_img.width(),self.input_img.height())
        self.input_img.setPixmap(qpixmap)
        self.target = np.asarray(target)
        data = np.asarray(data)

        # compute zero mean data
        self.zero_mean_data = data - np.mean(data,axis=0)
        return 

# main
def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()