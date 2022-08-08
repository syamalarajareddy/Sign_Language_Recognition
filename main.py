import os          # for removal of files
import shutil      # for removal of directories
import sys          # for pyqt
from tkinter import *
from tkinter import filedialog  # for file export module

import cv2  # for the camera operations
import keyboard  # for pressing keys
import numpy as np  # processing on images
import pyttsx3  # for tts assistance
import win32con  # for removing title cv2 window and always on top
import win32gui
from PyQt5 import QtCore  # importing pyqt5 libraries
from PyQt5 import QtGui
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QMessageBox
from matplotlib import pyplot as plt  # for gesture viewer
from matplotlib.widgets import Button

import winGuiAuto

index = 0                # index used for gesture viewer
engine = pyttsx3.init()  # engine initialization for audio tts assistance


def nothing(x):
    pass


image_x, image_y = 64, 64    # image resolution

from keras.models import load_model

classifier = load_model('ASLModel.h5')  # loading the model


def fileSearch():
    """Searches each file ending with .png in SampleGestures directory so that custom gesture could be passed to predictor() function"""
    fileEntry = []
    for file in os.listdir("SampleGestures"):
        if file.endswith(".png"):
            fileEntry.append(file)
    return fileEntry


def load_images_from_folder(folder):
    """Searches each images in a specified directory"""
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images


def toggle_imagesfwd(event):
    """displays next images act as a gesuture viewer"""
    img = load_images_from_folder('TempGest/')
    global index

    index += 1

    try:
        if index < len(img):
            plt.axes()
            plt.imshow(img[index])
            plt.draw()
    except:
        pass


def toggle_imagesrev(event):
    """displays previous images act as a gesutre viewer"""
    img = load_images_from_folder('TempGest/')
    global index

    index -= 1

    try:
        if index < len(img) and index >= 0:
            plt.axes()
            plt.imshow(img[index])
            plt.draw()
    except:
        pass


def openimg():
    """displays predefined gesture images at right most window"""
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    image = cv2.imread('template.png')
    cv2.imshow("Image", image)
    cv2.setWindowProperty("Image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.resizeWindow("Image", 298, 430)
    cv2.moveWindow("Image", 1052, 214)


def removeFile():
    """Removes the temp.txt and tempgest directory if any stop button is pressed oor application is closed"""
    try:
        os.remove("temp.txt")
    except:
        pass
    try:
        shutil.rmtree("TempGest")
    except:
        pass


def clearfunc(cam):
    """shut downs the opened camera and calls removeFile() Func"""
    cam.release()
    cv2.destroyAllWindows()
    removeFile()


def clearfunc2(cam):
    """shut downs the opened camera"""
    cam.release()
    cv2.destroyAllWindows()



def controlTimer(self):
    # if timer is stopped
    self.timer.isActive()
    # create video capture
    self.cam = cv2.VideoCapture(0)
    # start timer
    self.timer.start(20)


def predictor():
    """ Depending on model loaded, prediction is made by checking array"""
    import numpy as np
    from keras.preprocessing import image
    test_image = image.load_img('1.png', target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = classifier.predict(test_image)
    gesname = ''
    fileEntry = fileSearch()
    for i in range(len(fileEntry)):
        image_to_compare = cv2.imread("./SampleGestures/" + fileEntry[i])
        original = cv2.imread("1.png")
        sift = cv2.xfeatures2d.SIFT_create()
        kp_1, desc_1 = sift.detectAndCompute(original, None)
        kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)

        index_params = dict(algorithm=0, trees=5)
        search_params = dict()
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(desc_1, desc_2, k=2)

        good_points = []
        ratio = 0.6
        for m, n in matches:
            if m.distance < ratio * n.distance:
                good_points.append(m)
        if (abs(len(good_points) + len(
                matches)) > 20):  # goodpoints and matcches sum from 1.png and customgestureimages is grater than 20
            gesname = fileEntry[i]
            gesname = gesname.replace('.png', '')
            if (gesname == 'sp'):  # sp is replaced with <space>
                gesname = ' '
            return gesname

    if result[0][0] == 1:
        return 'A'
    elif result[0][1] == 1:
        return 'B'
    elif result[0][2] == 1:
        return 'C'
    elif result[0][3] == 1:
        return 'D'
    elif result[0][4] == 1:
        return 'E'
    elif result[0][5] == 1:
        return 'F'
    elif result[0][6] == 1:
        return 'G'
    elif result[0][7] == 1:
        return 'H'
    elif result[0][8] == 1:
        return 'I'
    elif result[0][9] == 1:
        return 'J'
    elif result[0][10] == 1:
        return 'K'
    elif result[0][11] == 1:
        return 'L'
    elif result[0][12] == 1:
        return 'M'
    elif result[0][13] == 1:
        return 'N'
    elif result[0][14] == 1:
        return 'O'
    elif result[0][15] == 1:
        return 'P'
    elif result[0][16] == 1:
        return 'Q'
    elif result[0][17] == 1:
        return 'R'
    elif result[0][18] == 1:
        return 'S'
    elif result[0][19] == 1:
        return 'T'
    elif result[0][20] == 1:
        return 'U'
    elif result[0][21] == 1:
        return 'V'
    elif result[0][22] == 1:
        return 'W'
    elif result[0][23] == 1:
        return 'X'
    elif result[0][24] == 1:
        return 'Y'
    elif result[0][25] == 1:
        return 'Z'


def checkFile():
    """retrieve the content of temp.txt for export module """
    checkfile = os.path.isfile('temp.txt')
    if (checkfile == True):
        fr = open("temp.txt", "r")
        content = fr.read()
        fr.close()
    else:
        content = "No Content Available"
    return content


class Dashboard(QtWidgets.QMainWindow):
    def __init__(self):
        super(Dashboard, self).__init__()
        self.setWindowFlags(QtCore.Qt.WindowMinimizeButtonHint)
        cap = cv2.VideoCapture('')

        # Read until video is completed
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                # Capture frame-by-frame
                ret, frame = cap.read()
                cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
                cv2.imshow("mask", frame)
                cv2.setWindowProperty("mask", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.resizeWindow("mask", 720, 400)
                cv2.moveWindow("mask", 320, 220)

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

            else:
                break

        # When everything done, release
        cap.release()

        # Closes all the frames
        cv2.destroyAllWindows()
        self.setWindowIcon(QtGui.QIcon('icons/windowLogo.png'))
        self.title = 'Sign language Recognition'
        uic.loadUi('UI_Files/dash.ui', self)
        self.setWindowTitle(self.title)
        self.timer = QTimer()
        if (self.scan_sinlge.clicked.connect(self.scanSingle) == True):
            self.timer.timeout.connect(self.scanSingle)
        self.scan_sinlge.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.exit_button.clicked.connect(self.quitApplication)
        self._layout = self.layout()
        self.label_3 = QtWidgets.QLabel()
        movie = QtGui.QMovie("icons/dashAnimation.gif")
        self.label_3.setMovie(movie)
        self.label_3.setGeometry(600, 400, 780, 441)
        movie.start()
        self._layout.addWidget(self.label_3)
        self.setObjectName('Message_Window')

    def quitApplication(self):
        """shutsdown the GUI window along with removal of files"""
        userReply = QMessageBox.question(self, 'Quit Application', "Are you sure you want to quit this app?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if userReply == QMessageBox.Yes:
            removeFile()
            keyboard.press_and_release('alt+F4')

    def scanSingle(self):
        """Single gesture scanner """
        try:
            clearfunc(self.cam)
        except:
            pass
        uic.loadUi('UI_Files/scan_single.ui', self)
        self.setWindowTitle(self.title)
        if (self.scan_sinlge.clicked.connect(self.scanSingle)):
            controlTimer(self)
        self.pushButton_2.clicked.connect(lambda: clearfunc(self.cam))
        self.linkButton.clicked.connect(openimg)
        self.scan_sinlge.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        try:
            self.exit_button.clicked.connect(lambda: clearfunc(self.cam))
        except:
            pass
        self.exit_button.clicked.connect(self.quitApplication)
        img_text = ''
        while True:
            ret, frame = self.cam.read()
            frame = cv2.flip(frame, 1)
            try:
                frame = cv2.resize(frame, (321, 270))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img1 = cv2.rectangle(frame, (150, 50), (300, 200), (0, 255, 0), thickness=2, lineType=8, shift=0)
            except:
                keyboard.press_and_release('esc')

            height1, width1, channel1 = img1.shape
            step1 = channel1 * width1
            # create QImage from image
            qImg1 = QImage(img1.data, width1, height1, step1, QImage.Format_RGB888)
            # show image in img_label
            try:
                self.label_3.setPixmap(QPixmap.fromImage(qImg1))
                slider1 = self.trackbar.value()
            except:
                pass

            lower_blue = np.array([0, 0, 0])
            upper_blue = np.array([179, 255, slider1])

            imcrop = img1[52:198, 152:298]
            hsv = cv2.cvtColor(imcrop, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower_blue, upper_blue)

            cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
            cv2.imshow("mask", mask)
            cv2.setWindowProperty("mask", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.resizeWindow("mask", 118, 108)
            cv2.moveWindow("mask", 894, 271)

            hwnd = winGuiAuto.findTopWindow("mask")
            win32gui.SetWindowPos(hwnd, win32con.HWND_TOP, 0, 0, 0, 0,
                                  win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_NOACTIVATE)

            try:
                self.textBrowser.setText("\n\n\t" + str(img_text))
            except:
                pass

            img_name = "1.png"
            save_img = cv2.resize(mask, (image_x, image_y))
            cv2.imwrite(img_name, save_img)
            img_text = predictor()

            if cv2.waitKey(1) == 27:
                break

        self.cam.release()
        cv2.destroyAllWindows()
        
app = QtWidgets.QApplication([])
win = Dashboard()
win.show()
sys.exit(app.exec())
