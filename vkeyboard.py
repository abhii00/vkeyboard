import cv2
import mediapipe as mp
import numpy as np
import pynput as pyn
import string

#TODO add keyboard presses rather than hold
#TODO add finger counter to check whether 1 or multiple fingers are up
#TODO add whole keyboard layout
#TODO consider hardware 

class Detector:
    def __init__(self):
        #video
        self.video = cv2.VideoCapture(0)

        #detector
        self.detector = mp.solutions.hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        #styles
        self.styles = {
            'landmarks': mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
            'handcons': mp.solutions.drawing_styles.get_default_hand_connections_style()
        }

        self.breakLoop = False

        self.indexes_coords = []
        self.boxes_coords = []

    def readImage(self):
        success, self.image = self.video.read()
        self.height, self.width, self.c = self.image.shape

    def detectHands(self):
        self.image.flags.writeable = False
        self.results = self.detector.process(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        self.image.flags.writeable = True

    def calcResults(self):
        if self.results.multi_hand_landmarks:
            self.indexes_coords = []
            self.boxes_coords = []

            for hand_landmarks in self.results.multi_hand_landmarks:
                #index
                normal_coords = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
                self.indexes_coords.append([normal_coords.x * self.width, normal_coords.y * self.height])

                #boxes
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.append((int(landmark.x * self.width), int(landmark.y * self.height), (landmark.z * self.width)))

                x_coordinates = np.array(landmarks)[:,0]
                y_coordinates = np.array(landmarks)[:,1]
            
                pad = 10
                x1 = int(np.min(x_coordinates) - pad)
                y1 = int(np.min(y_coordinates) - pad)
                x2 = int(np.max(x_coordinates) + pad)
                y2 = int(np.max(y_coordinates) + pad)

                self.boxes_coords.append([x1, y1, x2, y2])

    def drawResults(self, hands, boxes):
        if self.results.multi_hand_landmarks:
            if hands:
                for hand_landmarks in self.results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(self.image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS, self.styles['landmarks'], self.styles['handcons'])

            if boxes:
                for box_coords in self.boxes_coords:
                    cv2.rectangle(self.image, (box_coords[0], box_coords[1]), (box_coords[2], box_coords[3]), (155, 0, 255), 3, cv2.LINE_8)
        
    def showImage(self):
        cv2.imshow('Hand Detector', cv2.flip(self.image, 1))

    def checkBreak(self, break_key):
        if cv2.waitKey(1) & 0xFF == ord(break_key): self.breakLoop = True

class Keyboard:
    def __init__(self):
        self.no_keys = [2,13]
        self.key_size = 25
        self.characters = list(string.ascii_lowercase)
        self.keys = []

        for i in range(0, self.no_keys[0]):
            for j in range(0, self.no_keys[1]):
                self.keys.append(Key(self.characters[i*self.no_keys[1]+j], j*self.key_size, i*self.key_size, (j+1)*self.key_size, (i+1)*self.key_size))

class Key:
    def __init__(self, char, x1, y1, x2, y2, col_pressed = (255,255,255), col_default=(0,0,0)):
        self.char = char
        self.coords = [x1, y1, x2, y2]

        self.col_pressed = col_pressed
        self.col_default = col_default
        self.intersect = False

    def draw(self, image):
        cv2.rectangle(image, (self.coords[0], self.coords[1]), (self.coords[2], self.coords[3]), self.col_pressed if self.intersect else self.col_default, 3, cv2.LINE_8)

    def checkIntersectBox(self, box_coords):
        if np.any(np.intersect1d(range(box_coords[0],box_coords[2]), range(self.coords[0], self.coords[2]))) and np.any(np.intersect1d(range(box_coords[1],box_coords[3]), range(self.coords[1], self.coords[3]))):
            return True
        else:
            return False

    def checkContainPoint(self, point_coords):
        if (point_coords[0] > self.coords[0] and point_coords[0] < self.coords[2]) and (point_coords[1] > self.coords[1] and point_coords[1] < self.coords[3]):
            return True
        else:
            return False

    def checkPressed(self, points_coords):
        intersections = []
        for point_coords in points_coords:
            intersections.append(self.checkContainPoint(point_coords))
        if np.any(intersections): self.intersect = True
        else: self.intersect = False

        if self.intersect:
            cont = pyn.keyboard.Controller()
            cont.press(self.char)
            cont.release(self.char)        

if __name__ == '__main__':
    #initialise detector and keyboard
    det = Detector()
    keyb = Keyboard()

    #main loop
    while not det.breakLoop:
        #detect hands
        det.readImage()
        det.detectHands()
        det.calcResults()
        det.drawResults(hands=True, boxes=True)

        #check if keys are pressed and draw keys
        for key in keyb.keys: 
            key.checkPressed(det.indexes_coords)
            key.draw(det.image)

        #display image and misc.
        det.showImage()
        det.checkBreak(break_key='q')