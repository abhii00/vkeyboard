import cv2
import mediapipe as mp
import numpy as np

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

    def readImage(self):
        success, self.image = self.video.read()
        self.height, self.width, self.c = self.image.shape

    def drawResults(self, hands, boxes):
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if hands:
                    mp.solutions.drawing_utils.draw_landmarks(self.image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS, self.styles['landmarks'], self.styles['handcons'])

                if boxes:
                    landmarks = []
                    for landmark in hand_landmarks.landmark:
                        landmarks.append((int(landmark.x * self.width), int(landmark.y * self.height), (landmark.z * self.width)))

                    x_coordinates = np.array(landmarks)[:,0]
                    y_coordinates = np.array(landmarks)[:,1]
                
                    pad = 10
                    x1  = int(np.min(x_coordinates) - pad)
                    y1  = int(np.min(y_coordinates) - pad)
                    x2  = int(np.max(x_coordinates) + pad)
                    y2  = int(np.max(y_coordinates) + pad)

                    cv2.rectangle(self.image, (x1, y1), (x2, y2), (155, 0, 255), 3, cv2.LINE_8)

    def detectHands(self):
        self.image.flags.writeable = False
        self.results = self.detector.process(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        self.image.flags.writeable = True
        
    def showImage(self):
        cv2.imshow('Hand Detector', cv2.flip(self.image, 1))

    def checkBreak(self, break_key):
        if cv2.waitKey(1) & 0xFF == ord(break_key): self.breakLoop = True

if __name__ == '__main__':
    test = Detector()

    while not test.breakLoop:
        test.readImage()
        test.detectHands()
        test.drawResults(hands=True, boxes=True)
        test.showImage()
        test.checkBreak(break_key='q')