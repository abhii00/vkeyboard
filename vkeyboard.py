import cv2
import mediapipe as mp

class Detector:
    def __init__(self):
        self.video = cv2.VideoCapture(0) #video capture
        self.detector = mp.solutions.hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) #detector

        self.styles = {
            'landmarks': mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
            'handcons': mp.solutions.drawing_styles.get_default_hand_connections_style()
        }

        self.image = 0
        self.results = 0
        self.breakLoop = False

    def readImage(self):
        success, self.image = self.video.read()

    def detectHands(self):
        #make image unwriteable to improve performance, convert to correct colorspace
        self.image.flags.writeable = False
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        #apply hand detector to image
        self.results = self.detector.process(self.image)

        #make image writeable to draw hands
        self.image.flags.writeable = True
        self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(self.image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS, self.styles['landmarks'], self.styles['handcons'])

    def showImage(self):
        cv2.imshow('Hand Detector', cv2.flip(self.image, 1))

    def checkBreak(self, break_key):
        if cv2.waitKey(1) & 0xFF == ord(break_key): self.breakLoop = True



if __name__ == '__main__':
    test = Detector()

    while not test.breakLoop:
        test.readImage()
        test.detectHands()
        test.showImage()
        test.checkBreak('q')