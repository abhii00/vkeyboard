import cv2
import mediapipe as mp

def setupDetector():
    cap = cv2.VideoCapture(0) #video capture
    det = mp.solutions.hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) #detector

    ls_style = mp.solutions.drawing_styles.get_default_hand_landmarks_style() #landmarks style
    hc_style = mp.solutions.drawing_styles.get_default_hand_connections_style() #connections style
    styles = {'landmarks': ls_style, 'handcons': hc_style}

    return cap, det, styles

def detectHands(cap, det, break_key, styles):
    while True:
        success, image = cap.read()

        if not success:
            print('Ignoring Empty Camera Frame.')
            continue

        #make image unwriteable to improve performance, convert to correct colorspace
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #apply hand detector to image
        results = det.process(image)

        #make image writeable to draw hands
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS, styles['landmarks'], styles['handcons'])
        
        #show image
        cv2.imshow('Hand Detector', cv2.flip(image, 1))

        #exit
        if cv2.waitKey(1) & 0xFF == ord(break_key):
            break

    cap.release() 

if __name__ == '__main__':
    cap, det, styles = setupDetector()
    detectHands(cap, det, 'q', styles)
