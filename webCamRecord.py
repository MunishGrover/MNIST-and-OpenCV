import cv2
from collections import deque
import numpy as np
from keras.models import load_model

model = load_model('cnn-opencv.h5')

def main():
    cap = cv2.VideoCapture(0)
    Lower_green = np.array([110,50,50])
    Upper_green = np.array([130,255,255])
    pts = deque(maxlen=512)
    blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
    digit = np.zeros((200, 200, 3), dtype=np.uint8)
    finalString = ""
    while (cap.isOpened()):

        ret, img = cap.read()
        img = cv2.flip(img, 1)
        imagehsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(imagehsv, Lower_green, Upper_green)
        blur = cv2.medianBlur(mask, 15)
        blur = cv2.GaussianBlur(blur, (5, 5), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
        center = None

        if len(cnts) >= 1:
            cnt = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(cnt) > 200:
                ((x, y), radius) = cv2.minEnclosingCircle(cnt)
                cv2.circle(img, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.circle(img, center, 5, (0, 0, 255), -1)
                M = cv2.moments(cnt)
                center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
                pts.appendleft(center)
                for i in range(1, len(pts)):
                    if pts[i - 1] is None or pts[i] is None:
                        continue
                    cv2.line(blackboard, pts[i - 1], pts[i], (255, 255, 255), 8)
                    cv2.line(img, pts[i - 1], pts[i], (0, 0, 255), 5)
        elif len(cnts) == 0:
            if len(pts) != []:
                blackboard_gray = cv2.cvtColor(blackboard, cv2.COLOR_BGR2GRAY)
                blur1 = cv2.medianBlur(blackboard_gray, 15)
                blur1 = cv2.GaussianBlur(blur1, (5, 5), 0)
                thresh1 = cv2.threshold(blur1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                blackboard_cnts = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
                if len(blackboard_cnts) >= 1:
                    cnt = max(blackboard_cnts, key=cv2.contourArea)

                    if cv2.contourArea(cnt) > 2000:
                        x, y, w, h = cv2.boundingRect(cnt)
                        digit = blackboard_gray[y:y + h, x:x + w]

                        newimg = cv2.resize(digit, (28, 28))
                        newimg = np.array(newimg)

                        newimg = newimg.flatten()

                        newimg = newimg.reshape(newimg.shape[0], 1)

                        newimg = newimg.reshape(1, 28, 28, 1)
                        pred = model.predict(newimg)

                        print("Result: ")
                        finalString = pred.argmax()
                        print(pred.argmax())
            pts = deque(maxlen=512)
            blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
        x, y, w, h = 220, 100, 300, 300
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, str(finalString), (10, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 2)
        cv2.imshow("Frame", img)


        if cv2.waitKey(1) & 0xFF == ord("q"):
            pred = model.predict(newimg)
            break


main()