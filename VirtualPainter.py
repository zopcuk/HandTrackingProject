import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm

#################################
wCam, hCam = 1280, 720
brushThickness = 15
eraserThickness = 75
xp, yp = 0, 0
#################################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

folderPath = "Header"
myList = os.listdir(folderPath)
overlayList = []
for imgPath in myList:
    image = cv2.imread(f'{folderPath}/{imgPath}')
    overlayList.append(image)
header = overlayList[0]
drawColor = (127, 0, 255)
imgCanvas = np.zeros((720, 1280, 3), np.uint8)
detector = htm.handDetector(detectionCon=0.5)

while True:
    # 1.import image
    success, img = cap.read()
    img = cv2.flip(img, 1)
    # 2.Find Hand Landmarks
    img = detector.findHands(img)
    lmList, _ = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        # tip of index and middle finger
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # 3.Check which fingers are up
        fingers = detector.fingersUp()

        # 4.If Selection mode - Two fingers are up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            # Checking for the click
            if y1 < 125:

                if 250 < x1 < 450:
                    header = overlayList[0]
                    drawColor = (127, 0, 255)
                elif 550 < x1 < 750:
                    header = overlayList[1]
                    drawColor = (255, 0, 0)
                elif 800 < x1 < 950:
                    header = overlayList[2]
                    drawColor = (0, 255, 0)
                elif 1050 < x1 < 1200:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)
                elif x1 < 200:
                    imgCanvas = np.zeros((720, 1280, 3), np.uint8)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)
        # 5.If Drawing Mode - Index finger is up
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
            xp, yp = x1, y1
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)
    # cTime = time.time()
    # fps = 1 / (cTime - pTime)
    # pTime = cTime
    # cv2.putText(img, f'FPS: {int(fps)}', (1100, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    # Setting the header image
    img[0:125, 0:1280] = header
    # img = cv2.addWeighted(img,0.5,imgCanvas,0.5,0)
    cv2.imshow("Image", img)
    # cv2.imshow("Canvas", imgCanvas)
    cv2.waitKey(1)
