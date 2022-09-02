import cv2
import time
import os
import HandTrackingModule as htm

#################################
wCam, hCam = 1280, 720
#################################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

folderPath = "FingerImages"
myList = os.listdir(folderPath)
overlayList = []
for imgPath in myList:
    image = cv2.imread(f'{folderPath}/{imgPath}')
    overlayList.append(image)

detector = htm.handDetector(detectionCon=0.75)
tipIds = [4, 8, 12, 16, 20]
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, _ = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        fingers = []

        # Thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # 4 Fingers
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        totalFingers = fingers.count(1)
        h, w, c = overlayList[totalFingers - 1].shape
        img[0:200, 0:200] = overlayList[totalFingers - 1]

        cv2.rectangle(img, (20, 255), (170, 425), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (45, 400), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (1100, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
