import cv2

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4,720)

img_background = cv2.imread("Resources/background.png")

while True:
    success, img = cap.read()
    cv2.imshow("face attendance", img)
    cv2.imshow("img_background", img_background)
    cv2.waitKey(1)