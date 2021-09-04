import numpy as np
import cv2 as cv
import pyvirtualcam

path = "./model/face.xml"
trollface = cv.imread("image/trollface.png")
cap = cv.VideoCapture(0)

face_cascade = cv.CascadeClassifier(path)


def draw_trollface(frame, x, y, w, h):

    trollface2 = cv.resize(trollface, (w, h))
    # for thresholding
    gray_trollface = cv.cvtColor(trollface2, cv.COLOR_BGR2GRAY)
    threshold = cv.threshold(gray_trollface, thresh=150,
                             maxval=255, type=cv.THRESH_BINARY)[1]
    # this draw a mask over the frame
    countours = cv.findContours(
        threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[0]

    cv.drawContours(trollface2, countours, 1, (0, 0, 0), -1)

    threshold = cv.threshold(gray_trollface, thresh=150,
                             maxval=255, type=cv.THRESH_BINARY_INV)[1]
    trollface2 = cv.bitwise_and(frame[y:y+h, x:x+w], trollface2)
    threshold = cv.cvtColor(threshold, cv.COLOR_GRAY2BGR)
    trollface2 = cv.bitwise_or(trollface2, threshold)

    return trollface2


frame = cap.read()[1]

with pyvirtualcam.Camera(width=frame.shape[1], height=frame.shape[0], fps=20) as cam:
    cam.send(frame)
    cam.sleep_until_next_frame()
    while True:
        frame = cap.read()[1]

        frame2 = cv.resize(frame, (0, 0), fx=0.5, fy=0.5)
        gray = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray, (5, 5), 0)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) <= 0:

            frame = cv.dilate(frame, None, iterations=20)
        for (x, y, w, h) in faces:
            x, y = (x*2, y*2)
            w, h = (w*2, h*2)

            frame[y:y+h, x:x+w] = draw_trollface(frame, x, y, w, h)
        frame=cv.cvtColor(frame,cv.COLOR_BGR2RGB)
        cam.send(frame)
        cam.sleep_until_next_frame()
        #cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv.destroyAllWindows()
