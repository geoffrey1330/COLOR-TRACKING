import numpy as np
import time
import cv2
import imutils

min_color = np.array([0, 170, 170], dtype = "uint8")
max_color = np.array([50, 255, 255], dtype = "uint8")


camera = cv2.VideoCapture("video.mp4")

while True:
	(grabbed, frame) = camera.read()
	frame=imutils.resize(frame,height=500)

	
	if not grabbed:
		break

	vid = cv2.inRange(frame, min_color, max_color)
	vid = cv2.GaussianBlur(vid, (5, 5), 0)

	(contours,_) = cv2.findContours(vid, cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)

	
	if len(contours) > 0:
		
		contours = sorted(contours, key = cv2.contourArea, reverse = True)[0]
		
		rect = np.int32(cv2.boxPoints(cv2.minAreaRect(contours)))
		cv2.drawContours(frame, [rect], -1, (0, 255, 0), 2)

	
	cv2.imshow("Tracking", frame)
	
	time.sleep(0.025)

	
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

camera.release()
cv2.destroyAllWindows()