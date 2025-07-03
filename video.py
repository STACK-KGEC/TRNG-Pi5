from picamera2 import Picamera2
import cv2
import time

picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")

picam2.start()
time.sleep(1)

while True:
    frame = picam2.capture_array()
    frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
    cv2.imshow("Pi Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
picam2.close()
