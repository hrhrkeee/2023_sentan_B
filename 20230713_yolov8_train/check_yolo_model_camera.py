import cv2
import IPython
from PIL import Image
from io import BytesIO
from ultralytics import YOLO

model = YOLO(f"./output/yolov8/20230713-1353_yolov8n_sentan-B_ball_detection_yolo3/weights/best.pt")

cap = cv2.VideoCapture(0)
assert cap.isOpened(), 'Could not open video device'

try:
    while(True):
        ret, frame = cap.read()

        if ret:
            result = model(
                source    = frame,
                conf      = 0.4,
                iou       = 0.001,
                save      = False,
                max_det   = 300,
                augment   = True,
                classes   = None, # [1, 2, 3],
            )

            # f = BytesIO()
            # Image.fromarray(result[0].plot()).save(f, "jpeg")
            # IPython.display.display(IPython.display.Image(data=f.getvalue()))
            
            # IPython.display.clear_output(wait=True)
            
            cv2.imshow('frame',result[0].plot())
            cv2.waitKey(1)

except KeyboardInterrupt:
    cap.release()
    print('Stream stopped')

cap.release()
cv2.destroyAllWindows()
