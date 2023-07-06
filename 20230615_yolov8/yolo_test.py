import cv2
import IPython
from PIL import Image
from io import BytesIO
from ultralytics import YOLO

model_size = ["n", "s", "m", "l", "x"][0]
model = YOLO(f"./models/yolov8{model_size}.pt")

def show_camera(device=0, fmt="jpeg"):

    cap = cv2.VideoCapture(device)
    assert cap.isOpened(), 'Could not open video device'

    try:
        while(True):
            ret, frame = cap.read()

            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = model(
                                source    = frame,
                                conf      = 0.1,
                                iou       = 0.001,
                                save      = False,
                                max_det   = 300,
                                augment   = True,
                                # classes   = [74], # [1, 2, 3],
                                classes   = None, # [1, 2, 3],
                            )


                frame = result[0].plot()[:,:,::-1]

                # f = BytesIO()
                # Image.fromarray(frame).save(f, fmt)
                # IPython.display.display(IPython.display.Image(data=f.getvalue()))
                
                # IPython.display.clear_output(wait=True)

                
                cv2.imshow('frame',frame)
                cv2.waitKey(1)

    except KeyboardInterrupt:
        cap.release()
        print('Stream stopped')

    
        cv2.destroyAllWindows()

    return None

show_camera()

