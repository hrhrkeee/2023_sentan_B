import cv2, torch, torchvision, time
import numpy as np
import onnx
import onnxruntime


def box_label(
    img, box, line_width, label="", color=(128, 128, 128), txt_color=(255, 255, 255)
):
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(img, p1, p2, color, thickness=line_width, lineType=cv2.LINE_AA)
    if label:
        tf = max(3 - 1, 1)
        w, h = cv2.getTextSize(label, 0, fontScale=3 / 3, thickness=tf)[0]
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(img, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
            0,
            3 / 3,
            txt_color,
            thickness=tf,
            lineType=cv2.LINE_AA,
        )
    return img

def xywh2xyxy(x):
    y = x.clone()
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    agnostic=False,
    labels=(),
    max_det=300,
    nm=0,
):
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after

    t = time.time()
    mi = 5 + nc
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        x = x[xc[xi]]  # confidence
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        if not x.shape[0]:
            continue
        x[:, 5:] *= x[:, 4:5]
        box = xywh2xyxy(x[:, :4])
        mask = x[:, mi:]

        conf, j = x[:, 5:mi].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        n = x.shape[0]
        if not n:
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        c = x[:, 5:6] * (0 if agnostic else max_wh)
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        i = i[:max_det]

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break

    return output

def letterbox(
    im,
    new_shape=(640, 640),
    color=(114, 114, 114),
    scaleup=True,
    stride=32,
):
    shape = im.shape[:2]  # current shape [height, width]

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return im, ratio, (dw, dh)

def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    boxes[..., 0].clamp_(0, shape[1])  # x1
    boxes[..., 1].clamp_(0, shape[0])  # y1
    boxes[..., 2].clamp_(0, shape[1])  # x2
    boxes[..., 3].clamp_(0, shape[0])  # y2

def scale_boxes(img1_shape, boxes, img0_shape):
    gain = min(
        img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]
    )  # gain  = old / new
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (
        img1_shape[0] - img0_shape[0] * gain
    ) / 2  # wh padding

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes




def main():
    colors = [(0, 0, 200), (200, 200, 0)]
    line_width = 6

    opt_session = onnxruntime.SessionOptions()
    opt_session.enable_mem_pattern = False
    opt_session.enable_cpu_mem_arena = False
    opt_session.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    model_path = './onnx_models/yolov8_mask_detection.onnx'
    EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']

    ort_session = onnxruntime.InferenceSession(model_path, providers=EP_list)
    model_inputs = ort_session.get_inputs()
    input_names = [model_inputs[i].name for i in range(len(model_inputs))]
    model_input_shape = model_inputs[0].shape

    model_output = ort_session.get_outputs()
    output_names = [model_output[i].name for i in range(len(model_output))]
    
    cls_names = None
    meta = ort_session.get_modelmeta().custom_metadata_map
    if "stride" in meta:
        cls_names = eval(meta["names"])

    capture = cv2.VideoCapture(0)
    if capture.isOpened() is False:
        raise IOError
    

    max_det = 1000  # maximum detections per image
    # conf_thres = 0.25  # confidence threshold
    # iou_thres = 0.45  # NMS IOU threshold
    conf_thres = 0.5  # confidence threshold
    iou_thres = 0.45  # NMS IOU threshold
    imgsz = (640, 640)
    agnostic_nms = False
    hide_conf = False
    stride = 32


    while(True):
        try:
            ret, frame = capture.read()

            if ret is False:
                raise IOError
            
            
            frame_t = frame[:,:,::-1]
            frame_t = cv2.resize(frame_t, (model_input_shape[2], model_input_shape[3]))
            
            frame_t = frame_t / 255.0
            frame_t = frame_t.transpose(2,0,1)
            frame_t = frame_t[np.newaxis, :, :, :].astype(np.float32)

            y = ort_session.run(output_names, {input_names[0]: frame_t})[0]

            pred = torch.from_numpy(y[0])
            # pred = non_max_suppression(
            #     pred, conf_thres, iou_thres, agnostic_nms, max_det=max_det
            # )

            for det in pred:
                print("det", det)
                im0 = frame.copy()
                if len(det):
                    print(det[:4])
                    det[:4] = scale_boxes(frame_t.shape[2:], det[:4], im0.shape).round()
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)
                        label = cls_names if hide_conf else f"{cls_names} {conf:.2f}"  #

                        im0 = box_label(
                            im0, xyxy, line_width, label, color=colors
                        )

            cv2.imshow('frame',frame)
            cv2.waitKey(1)

        except KeyboardInterrupt:
            # 終わるときは CTRL + C を押す
            break

    capture.release()
    cv2.destroyAllWindows()


    pass

if __name__ == '__main__':
    main()









print("##################################")
