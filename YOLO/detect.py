import cv2
import numpy as np

net = cv2.dnn.readNetFromONNX("yolov8n.onnx")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640),
                                 swapRB=True, crop=False)
    net.setInput(blob)

    preds = net.forward()
    preds = preds[0].transpose(1, 0)

    boxes, class_ids, confidences = [], [], []

    for det in preds:
        conf = det[4]
        if conf < 0.5:
            continue

        cls_scores = det[5:]
        cls_id = np.argmax(cls_scores)
        cls_conf = cls_scores[cls_id]
        if cls_conf < 0.5:
            continue

        cx, cy, bw, bh = det[:4]
        x1 = int((cx - bw/2) * w / 640)
        y1 = int((cy - bh/2) * h / 640)
        x2 = int((cx + bw/2) * w / 640)
        y2 = int((cy + bh/2) * h / 640)

        boxes.append([x1, y1, x2, y2])
        class_ids.append(cls_id)
        confidences.append(float(conf))

    for (x1, y1, x2, y2), cid, conf in zip(boxes, class_ids, confidences):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame,
                    f"{cid}:{conf:.2f}",
                    (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0,255,0), 2)

    cv2.imshow("YOLO Cam", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
