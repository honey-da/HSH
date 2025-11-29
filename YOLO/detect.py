import cv2
import numpy as np
import socket
import onnxruntime as ort

# ============================
# 서버 연결
# ============================
HOST = "127.0.0.1"
PORT = 9000
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((HOST, PORT))
print("서버 연결됨!")

# ============================
# ONNX 로드 (GPU)
# ============================
model_path = "yolov8m.onnx"

sess = ort.InferenceSession(
    model_path,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

input_name = sess.get_inputs()[0].name

# ============================
# 카메라
# ============================
cap = cv2.VideoCapture(0)

INPUT_W, INPUT_H = 640, 640
CONF_THRES = 0.25
NMS_THRES = 0.45

printed_shape = False

def preprocess(image):
    img = cv2.resize(image, (INPUT_W, INPUT_H))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    ih, iw = frame.shape[:2]
    blob = preprocess(frame)

    # ============================
    # 모델 추론
    # ============================
    raw = sess.run(None, {input_name: blob})[0]   # (1,84,8400)

    if not printed_shape:
        print("MODEL OUTPUT SHAPE =", raw.shape)
        printed_shape = True

    # (1,84,8400) → (8400,84)
    preds = np.transpose(raw, (0, 2, 1))[0]

    boxes = []
    confs = []
    class_ids = []

    for det in preds:
        # det: [cx,cy,w,h, class0..class79]

        cls_scores = det[4:]         # 클래스 점수 80개
        cls_id = int(np.argmax(cls_scores))
        cls_conf = cls_scores[cls_id]

        if cls_id != 0:              # 사람만
            continue

        if cls_conf < CONF_THRES:
            continue

        cx, cy, w, h = det[:4]

        # 좌표 복원
        x1 = int((cx - w/2) * iw / INPUT_W)
        y1 = int((cy - h/2) * ih / INPUT_H)
        x2 = int((cx + w/2) * iw / INPUT_W)
        y2 = int((cy + h/2) * ih / INPUT_H)

        boxes.append([x1, y1, x2 - x1, y2 - y1])
        confs.append(float(cls_conf))
        class_ids.append(cls_id)

    # ============================
    # NMS
    # ============================
    idxs = cv2.dnn.NMSBoxes(boxes, confs, CONF_THRES, NMS_THRES)

    detected = False

    if len(idxs) > 0:
        detected = True
        for i in idxs.flatten():
            x, y, w, h = boxes[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Person", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # ============================
    # 사람 감지 → SLEEP
    # 사람 없음 → WAKE
    # ============================
    if detected:
        print("사람 감지됨 → SLEEP")
        sock.sendall(b"SLEEP\n")
    else:
        print("사람 없음 → WAKE")
        sock.sendall(b"WAKE\n")

    cv2.imshow("YOLOv8m GPU", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
sock.close()
cv2.destroyAllWindows()
