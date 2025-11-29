import cv2
import numpy as np
import onnxruntime as ort
import socket

# 서버 연결
HOST = "127.0.0.1"
PORT = 9000
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((HOST, PORT))

# YOLOv8m ONNX 로드 (GPU 실행)
sess = ort.InferenceSession(
    "yolov8m.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

input_name = sess.get_inputs()[0].name

INPUT_W, INPUT_H = 640,640
CONF_THRES = 0.5
NMS_THRES = 0.45

def preprocess(img):
    img = cv2.resize(img,(640,640))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)/255.0
    img = np.transpose(img,(2,0,1))[None,:,:,:]
    return img

def sigmoid(x):
    return 1/(1+np.exp(-x))

cap = cv2.VideoCapture(0)

while True:
    ret,frame = cap.read()
    if not ret: break
    h,w = frame.shape[:2]

    blob = preprocess(frame)
    pred = sess.run(None,{input_name:blob})[0]  # (1,84,8400)
    pred = np.transpose(pred,(0,2,1))[0]        # (8400,84)

    # YOLOv8 후처리
    pred[:,:4] = sigmoid(pred[:,:4])
    pred[:,4:] = sigmoid(pred[:,4:])

    boxes=[]
    confs=[]

    for det in pred:
        cx,cy,bw,bh = det[:4]
        obj = det[4]
        cls = det[5]   # person only

        conf = obj*cls
        if conf < CONF_THRES:
            continue

        cx = cx * w
        cy = cy * h
        bw = bw * w
        bh = bh * h

        x1 = int(cx - bw/2)
        y1 = int(cy - bh/2)

        boxes.append([x1,y1,int(bw),int(bh)])
        confs.append(float(conf))

    idxs = cv2.dnn.NMSBoxes(boxes, confs, CONF_THRES,NMS_THRES)

    detected = False

    if len(idxs)>0:
        detected=True
        for i in idxs.flatten():
            x,y,bw,bh= boxes[i]
            cv2.rectangle(frame,(x,y),(x+bw,y+bh),(0,255,0),2)

    if detected:
        sock.sendall(b"WAKE\n")

    cv2.imshow("YOLOv8m GPU",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
sock.close()
cv2.destroyAllWindows()
