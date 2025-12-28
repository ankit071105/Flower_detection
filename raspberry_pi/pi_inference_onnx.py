import cv2
import numpy as np
import onnxruntime as ort
import time

MODEL_PATH = "flower_ssd.onnx"

# Load ONNX
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

def preprocess(frame):
    img = cv2.resize(frame, (320, 320))
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2,0,1)
    return img[np.newaxis, :]

def detect(frame):
    inp = preprocess(frame)
    out = session.run([output_name], {input_name: inp})[0]

    # Model returns dict-like output
    boxes = out["boxes"]
    scores = out["scores"]
    labels = out["labels"]

    dets = []
    for b, s, l in zip(boxes, scores, labels):
        if s < 0.35: 
            continue
        if l != 1: 
            continue
        x1,y1,x2,y2 = map(int, b)
        dets.append((x1,y1,x2,y2,s))
    return dets

def main():
    cap = cv2.VideoCapture(0)
    print("Running flower detector...")

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        detections = detect(frame)

        for (x1,y1,x2,y2,score) in detections:
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,f"{score:.2f}",(x1,y1-5),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

        fps = 1/(time.time()-t0)
        cv2.putText(frame,f"FPS:{fps:.1f}",(10,30),
            cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

        cv2.imshow("Flower Detector Pi (ONNX)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
