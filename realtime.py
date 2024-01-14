import argparse
import cv2
import time
import numpy as np
from ultralytics import YOLO
from PIL import Image
import torch

from engine.engine import Engine

LABELS = ["man", "woman"]

def main(args):
    engine = Engine(args.config)

    try:
        source = int(args.source)
    except ValueError:
        # If it's not an integer, return it as a string (for file path)
        source = args.source

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Couldn't read from the source {source}")
        return -1

    while True:
        t_start = time.time()
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_height, frame_width = frame.shape[:2]
        
        t_facedet_start = time.time()
        results = face_detector_model(frame, verbose=False, device=engine.device)
        boxes = results[0].boxes.cpu().numpy()
        t_facedet_end = time.time()

        new_frame = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        for i, box_info in enumerate(boxes):
            box = box_info.xywh.astype(int)
            xc, yc, w, h = box[0]
            x1, y1 = int(xc - w/2), int(yc - h/2)
            x2, y2 = int(xc + w/2), int(yc + h/2)

            offset_w, offset_h = int(w*0.2), int(h*0.2)       
            
            x1_ = max(0, (x1 - offset_w))
            y1_ = max(0, (y1 - offset_h))
            x2_ = min(frame_width, (x2 + offset_w))
            y2_ = min(frame_height, (y2 + offset_h))

            face_img = np.copy(frame[y1_:y2_, x1_:x2_])
            face_img = cv2.resize(face_img, (96, 96))
            pil_image = Image.fromarray(face_img)

            processed_face = engine.test_loader.dataset.test_transform(pil_image).unsqueeze(0)
            processed_face = processed_face.to(engine.device)

            t_genderdet_start = time.time()
            outputs = engine.model(processed_face)
            t_genderdet_end = time.time()
            
            predictions = (torch.sigmoid(outputs) > 0.5).int()

            label_id = predictions[0, 0]
            conf = torch.softmax(outputs, dim=1)[0, 0]

            new_frame = cv2.putText(new_frame, f'{LABELS[label_id]} {conf:.2f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.rectangle(new_frame, (x1_, y1_), (x2_, y2_), (0, 255, 0), 2)

        cv2.imshow("Output", new_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        t_end = time.time()
        print(f"FPS: {1/(t_end-t_start):.2f}\tFace Detection: {(t_facedet_end - t_facedet_start)*1000:.2f}ms \t Gender Detection: {(t_genderdet_end - t_genderdet_start)*1000:.2f}ms")

    cv2.destroyAllWindows()
    cap.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--source', type=str, default='0')

    face_detector_model = YOLO("models/yolov8n-face.pt")

    args = parser.parse_args()
    main(args)