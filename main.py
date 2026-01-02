from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import io
import os
import uuid

MODEL_PATH = "models/best.pt"
OUTPUT_DIR = "output"
CONF_THRESHOLD = 0.5
ALLOWED_CLASSES = ["pothole", "garbage"]


try:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
except FileExistsError:
    pass

app = FastAPI(title="AI Complaint Validation API")
model = YOLO(MODEL_PATH)

@app.post("/validate-complaint")
async def validate_complaint(image: UploadFile = File(...)):
    try:
        pil_img = Image.open(image.file).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    img_np = np.array(pil_img)
    results = model(img_np, conf=CONF_THRESHOLD, verbose=False)[0]

    detections = []
    detected_classes = set()

    if results.boxes is not None:
        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]

            if cls_name in ALLOWED_CLASSES:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detected_classes.add(cls_name)

                detections.append({
                    "class": cls_name,
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                })

                cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    img_np, cls_name, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
                )

    image_id = f"{uuid.uuid4()}.jpg"
    image_path = os.path.join(OUTPUT_DIR, image_id)
    cv2.imwrite(image_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

    if detected_classes:
        return {
            "complaint_valid": True,
            "detected_objects": list(detected_classes),
            "detections": detections,
            "annotated_image_url": f"/annotated-image/{image_id}",
            "remark": f"{' and '.join(detected_classes)} detected"
        }

    return {
        "complaint_valid": False,
        "detected_objects": [],
        "detections": [],
        "annotated_image_url": f"/annotated-image/{image_id}",
        "remark": "No pothole or garbage detected"
    }

@app.get("/annotated-image/{image_name}")
def get_image(image_name: str):
    path = os.path.join(OUTPUT_DIR, image_name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path, media_type="image/jpeg")
