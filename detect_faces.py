from retinaface import RetinaFace
import cv2
import numpy as np

def detect_faces(img):
    """Detects faces in an image using RetinaFace and returns cropped, preprocessed faces with bounding boxes."""
    faces = RetinaFace.detect_faces(img)

    if not faces:
        print("⚠️ No faces detected!")
        return [], []

    detected_faces = []
    boxes = []

    for key in faces.keys():
        identity = faces[key]
        x1, y1, x2, y2 = identity["facial_area"]

        # Ensure valid crop
        if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
            print("⚠️ Face cropping out of bounds, skipping...")
            continue

        face = img[y1:y2, x1:x2]

        # Check if face is valid
        if face.size == 0:
            print("⚠️ Empty face detected, skipping...")
            continue

        # Resize and normalize face for FaceNet
        face = cv2.resize(face, (160, 160))
        face = face.astype('float32') / 255.0  # Normalize to [0,1]

        detected_faces.append(face)
        boxes.append((x1, y1, x2, y2))

    return detected_faces, boxes


