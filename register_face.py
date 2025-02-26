import cv2
import numpy as np
from detect_faces import detect_faces
from extract_features import extract_features

name = input("Enter user name: ")
image_path = f"data/images/{name}.jpg"

img = cv2.imread(image_path)

if img is None:
    print(f"Error: Unable to load image from {image_path}")
    exit()

faces, _ = detect_faces(img)  # Ensure detect_faces() returns a list

print(f"Detected {len(faces)} face(s)")

if len(faces) == 0:
    print("No face detected. Try another image.")
    exit()

embedding = extract_features(faces[0])  # Extract features
np.save(f"data/embeddings/{name}.npy", embedding)  # Save embedding
print(f"Face embedding saved for {name}")
