import cv2
import numpy as np
from detect_faces import detect_faces
from extract_features import extract_features
from sklearn.metrics.pairwise import cosine_similarity
import os

# Load stored embeddings
stored_embeddings = {
    name.split(".")[0]: np.load(f"data/embeddings/{name}")
    for name in os.listdir("data/embeddings")
}

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces and get bounding boxes
    faces, boxes = detect_faces(frame)

    for face, (x1, y1, x2, y2) in zip(faces, boxes):
        embedding = extract_features(face)
        
        best_match = "Unknown"
        best_score = 0.5  # Similarity threshold

        for name, stored_emb in stored_embeddings.items():
            score = cosine_similarity([embedding], [stored_emb])[0][0]
            print(f"Match score with {name}: {score:.2f}")
            if score > best_score:
                best_match = name
                best_score = score

        # Draw bounding box around face
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, best_match, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
