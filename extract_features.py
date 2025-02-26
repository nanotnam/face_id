import cv2
import numpy as np
from keras_facenet import FaceNet

embedder = FaceNet()

def preprocess_face(face_image):
    """Preprocess face before feature extraction."""
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    face_image = cv2.resize(face_image, (160, 160))  # Resize to FaceNet input size
    face_image = np.expand_dims(face_image, axis=0) / 255.0  # Normalize (0-1)
    return face_image

def extract_features(face_image):
    """Extract features from a face image."""
    face_image = preprocess_face(face_image)
    print(face_image)
    embedding = embedder.embeddings(face_image)  # Get feature vector
    print(embedding[0].shape)
    return embedding[0]  # Return 1D array

