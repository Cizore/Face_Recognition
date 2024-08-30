import os
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
from ultralytics import YOLO
import pickle
import numpy as np

# Load YOLOv8 model
try:
    model = YOLO("detection/weights/best.pt")  
    print("YOLOv8 model loaded successfully.")
except Exception as e:
    print(f"Error loading YOLOv8 model: {e}")

# Load MTCNN and InceptionResnetV1 models
try:
    mtcnn = MTCNN(keep_all=True)
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    print("MTCNN and InceptionResnetV1 models loaded successfully.")
except Exception as e:
    print(f"Error loading MTCNN/InceptionResnetV1: {e}")

# Load known embeddings
def load_known_embeddings():
    try:
        with open('known_embeddings.pkl', 'rb') as f:
            known_embeddings = pickle.load(f)
            print("Known embeddings loaded successfully.")
    except Exception as e:
        known_embeddings = {}
        print(f"Error loading known embeddings: {e}")
    return known_embeddings

known_embeddings = load_known_embeddings()

# Function to compare embeddings
def compare_embeddings(embedding, known_embeddings):
    threshold = 0.2
    min_dist = float('inf')
    match = "Unknown"

    for name, known_embedding in known_embeddings.items():
        dist = np.linalg.norm(np.array(embedding) - np.array(known_embedding))
        if dist < min_dist:
            min_dist = dist
            match = name if dist < threshold else "Unknown"

    print(f"Min distance: {min_dist}, Match: {match}")
    return match

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to open the webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read from the webcam.")
        break

    # Detect faces using YOLOv8
    results = model(frame)
    boxes = results[0].boxes
    print(f"Number of faces detected: {len(boxes)}")

    faces = []
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
        face = frame[y1:y2, x1:x2]
        faces.append(face)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    embeddings = []
    for face in faces:
        # Ensure the face is converted to a format suitable for MTCNN
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        boxes, probs = mtcnn.detect(face_rgb)
        if boxes is not None:
            face_tensor = mtcnn(face_rgb).squeeze().unsqueeze(0)
            face_embedding = resnet(face_tensor).detach().cpu().numpy().flatten()
            embeddings.append(face_embedding)
            print("Embedding extracted:", face_embedding)
        else:
            print("No faces detected by MTCNN in the cropped image.")
    
    for embedding in embeddings:
        match = compare_embeddings(embedding, known_embeddings)
        print("Face recognized as:", match)
        for box in boxes:
            if boxes is not None:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, match, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)


    cv2.imshow('Camera Feed', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
