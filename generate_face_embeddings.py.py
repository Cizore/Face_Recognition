import os
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
from ultralytics import YOLO
import pickle

# Initialize YOLOv8 model for face detection
model = YOLO("detection/weights/best.pt")
print("YOLOv8 model loaded successfully.")

# Initialize MTCNN for face detection and InceptionResnetV1 for face recognition
mtcnn = MTCNN(keep_all=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval()
print("MTCNN and InceptionResnetV1 models loaded successfully.")

# Load known embeddings
try:
    with open('known_embeddings_claude.pkl', 'rb') as f:
        known_embeddings = pickle.load(f)
        print("Known embeddings loaded successfully.")
except FileNotFoundError:
    known_embeddings = {}
    print("No known embeddings found. Starting with an empty dictionary.")

# Function to save embeddings for images in a directory structure
def save_embeddings_from_directory(directory_path):
    if not os.path.isdir(directory_path):
        print(f"Error: '{directory_path}' is not a valid directory.")
        return
    for person_dir in os.listdir(directory_path):
        person_path = os.path.join(directory_path, person_dir)
        if os.path.isdir(person_path):
            name = person_dir
            person_embeddings = []
            print(f"Processing '{name}' directory...")
            for filename in os.listdir(person_path):
                if filename.endswith(('.jpg', '.png', '.bmp', '.jpeg')):  # Add any other supported image extensions
                    image_path = os.path.join(person_path, filename)
                    img = cv2.imread(image_path)
                    if img is None:
                        print(f"Error: Unable to read image '{image_path}'.")
                        continue

                    print(f"Processing '{filename}'...")
                    results = model(img)
                    boxes = results[0].boxes.cpu().numpy()

                    faces = []
                    for box in boxes:
                        x1, y1, x2, y2 = [int(val) for val in box]
                        face = img[y1:y2, x1:x2]
                        faces.append(face)

                    embeddings = []
                    for face in faces:
                        boxes, probs = mtcnn.detect(face)
                        if boxes is not None:
                            face_tensor = mtcnn(face).unsqueeze(0)
                            face_embedding = resnet(face_tensor).detach().cpu().numpy().flatten()
                            embeddings.append(face_embedding)

                    person_embeddings.extend(embeddings)
                    print(f"Embeddings saved for '{filename}'.")


            known_embeddings[name] = person_embeddings
            print(f"Embeddings saved for '{name}'.")

    # Save the updated embeddings dictionary
    with open('known_embeddings.pkl', 'wb') as f:
        pickle.dump(known_embeddings, f)
        print("Known embeddings saved successfully.")

# Save embeddings for images in a directory structure
# Directory structure should be:
# known_faces/
# ├── person1_name/
# │   ├── img1.jpg
# │   ├── img2.jpg
# │   └── ...
# ├── person2_name/
# │   ├── img1.jpg
# │   ├── img2.jpg
# │   └── ...
# └── ...
directory_path = "/path/to/directory" #Enter your directory path here  
save_embeddings_from_directory(directory_path)