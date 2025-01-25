import cv2
import face_recognition
from ultralytics import YOLO

# Carregue o modelo YOLO uma vez
model = YOLO('dataset/runs/detect/train5/weights/best.pt')

def detect_image(image):
    print("Iniciando detecção de imagem...")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.predict(image_rgb, conf=0.25)

    for result in results[0].boxes:
        box = result.xyxy[0].cpu().numpy()
        class_id = int(result.cls[0].item())
        confidence = result.conf[0].item()
        class_name = model.names[class_id]
        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
        cv2.putText(image, f"{class_name} {confidence:.2f}", (int(box[0]), int(box[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return image

def recognize_faces(image):
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(image, "Face", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return image
