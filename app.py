import os
import pandas as pd
import cv2
import torch
import tkinter as tk
from tkinter import simpledialog, messagebox
from tkinter import ttk
import face_recognition
import numpy as np
from PIL import Image
from ultralytics import YOLO
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
from utils import detect_image, recognize_faces
from detect import process_image, process_video, generate_report
from config import UPLOAD_FOLDER
import csv
import datetime
import matplotlib
matplotlib.use('Agg')  # Backend não interativo


matplotlib.use('Agg')  # Backend não interativo

# Arquivos e configurações
log_file = 'static/reports/detection_log.csv'
faces_file = 'faces.csv'
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
REPORT_FOLDER = 'static/reports'
LOG_FILE = os.path.join(REPORT_FOLDER, 'detection_log.csv')

# Criar diretórios se não existirem
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

# Modelo YOLO
model = YOLO('dataset/runs/detect/train9/weights/last.pt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Inicializar DataFrame para armazenar os rostos reconhecidos
faces_df = pd.DataFrame(columns=['name', 'encoding'])

# Carregar dados existentes se o arquivo já existir
if os.path.exists(faces_file) and os.path.getsize(faces_file) > 0:
    try:
        faces_df = pd.read_csv(faces_file)
        faces_df['encoding'] = faces_df['encoding'].apply(eval)  # Converte de string para lista
    except pd.errors.EmptyDataError:
        print(f"O arquivo {faces_file} está vazio. Iniciando com DataFrame vazio.")

# Configuração do Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
progress = {"current": 0}  # Progresso global

# Rota de Página Inicial
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/report/<filename>')
def report(filename):
    # Caminho para o relatório HTML gerado
    report_path = os.path.join('static', 'reports', f'{filename}_report.html')
    if not os.path.exists(report_path):
        return "Relatório não encontrado.", 404

    # Dados de exemplo para gráficos
    timestamps = ["10:00", "10:05", "10:10"]
    data = {
        "dormindo": [1, 2, 3],
        "acordado": [4, 5, 6],
        "copiando": [2, 3, 1],
        "celular": [3, 4, 2],
        "cigarro_eletronico": [1, 1, 2]
    }

    # Renderiza o relatório
    return render_template(
        'report.html',
        filename=filename,
        timestamps=timestamps,
        data=data
    )

# Progresso
@app.route('/progress')
def progress_status():
    global progress
    return jsonify(progress)

# Upload de Arquivos
@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "Nenhum arquivo foi enviado"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Nenhum arquivo selecionado"}), 400

        # Salvar arquivo
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Processar arquivo
        if file.filename.endswith(('.jpg', '.jpeg', '.png')):
            detections, processed_path = process_image(file_path)
        elif file.filename.endswith(('.mp4', '.avi', '.mkv', '.webm')):
            result = process_video(file_path)
            processed_path = result['processed_video']
            detections = result['detections']
        else:
            return jsonify({"error": "Formato de arquivo não suportado"}), 400

        # Gera relatório
        report = generate_report(detections, processed_path)
        return render_template('report.html',
                            graph_path=report["graph"],
                            pie_path=report["pie"],
                            video_path=processed_path)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Arquivos processados
@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

@app.route('/history')
def history():
    processed_dir = 'processed'
    files = os.listdir(processed_dir)
    history_data = []
    for filename in files:
        if filename.endswith(('.mp4', '.avi', '.mkv')):
            report_path = url_for('report', filename=os.path.splitext(filename)[0])
            history_data.append({
                "video_path": url_for('processed_file', filename=filename),
                "report_path": report_path,
                "filename": filename,
                "date": "14/12/2024",
                "object_counts": {"Pessoa": 3, "Cadeira": 5}
            })
    return render_template('history.html', history=history_data)



# Dados do Gráfico
@app.route('/graph-data')
def graph_data():
    try:
        timestamps = []
        data = {'dormindo': [], 'acordado': [], 'copiando': [], 'celular': [], 'cigarro_eletronico': []}

        with open(LOG_FILE, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
                timestamps.append(row[0])
                data['dormindo'].append(int(row[1]))
                data['acordado'].append(int(row[2]))
                data['copiando'].append(int(row[3]))
                data['celular'].append(int(row[4]))
                data['cigarro_eletronico'].append(int(row[5]))

        return jsonify({"timestamps": timestamps, "data": data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- FUNÇÕES DE PROCESSAMENTO ---

def process_image(image_path):
    """Processa uma imagem"""
    image = cv2.imread(image_path)
    results = model.predict(image, conf=0.25)
    detections = {}

    for result in results[0].boxes:
        class_id = int(result.cls[0].item())
        class_name = model.names[class_id]
        detections[class_name] = detections.get(class_name, 0) + 1

    return detections, image_path


def process_video(video_path):
    """Processa um vídeo"""
    # Abre o vídeo
    cap = cv2.VideoCapture(video_path)

    # Configurações de vídeo
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define o caminho de saída do vídeo processado
    output_path = os.path.join('processed', f"processed_{os.path.basename(video_path)}.mp4")

    # Codec H.264 (Compatível com MP4 para navegadores)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Use 'avc1' para H.264 ou 'VP80' para WebM
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Inicializa as contagens de detecções
    detections = {'dormindo': 0, 'acordado': 0, 'copiando': 0, 'celular': 0, 'cigarro_eletronico': 0}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Sai do loop se não houver mais frames

        # Realiza inferência no frame
        results = model.predict(frame, conf=0.25)
        for result in results[0].boxes:
            # Extrai as informações da detecção
            class_id = int(result.cls[0].item())
            class_name = model.names[class_id]
            confidence = result.conf[0].item()  # Confiança da detecção

            # Atualiza as contagens de detecções
            if class_name in detections:
                detections[class_name] += 1

            # Desenha a caixa no frame
            box = result.xyxy[0].cpu().numpy()
            cv2.rectangle(frame, 
                        (int(box[0]), int(box[1])), 
                        (int(box[2]), int(box[3])), 
                        (0, 255, 0), 2)  # Cor verde

            # Cria o texto para o rótulo (classe + confiança)
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(frame, label, 
                        (int(box[0]), int(box[1]) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                        (255, 255, 255), 2)  # Texto branco

        # Escreve o frame processado no vídeo de saída
        out.write(frame)

    # Finaliza o processamento
    cap.release()
    out.release()

    # Retorna informações sobre o vídeo processado
    return {"processed_video": output_path, "detections": detections}


    log_detection(detections)
    return {"processed_video": output_path, "detections": detections}


def log_detection(detections):
    """Registra os dados no log"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            timestamp,
            detections.get('dormindo', 0),
            detections.get('acordado', 0),
            detections.get('copiando', 0),
            detections.get('celular', 0),
            detections.get('cigarro_eletronico', 0)
        ])


def generate_report(detections, processed_video_path):
    try:
        filename = os.path.splitext(os.path.basename(processed_video_path))[0]
        report_path = os.path.join('static', 'reports', f'{filename}_report.html')

        # Geração de dados
        timestamps = ["10:00", "10:05", "10:10"]
        data = {
            "dormindo": [1, 2, 3],
            "acordado": [4, 5, 6],
            "copiando": [2, 3, 1],
            "celular": [3, 4, 2],
            "cigarro_eletronico": [1, 1, 2]
        }

        # Renderiza o relatório usando Jinja
        with open(report_path, 'w') as file:
            file.write(render_template(
                'report.html',
                filename=filename,
                timestamps=timestamps,
                data=data
            ))

        return {
            "report": report_path,
            "processed_video": processed_video_path
        }
    except Exception as e:
        print(f"Erro ao gerar relatório: {str(e)}")
        raise



# --- EXECUÇÃO DO FLASK ---
if __name__ == '__main__':
    app.run(debug=True)

def recognize_faces(image):
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(faces_df['encoding'].tolist(), face_encoding)
        name = "Desconhecido"

        face_distances = face_recognition.face_distance(faces_df['encoding'].tolist(), face_encoding)
        if face_distances.size > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = faces_df.iloc[best_match_index]['name']

        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(image, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    return image


def live_detection(selected_camera_index=0):
    cap = cv2.VideoCapture(selected_camera_index)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Ajustar a resolução da imagem para uma melhor performance sem perder muito em qualidade
        frame = cv2.resize(frame, (640, 480))
        
        frame_with_boxes = detect_image(frame)
        frame_with_faces = recognize_faces(frame_with_boxes)
        cv2.imshow('Detecção ao Vivo - Pressione "q" para sair', frame_with_faces)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def add_face():
    cap = cv2.VideoCapture(select_camera())
    if not cap.isOpened():
        print("Erro ao abrir a câmera")
        return None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro ao capturar imagem")
            break

        cv2.imshow('Pressione "s" para salvar o rosto, "q" para sair', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):  # Save face
            cap.release()
            cv2.destroyAllWindows()
            
            # Save the frame as a temporary file and use its path
            temp_image_path = "temp_face.jpg"
            cv2.imwrite(temp_image_path, frame)
            
            # Load the image using face_recognition
            img = face_recognition.load_image_file(temp_image_path)
            
            # Detect face encodings
            face_encodings = face_recognition.face_encodings(img)
            if face_encodings:
                face_encoding = face_encodings[0]
                name = simpledialog.askstring("Nome", "Qual é o nome da pessoa?")
                if name:
                    faces_df.loc[len(faces_df)] = [name, face_encoding.tolist()]
                    faces_df.to_csv(faces_file, index=False)
                    messagebox.showinfo("Adicionado", f"Rosto de {name} adicionado com sucesso.")
            else:
                messagebox.showwarning("Falha", "Nenhum rosto encontrado na imagem.")
            
            # Remove the temporary file
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
            return

        elif key == ord('q'):  # Quit
            cap.release()
            cv2.destroyAllWindows()
            return None


def select_camera():
    camera_index = simpledialog.askinteger("Selecionar Câmera", "Digite o índice da câmera (0 para a primeira câmera, 1 para a segunda, etc.):")
    return camera_index

# Configuração da interface gráfica
root = tk.Tk()
root.title("Reconhecimento de Objetos com YOLOv5")
root.geometry("600x400")
root.configure(background="#f0f0f0")

style = ttk.Style()
style.configure("TButton", font=("Helvetica", 12), padding=10)
style.configure("TLabel", font=("Helvetica", 14), background="#f0f0f0")

title_label = ttk.Label(root, text="Reconhecimento de Objetos com YOLOv5")
title_label.pack(pady=20)

btn_live_detection = ttk.Button(root, text="Detecção ao Vivo", command=lambda: live_detection(select_camera()))
btn_live_detection.pack(pady=10)

btn_add_face = ttk.Button(root, text="Adicionar Rosto", command=add_face)
btn_add_face.pack(pady=10)

root.mainloop()
