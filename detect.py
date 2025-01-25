import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from config import UPLOAD_FOLDER
from utils import detect_image
import datetime
import csv

log_file = 'static/reports/detection_log.csv'


# Criar arquivo CSV se não existir
if not os.path.exists(log_file):
    with open(log_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Timestamp', 'Dormindo', 'Acordado', 'Copiando', 'Celular', 'Cigarro_eletronico'])

# Modelo YOLO
model = YOLO('dataset/runs/detect/train9/weights/last.pt')

def log_detection(counts):
    """Registra os números de detecção no CSV."""
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            current_time,
            counts.get('dormindo', 0),
            counts.get('acordado', 0),
            counts.get('copiando', 0),
            counts.get('celular', 0),
            counts.get('cigarro_eletronico', 0)
        ])

def process_video(video_path):
    """Processa o vídeo com o modelo YOLO."""
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_skip = 3
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:  # Pula os frames
            continue

    # Criação do diretório processado
    processed_dir = 'processed'
    os.makedirs(processed_dir, exist_ok=True)

    # Caminho de saída do vídeo processado
    output_path = os.path.join(processed_dir, f"processed_{os.path.basename(video_path)}")
    ourcc = cv2.VideoWriter_fourcc(*'VP80')  # Codec VP8 para WebM
    output_path = os.path.join('processed', f"processed_{os.path.basename(video_path)}.mp4")

    # Codec H.264 (Compatível com MP4 para navegadores)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Use 'avc1' para H.264 ou 'VP80' para WebM
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Contagem por classe
    detections_count = {"dormindo": 0, "acordado": 0, "copiando": 0, "celular": 0, "cigarro_eletronico": 0}
    detected_ids = set()  # Para evitar contagens duplicadas

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Realiza a detecção
        results = model.predict(frame, conf=0.25)

        for result in results[0].boxes:
            box = result.xyxy[0].cpu().numpy()
            confidence = result.conf[0].item()
            class_id = int(result.cls[0].item())
            class_name = model.names[class_id]

            # Atualiza contagem se for um novo ID
            if class_name in detections_count:
                detections_count[class_name] += 1

            # Desenha caixas e rótulos
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
            cv2.putText(frame, f"{class_name} {confidence:.2f}", (int(box[0]), int(box[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Escreve o frame processado
        out.write(frame)

    cap.release()
    out.release()
    print(f"Vídeo processado salvo em: {output_path}")

    # Salva logs
    log_detection(detections_count)

    return {"processed_video": output_path, "detections": detections_count}

def generate_report(detections, processed_video_path):
    try:
        # Validação - Garante que haja pelo menos 1 detecção
        if not detections or all(count == 0 for count in detections.values()):
            raise ValueError("Nenhuma detecção encontrada no vídeo.")

        # Diretório de relatórios
        reports_dir = 'static/reports'
        os.makedirs(reports_dir, exist_ok=True)

        # Gráfico de Barras
        graph_filename = os.path.basename(processed_video_path).replace('.mp4', '_graph.png')
        graph_path = os.path.join(reports_dir, graph_filename)

        plt.figure(figsize=(10, 5))
        plt.bar(detections.keys(), detections.values(), color='skyblue')
        plt.title("Contagem de Objetos Detectados")
        plt.xlabel("Objeto")
        plt.ylabel("Quantidade")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(graph_path)
        plt.close()

        # Gráfico de Pizza
        pie_filename = os.path.basename(processed_video_path).replace('.mp4', '_pie.png')
        pie_path = os.path.join(reports_dir, pie_filename)

        plt.figure(figsize=(8, 8))
        plt.pie(detections.values(), labels=detections.keys(), autopct='%1.1f%%', startangle=140)
        plt.title("Distribuição de Objetos Detectados")
        plt.savefig(pie_path)
        plt.close()

        print(f"Gráficos salvos em: {graph_path} e {pie_path}")

        return {
            "graph": f"/{graph_path}",
            "pie": f"/{pie_path}",
            "processed_video": processed_video_path
        }
    except Exception as e:
        print(f"Erro ao gerar relatório: {str(e)}")
        raise


def process_image(image_path):
    global progress
    progress["current"] = 0

    # Lê a imagem
    image = cv2.imread(image_path)
    progress["current"] = 20

    # Verifica se a imagem foi carregada corretamente
    if image is None:
        raise ValueError(f"Erro ao carregar a imagem: {image_path}")

    # Realiza inferência com o modelo YOLO
    results = model.predict(image, conf=0.35)  # Ajuste da confiança para 0.35
    progress["current"] = 70

    # Lista de detecções para o relatório
    detections = {}  # Alterado para dicionário para contagens

    # Processa cada detecção
    for result in results[0].boxes:
        # Extrai informações da detecção
        box = result.xyxy[0].cpu().numpy()  # Coordenadas da caixa
        confidence = result.conf[0].item()  # Confiança da detecção
        class_id = int(result.cls[0].item())  # ID da classe detectada
        class_name = model.names[class_id]  # Nome da classe

        # Atualiza contagem no dicionário
        detections[class_name] = detections.get(class_name, 0) + 1

        # Desenha caixas e textos na imagem
        label = f"{class_name} {confidence:.2f}"  # Texto exibido
        cv2.rectangle(image, 
                    (int(box[0]), int(box[1])), 
                    (int(box[2]), int(box[3])), 
                    (0, 255, 0), 2)  # Cor Verde
        cv2.putText(image, label, 
                    (int(box[0]), int(box[1]) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                    (255, 255, 255), 2)  # Texto Branco

    # Salva a imagem processada com as caixas e rótulos
    output_path = os.path.splitext(image_path)[0] + '_processed.jpg'
    cv2.imwrite(output_path, image)
    progress["current"] = 100

    # Retorna as detecções e o caminho do arquivo processado
    return detections, output_path