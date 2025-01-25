import os

# Configuração do diretório de uploads
UPLOAD_FOLDER = 'uploads'

# Criação do diretório se não existir
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
