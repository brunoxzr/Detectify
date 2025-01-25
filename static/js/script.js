function updateProgress() {
    fetch('/progress')
        .then(response => response.json())
        .then(data => {
            const progressBar = document.getElementById('progress-bar');
            progressBar.style.width = data.current + '%';
            progressBar.innerText = data.current + '%';
            if (data.current < 100) {
                setTimeout(updateProgress, 500); // Atualiza a cada 500ms
            }
        })
        .catch(error => console.error('Erro ao obter progresso:', error));
}

document.getElementById('upload-form').addEventListener('submit', function () {
    updateProgress(); // Inicia o monitoramento do progresso
});
