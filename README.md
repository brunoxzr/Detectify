# Detectify Documentation

## Table of Contents
1. [Introduction](#introduction)  
2. [Project Objectives](#project-objectives)  
3. [Requirements](#requirements)  
4. [Installation](#installation)  
5. [Project Structure](#project-structure)  
6. [System Workflow](#system-workflow)  
7. [API Routes](#api-routes)  
8. [Image and Video Processing](#image-and-video-processing)  
9. [Report Generation](#report-generation)  
10. [Processing History](#processing-history)  
11. [Project Benefits](#project-benefits)  
12. [Required Investments](#required-investments)  
13. [Final Considerations](#final-considerations)  

---

## Introduction  
**Detectify** is a Flask-based system designed for object detection and facial recognition using the YOLO (You Only Look Once) model. It enables image and video processing, generates visual reports, and maintains a history of analyses.

Detectify is specifically designed for educational environments to improve discipline, attention, and student focus. It detects mobile phones, e-cigarettes, and behaviors like sleeping, copying, or distractions.

The **Federal University of Technology - Paraná (UTFPR)** can act as a partner in the development of this technology, offering research and infrastructure support.

---

## Project Objectives  
Detectify was created with the following goals:  
- **Classroom Monitoring:** Detect the use of mobile phones, e-cigarettes, and monitor inappropriate behaviors.  
- **Real-Time Analysis:** Process videos and images to generate automatic alerts and reports.  
- **Enhancing Focus:** Minimize distractions and help maintain student attention.  
- **Detailed Reporting:** Provide graphs and statistics for post-analysis.  

---

## Requirements  
- Python 3.10 or higher  
- Flask  
- OpenCV  
- Matplotlib  
- NumPy  
- Face Recognition  
- Ultralytics (YOLO)  

### Install Required Libraries  

pip install flask opencv-python-headless matplotlib numpy face-recognition ultralytics

### Installation
Clone the repository:
bash
## git clone https://github.com/your-repo/detectify.git
Navigate to the project directory:
bash
cd detectify
## Install the dependencies:
bash
## pip install -r requirements.txt
## Run the server:
bash
## python app.py
Access the application in your browser:
http://127.0.0.1:8080
## Project Structure
plaintext
/detectify
├── app.py              # Flask application code
├── detect.py           # Image and video processing functions
├── config.py           # Global settings
├── uploads/            # Uploaded and processed files
├── static/             # Static files (CSS, JS, images)
│   ├── css/
│   │   └── styles.css  # Application styles
│   ├── js/
│   │   └── scripts.js  # Website functionality scripts
│   ├── images/
│   │   └── logo.png    # Project logo
│   └── reports/        # Generated graphs
├── templates/          # HTML templates
│   ├── index.html      # Home page
│   ├── history.html    # Processing history
│   └── report.html     # Analysis reports
└── requirements.txt    # Project dependencies
System Workflow
Users upload an image or video through the homepage form.
The system processes the file and applies the YOLO model for object detection.
A graphical report is generated, and the processed file is saved.
The history of processed files is saved locally and can be accessed on the "History" page.
API Routes
Homepage
GET /

Displays the interface for uploading and processing files.
History
GET /history

Lists all processed files, generated graphs, and statistics.
Upload
POST /upload

Receives image or video files.
Processes and returns the path to the report and processed file.
Example Response:

json
{
  "success": true,
  "graph_path": "/static/reports/report_graph.png",
  "video_path": "/uploads/processed_video.mp4"
}
Processed Files
GET /uploads/<filename>

Allows downloading or viewing processed files.
Image and Video Processing
Images
Images are uploaded and processed using OpenCV.
The YOLO model detects objects and draws bounding boxes.
The processed image is saved in the uploads folder.
Videos
Each video frame is processed with the YOLO model.
The processed video is saved in MP4 format.
Progress is displayed during processing.
Report Generation
The reports include:

Bar graphs showing counts of detected objects.
Files saved in static/reports/.
Results organized by date in the history.
Example Graphs:
Count of detected objects during a class.
Percentage of frames with identified faces.
Processing History
Processed videos and images are saved in the uploads/ folder.
Graphical reports are stored in the static/reports/ folder.
Processed files can be viewed on the /history page.
Project Benefits
Enhances Classroom Focus: Prevents distractions and inappropriate behaviors.
Automated Reporting: Provides easy-to-analyze graphical data.
Low Cost: Utilizes accessible hardware.
Required Investments
Infrastructure: Laboratories for testing.
Training: Skill development for students and technical staff.
Partnerships: Academic and industrial collaborations.
Final Considerations
Detectify is an innovative solution for classroom monitoring, promoting discipline and focus. With support from UTFPR and other institutions, it can be enhanced to meet more complex demands.

Developed by: Bruno Yudi Kay
