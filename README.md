# 🏗️ Construction Material Detection & Reporting System

An AI-powered construction site monitoring system that uses YOLOv8 object detection to automatically identify, count, and generate reports for construction materials from site images.

---

## 📌 Project Overview

Construction sites require continuous monitoring of materials such as sand, aggregate, cement bags, rebar bundles, and bitumen drums. Manual inventory tracking is time-consuming and prone to errors.

This project automates the process using Computer Vision and Deep Learning. Users can upload construction site images through a Streamlit dashboard, and the system automatically:

- Detects construction materials
- Counts detected materials
- Estimates pile sizes
- Generates inventory reports
- Provides downloadable material reports

---

## 🚀 Features

### Material Detection
Detects the following construction materials:

- Aggregate Pile
- Sand Pile
- Cement Bags
- Rebar Bundle
- Bitumen Drums

### Material Inventory
- Automatic counting of detected materials
- Structured inventory table generation

### Pile Size Estimation
- Estimates pile sizes using bounding box area
- Categorizes piles into:
  - Small
  - Medium
  - Large

### Automated Reporting
- Generates construction material reports
- Summarizes detected materials
- Includes pile size information
- Downloadable as TXT file

### Interactive Dashboard
Built using Streamlit with:
- Image Upload
- Detection Visualization
- Inventory Tables
- Material Reports
- Report Download Functionality

---

## 🧠 Model Information

### Model
YOLOv8 Object Detection Model

### Classes

| Class ID | Material |
|-----------|-----------|
| 0 | Aggregate Pile |
| 1 | Bitumen Drums |
| 2 | Cement Bags |
| 3 | Rebar Bundle |
| 4 | Sand Pile |

---

## 📊 Model Performance

### Validation Results

| Metric | Score |
|----------|----------|
| Precision | 0.699 |
| Recall | 0.561 |
| mAP@50 | 0.618 |
| mAP@50-95 | 0.473 |

### Manual Testing Accuracy

| Material | Correct Predictions |
|-----------|-------------------|
| Aggregate Pile | 8 / 9 |
| Bitumen Drums | 9 / 9 |
| Cement Bags | 8 / 9 |
| Rebar Bundle | 8 / 8 |
| Sand Pile | 8 / 9 |

Overall testing accuracy: **~90%**

---

## 🛠️ Technologies Used

- Python
- YOLOv8
- Ultralytics
- OpenCV
- NumPy
- Pandas
- Pillow
- Streamlit

---

## 📂 Project Structure

```text
Construction_Material_Detection/
│
├── app.py
├── requirements.txt
├── README.md
│
├── model/
│   └── best.pt
│
├── results/
│   ├── confusion_matrix.png
│   ├── results.png
│   └── validation_images
│
└── reports/
    └── material_report.txt
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/BhaktiBhardwaj/Construction-Material-Detection-Reporting-System

cd construction-material-detection
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Application

```bash
streamlit run app.py
```

The dashboard will open automatically in your browser.

---

## 📷 Workflow

1. Upload construction site image
2. Click "Detect Materials"
3. View detected objects
4. Analyze material inventory
5. Review pile size estimation
6. Generate material report
7. Download report

---

## 📈 Future Enhancements

- Multi-image batch processing
- Real-world volume estimation
- PDF report generation
- Cloud deployment
- CCTV/live camera integration
- Construction progress tracking
- Material usage analytics

---

## 🎯 Applications

- Construction Site Monitoring
- Material Inventory Management
- Project Progress Tracking
- Resource Planning
- Site Auditing
- Smart Construction Systems

---

## 👩‍💻 Author

**Bhakti Bhardwaj**

AI/ML Engineer | Full-Stack Developer

Passionate about Artificial Intelligence, Computer Vision, Machine Learning, and Smart Infrastructure Solutions.

---

## 📄 License

This project is developed for educational and research purposes.
