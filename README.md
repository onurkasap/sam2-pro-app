
# 🚀 SAM 2 Video Segmentation Web App

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95%2B-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![SAM 2](https://img.shields.io/badge/Model-SAM%202-purple)
![License](https://img.shields.io/badge/License-MIT-grey)

A high-performance web interface for **Meta's Segment Anything Model 2 (SAM 2)**. This application allows users to upload videos, interactively select objects via click prompts, and generate segmented videos with high precision using GPU acceleration.

> **Note:** This project is designed to run on local GPUs (tested on RTX 3060 Ti) with optimized inference and memory management.

---

## 🌟 Features

* **Interactive Segmentation:** Select objects in the first frame simply by clicking.
* **Real-time Progress Tracking:** Accurate loading bars for upload, AI processing, and video rendering phases.
* **GPU Acceleration:** Built on PyTorch with CUDA support for fast inference.
* **Optimized Performance:**
    * JPEG quality optimization for faster disk I/O.
    * Smart frame extraction and resizing.
    * Async/Sync handling in FastAPI to prevent blocking.
* **User-Friendly Interface:** Modern Dark Mode UI with pure HTML/JS (No complex frontend framework required).

## 🏗️ Architecture

The system follows a monolithic architecture designed for local deployment:

1.  **Client:** HTML5/JS Frontend sends video and click coordinates.
2.  **API Layer (FastAPI):** Handles requests, manages file uploads, and coordinates the pipeline.
3.  **Processing Unit:**
    * **OpenCV:** Extracts frames and renders the final video.
    * **SAM 2 Model:** Performs the segmentation on the GPU.
4.  **Storage:** Temporary frame caching for efficient processing.

## 🛠️ Installation

### 1. Clone the Repository

git clone [https://github.com/onurkasap/sam2-pro-app.git](https://github.com/onurkasap/sam2-pro-app.git)
cd sam2-pro-app



### 2. Install Dependencies

Make sure you have Python 3.10+ and CUDA installed.


pip install -r requirements.txt



### 3. Download the SAM 2 Checkpoint ⚠️

Since the model file is too large for GitHub, you must download it manually.

1. Download `sam2.1_hiera_base_plus.pt` from the official [Meta SAM 2 repository](https://github.com/facebookresearch/sam2).
2. Create a `checkpoints` folder in the root directory.
3. Place the file inside: `checkpoints/sam2.1_hiera_base_plus.pt`

### 4. Run the Application


uvicorn app.main:app --reload --port 8000



## 🎮 Usage

1. Open your browser and go to `http://127.0.0.1:8000`.
2. Click on the **"Video Segmentation"** tab.
3. Upload an MP4 video (keep it short for faster processing).
4. Wait for the frames to extract.
5. **Click on the object** you want to track in the first frame.
6. Watch the AI process the video and download the result!

---

---

# 🇹🇷 SAM 2 Video Segmentasyon Web Uygulaması

Meta'nın **Segment Anything Model 2 (SAM 2)** modeli için geliştirilmiş, yüksek performanslı bir web arayüzü. Bu uygulama, kullanıcıların video yüklemesine, tıklama yoluyla nesne seçmesine ve GPU hızlandırması kullanarak segmente edilmiş videolar oluşturmasına olanak tanır.

> **Not:** Bu proje, optimize edilmiş bellek yönetimi ve çıkarım süreçleri ile yerel GPU'larda (RTX 3060 Ti üzerinde test edilmiştir) çalışacak şekilde tasarlanmıştır.

## 🌟 Özellikler

* **Etkileşimli Segmentasyon:** İlk karede nesneyi sadece tıklayarak seçin.
* **Gerçek Zamanlı Takip:** Yükleme, AI işleme ve video oluşturma aşamaları için doğru ilerleme çubukları.
* **GPU Hızlandırma:** Hızlı çıkarım (inference) için CUDA destekli PyTorch altyapısı.
* **Optimize Edilmiş Performans:**
* Daha hızlı disk I/O işlemleri için JPEG kalite optimizasyonu.
* Akıllı kare ayrıştırma ve boyutlandırma.
* FastAPI üzerinde bloklamayı önleyen asenkron yapı.


* **Kullanıcı Dostu Arayüz:** Modern Karanlık Mod (Dark Mode) UI.

## 🛠️ Kurulum

### 1. Projeyi İndirin (Clone)


git clone [https://github.com/onurkasap/sam2-pro-app.git](https://github.com/onurkasap/sam2-pro-app.git)
cd sam2-pro-app



### 2. Kütüphaneleri Yükleyin

Python 3.10+ ve CUDA kurulu olduğundan emin olun.


pip install -r requirements.txt



### 3. SAM 2 Model Dosyasını İndirin ⚠️

Model dosyası GitHub için çok büyük olduğundan manuel indirmeniz gerekir.

1. `sam2.1_hiera_base_plus.pt` dosyasını resmi [Meta SAM 2 sayfasından](https://github.com/facebookresearch/sam2) indirin.
2. Ana dizinde `checkpoints` adında bir klasör oluşturun.
3. Dosyayı içine atın: `checkpoints/sam2.1_hiera_base_plus.pt`

### 4. Uygulamayı Başlatın


uvicorn app.main:app --reload --port 8000



## 👨‍💻 Geliştirici / Developer

Developed by **[Adınız Soyadınız]**

* LinkedIn: [Profil Linkiniz]
* GitHub: [GitHub Profiliniz]
