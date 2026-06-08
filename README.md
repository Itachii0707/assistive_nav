# VisionAI - Assistive Navigation System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![YOLO26](https://img.shields.io/badge/YOLO-26-green.svg)](https://github.com/ultralytics/ultralytics)
[![Flask](https://img.shields.io/badge/Flask-Web%20Server-red.svg)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**VisionAI** is a real-time object detection and audio guidance system designed to help visually impaired individuals navigate their surroundings safely. It utilizes deep learning to identify obstacles, assess their risk level based on proximity, and provide natural language voice directions.

---

> [!WARNING]
> **Deployment & Accuracy Note:** This project must be run using its local Python backend (`run_web.py` or `main.py`). Do not deploy it as a purely static website (e.g., via GitHub Pages or Vercel). Static deployments force the app to use a lightweight client-side TensorFlow.js fallback model, which results in **poor accuracy and bounding box errors**. The high accuracy and custom object detection are only available when the backend Python server is running the custom YOLO26 model.

## 🌟 Key Features

*   **Real-time Object Detection:** Powered by a dual-model system (general COCO + custom trained navigation objects) using the YOLO architecture for high accuracy and speed.
*   **Intelligent Risk Assessment:** Divides the camera view into zones (Left, Center, Right) and calculates object proximity to determine risk levels (Low, Medium, High).
*   **Voice Guidance (TTS):** Generates clear, concise audio instructions (e.g., "Stop, chair ahead", "Path clear") using Text-To-Speech to guide the user.
*   **Cross-Platform Web App:** Features a modern, accessible web interface accessible via a PC or a smartphone browser (when connected to the same WiFi network).
*   **Glassmorphic UI:** A beautifully designed frontend with smooth scanning animations, dynamic status banners, and visual bounding boxes.

---

## 🛠️ Architecture

The system operates using a multi-threaded pipeline:

1.  **Input:** Camera stream (Front/Rear via web browser or local PC camera).
2.  **Detection:** Dual YOLO models process frames to identify objects.
3.  **Analysis:**
    *   *Zone Mapping:* Where is the object?
    *   *Proximity Check:* How close is the object?
    *   *Risk Classification:* How dangerous is the obstacle?
4.  **Decision:** Calculates the safest path (Left, Right, or Stop).
5.  **Output:** Scene description and voice alerts via the device's audio.

---

## 🚀 Getting Started

### Prerequisites

*   Python 3.8 or higher
*   A webcam (built-in or USB) or a smartphone

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Itachii0707/assistive_nav.git
    cd assistive_nav
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # macOS/Linux
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 🖥️ Usage

### Running the Local Desktop App

To run the system directly on your PC with a connected webcam:

```bash
python main.py
```
*   Press `q` to quit the application.
*   Press `b` to print benchmark statistics.

### Running the Web Application (Phone/PC)

The web server allows you to use your smartphone's camera (or another PC) as the navigation device.

1.  **Start the server:**
    ```bash
    # Run with dual model configuration (recommended)
    python run_web.py --port 8080 --dual --model runs/detect/runs/train/combined_model_max/weights/best.pt
    ```

2.  **Access the app:**
    *   **On the host PC:** Open your browser and navigate to `http://localhost:8080`
    *   **On a Smartphone:** Ensure your phone is on the **same WiFi network** as the PC. The terminal will display the local IP address (e.g., `http://192.168.1.5:8080`). Open this link in your mobile browser.

3.  **Permissions:** When prompted by the browser, allow access to the Camera and Microphone.

---

## 📁 Project Structure

*   `core/`: Contains the backend logic (`detector.py`, `decision.py`, `proximity.py`, `risk_classifier.py`).
*   `web/`: Contains the Flask web application, including the modern `index.html` frontend.
*   `models/` & `runs/`: Stored weights (`.pt` files) for the object detection models.
*   `datasets/`: Directory for managing training data (ignored by git).
*   `main.py`: Entry point for the local desktop application.
*   `run_web.py`: Entry point for the Flask web server.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the project.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.