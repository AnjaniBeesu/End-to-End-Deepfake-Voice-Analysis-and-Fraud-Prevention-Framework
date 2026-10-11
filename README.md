# 🎙️ End-to-End Deepfake Voice Analysis Framework

![NSoC 2026](https://img.shields.io/badge/NSoC-2026-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10+-yellow.svg)
![Next.js](https://img.shields.io/badge/Next.js-14-black.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)

An open-source ecosystem designed to detect synthetic audio and prevent voice-cloning fraud. This project combines a modern web dashboard with an AI-driven machine learning backend to analyze audio files for deepfake signatures.

## 🏗️ Architecture
This is a monorepo consisting of:
*   **Frontend:** A Next.js web application allowing users to upload audio files and view analysis reports.
*   **Backend:** A Python FastAPI server handling API requests, file uploads, and database logging.
*   **Model:** PyTorch/TensorFlow scripts for audio preprocessing and deepfake classification.

## 🚀 Quick Start (Local Setup)

### 1. Backend Setup
\`\`\`bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
\`\`\`
The API will run at `http://localhost:8000`

### 2. Frontend Setup
\`\`\`bash
cd frontend
npm install
npm run dev
\`\`\`
The UI will run at `http://localhost:3000`

## 🤝 Contributing
We welcome contributions for the **Nexus Spring of Code (NSoC) 2026**! 
Please check out our [Issues](https://github.com/vanshi18s/End-to-End-Deepfake-Voice-Analysis-and-Fraud-Prevention-Framework/issues) tab for tasks labeled `good first issue`. 

Before submitting a Pull Request, please read our `CONTRIBUTING.md` (coming soon).

## 📄 License
This project is licensed under the MIT License.