# 🩺 Symptom-Disease Predictor  

This project predicts possible diseases based on user-entered symptoms using Machine Learning.  
It has two main parts:  

- **Backend (FastAPI)** → Handles model predictions and API endpoints.  
- **Frontend (Streamlit)** → User-friendly web interface for interaction.  

---

## 📂 Project Structure  

symptom-disease-predictor/
│
├── backend/
│ ├── main.py ← FastAPI backend code
│ ├── model.joblib ← Trained ML model
│ └── requirements.txt ← Backend dependencies
│
├── frontend/
│ ├── app.py ← Streamlit frontend code
│ └── requirements.txt ← Frontend dependencies
│
├── README.md ← Project documentation

---

## 📌 Features
- Predicts disease based on multiple symptoms  
- FastAPI backend for ML model serving  
- Streamlit frontend for easy user interaction  
- Modular project structure for clarity  

---

## 📊 Example Usage
1. Enter symptoms in the Streamlit interface.  
2. The system sends the request to the FastAPI backend.  
3. Backend predicts possible disease and returns result.  

---

## 🚀 Tech Stack
- **Python**  
- **Scikit-learn** (for ML model)  
- **FastAPI** (backend)  
- **Streamlit** (frontend)  

---

## ⚙️ Setup Instructions

### 🔹 Backend (FastAPI)
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload

**### 🔹 Frontend (Streamlit)**
```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py

---

**## 🌟 Future Improvements**
Add more medical datasets for better accuracy
Deploy app using Docker and Cloud (Heroku / AWS / Azure)
Improve model with Deep Learning approaches
Add authentication for secure access

---

## ⚙️ Installation & Setup  

### 1. Clone the Repository  

```bash
git clone https://github.com/<your-username>/symptom-disease-predictor.git
cd symptom-disease-predictor
