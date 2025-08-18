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

## 🚀 Features  

- Predicts diseases from given symptoms.  
- REST API built with **FastAPI**.  
- Interactive **Streamlit UI** for users.  
- Easy to extend with more datasets and models.  

---

## ⚙️ Installation & Setup  

### 1. Clone the Repository  

```bash
git clone https://github.com/<your-username>/symptom-disease-predictor.git
cd symptom-disease-predictor
