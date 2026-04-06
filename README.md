# 🧠 AI-Based Thyroid Disease Prediction & Smart Healthcare Assistant

## 📌 Overview

This project is a full-stack web application that uses Machine Learning to predict thyroid diseases based on user symptoms and clinical inputs. It also acts as a smart healthcare assistant by providing hospital suggestions, doctor recommendations, and downloadable reports.

---

## 🚀 Features

* 🔐 User Authentication (Login/Signup with session management)
* 🎤 Symptom input via Text and Voice
* 🧠 Machine Learning Prediction (XGBoost)
* 📊 Interactive Charts showing prediction confidence
* 🗺️ Nearby Hospital Suggestions (Location-based)
* 👨‍⚕️ Doctor Recommendations based on condition
* 📄 Downloadable Medical Report

---

## 🛠️ Technologies Used

* Python
* Flask
* XGBoost
* Scikit-learn
* Pandas
* HTML, CSS, JavaScript
* Chart.js
* OpenStreetMap API

---

## ⚙️ How It Works

1. User logs into the system
2. Answers symptom-based questions (text/voice)
3. Data is processed and sent to ML model
4. Model predicts disease (Hypothyroid / Hyperthyroid / Normal)
5. Results displayed with confidence score and charts
6. Nearby hospitals and doctors are suggested
7. User can download a detailed report

---

## 📂 Project Structure

```
├── app.py
├── templates/
│   ├── login.html
│   ├── symptoms.html
│   └── result.html
├── static/
├── synthetic_thyroid_dataset.csv
└── README.md
▶️ Installation & Setup
1. Clone the repository:
git clone https://github.com/your-username/thyroid-prediction.git
cd thyroid-prediction
2. Install dependencies:
pip install -r requirements.txt
3. Run the application:
python app.py

4. Open in browser:
http://127.0.0.1:5000/
 📊 Machine Learning Model

* Algorithm: XGBoost Classifier
* Input Features: Age, Sex, T3, TT4, TSH, Symptoms
* Output: Thyroid condition prediction

 🔮 Future Enhancements

* Use real-world medical datasets
* Deploy application (AWS / Render)
* Improve UI/UX design
* Add more diseases prediction

 Acknowledgment

This project was developed as part of learning Machine Learning and Full-Stack Development, focusing on solving real-world healthcare problems.

 📧 Contact

For any queries or collaboration, feel free to connect!


⭐ If you like this project, don't forget to give it a star!
