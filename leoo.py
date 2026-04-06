from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from flask import Flask, request, jsonify, render_template, redirect, session, send_file
import base64
from datetime import datetime

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd

app = Flask(__name__)
app.secret_key = "secret123"

users = {}
latest_result = {}

# ---------------- AUTH ----------------
@app.route("/auth", methods=["POST"])
def auth():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")
    name = data.get("name")
    is_login = data.get("isLogin")

    if is_login:
        if email in users and users[email]["password"] == password:
            session["user"] = email
            return jsonify({"success": True, "redirect": "/symptoms"})
        return jsonify({"success": False})
    else:
        if email in users:
            return jsonify({"success": False})
        users[email] = {"name": name, "password": password}
        session["user"] = email
        return jsonify({"success": True, "redirect": "/symptoms"})

# ---------------- LOAD DATA ----------------
df = pd.read_csv("synthetic_thyroid_dataset.csv")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

feature_cols = [
    "age","sex","t3","tt4","tsh","hair_loss",
    "cold_intolerance","heat_intolerance","palpitations",
    "pregnant","thyroid_surgery"
]

target_col = df.columns[-1]

for col in feature_cols:
    if col == "sex":
        df[col] = df[col].astype(str).map({"male":1,"female":0}).fillna(0)
    elif col in ["age","t3","tt4","tsh"]:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    else:
        df[col] = df[col].astype(str).str.lower().map({"yes":1,"no":0}).fillna(0)

X = df[feature_cols]
y = df[target_col]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

model = XGBClassifier(n_estimators=200, learning_rate=0.1)
model.fit(X, y_encoded)

# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return render_template("login.html")

@app.route("/symptoms")
def symptoms():
    if "user" not in session:
        return redirect("/")

    questions = {
        "age": "What is your age?",
        "sex": "Select your sex",
        "t3": "Enter your T3 value (optional)",
        "tt4": "Enter your TT4 value (optional)",
        "tsh": "Enter your TSH value (optional)",
        "hair_loss": "Do you have hair loss?",
        "cold_intolerance": "Do you feel cold often?",
        "heat_intolerance": "Do you feel hot often?",
        "palpitations": "Do you feel heart racing?",
        "pregnant": "Are you pregnant?",
        "thyroid_surgery": "Had thyroid surgery?"
    }

    return render_template("symptoms.html", features=feature_cols, questions=questions)

@app.route("/result")
def result_page():
    return render_template("result.html", data=latest_result)

# ---------------- SUBMIT ----------------
@app.route("/submit-quiz", methods=["POST"])
def submit():
    global latest_result

    data = request.get_json()
    input_data = []

    for col in feature_cols:
        val = data.get(col)

        if col == "age":
            input_data.append(int(val) if val else 0)
        elif col == "sex":
            input_data.append(1 if str(val).lower()=="male" else 0)
        elif col in ["t3","tt4","tsh"]:
            input_data.append(float(val) if val else 0)
        else:
            input_data.append(1 if str(val).lower()=="yes" else 0)

    pred = model.predict([input_data])[0]
    probs = model.predict_proba([input_data])[0]
    result = le.inverse_transform([pred])[0]

    scores = [float(p*100) for p in probs]
    labels = [str(l) for l in le.classes_]

    doctors = {
        "Hypothyroid": ["Endocrinologist", "General Physician"],
        "Hyperthyroid": ["Endocrinologist", "Cardiologist"],
        "Normal": ["Healthy"]
    }

    latest_result = {
        "result": str(result),
        "confidence": float(round(max(scores),2)),
        "risk": "High" if max(scores)>80 else "Medium",
        "labels": labels,
        "scores": scores,
        "doctors": doctors.get(result, [])
    }

    return jsonify({"redirect": "/result"})

# ---------------- PDF DOWNLOAD ----------------
@app.route("/download-report", methods=["POST"])
def download_report():
    global latest_result

    data = request.get_json()
    chart_base64 = data.get("chart")

    user_email = session.get("user")
    user_name = users.get(user_email, {}).get("name", "User")

    file_path = "report.pdf"

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(file_path)

    content = []

    # TITLE
    content.append(Paragraph("Thyroid Prediction Report", styles["Title"]))
    content.append(Spacer(1, 20))

    # USER DETAILS
    content.append(Paragraph(f"Name: {user_name}", styles["Normal"]))
    content.append(Paragraph(f"Date: {datetime.now().strftime('%d-%m-%Y %H:%M')}", styles["Normal"]))
    content.append(Spacer(1, 20))

    # RESULT
    content.append(Paragraph(f"Result: {latest_result.get('result')}", styles["Heading2"]))
    content.append(Spacer(1, 10))
    content.append(Paragraph(f"Confidence: {latest_result.get('confidence')}%", styles["Normal"]))
    content.append(Paragraph(f"Risk Level: {latest_result.get('risk')}", styles["Normal"]))
    content.append(Spacer(1, 20))

    # CHART IMAGE
    if chart_base64:
        img_data = base64.b64decode(chart_base64.split(",")[1])
        with open("chart.png", "wb") as f:
            f.write(img_data)

        content.append(Paragraph("Prediction Chart:", styles["Heading2"]))
        content.append(Spacer(1, 10))
        content.append(Image("chart.png", width=400, height=300))
        content.append(Spacer(1, 20))

    # PROBABILITIES
    content.append(Paragraph("Probabilities:", styles["Heading2"]))
    content.append(Spacer(1, 10))

    for label, score in zip(latest_result.get("labels", []), latest_result.get("scores", [])):
        content.append(Paragraph(f"{label}: {round(score,2)}%", styles["Normal"]))

    content.append(Spacer(1, 20))

    # DOCTORS
    content.append(Paragraph("Recommended Doctors:", styles["Heading2"]))
    content.append(Spacer(1, 10))

    for d in latest_result.get("doctors", []):
        content.append(Paragraph(f"- {d}", styles["Normal"]))

    doc.build(content)

    return send_file(file_path, as_attachment=True)

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)