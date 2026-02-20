# 🎓 ScoreIQ — Academic Performance Intelligence

ScoreIQ is an interactive machine learning web application that predicts student academic performance based on behavioral, academic, and environmental factors.

Built with **Streamlit**, powered by a **Ridge Regression model**, and enhanced with real-time simulation and sensitivity analysis.

---

## 🚀 Live Features

### 🔹 1. Predict
Real-time score prediction based on:
- Study hours
- Attendance rate
- Sleep
- Previous scores
- Tutoring sessions
- Motivation
- Teacher quality
- Family & environmental factors

Displays:
- Predicted score (40–100)
- Grade classification (A+ → D)
- Performance drivers
- Improvement insights
- Potential gain estimate

---

### 🔹 2. Scenario Simulator
"What-if" analysis engine.

Compare baseline profile against:
- +5 / +10 study hours
- Improved attendance
- Better sleep
- Increased tutoring
- Higher motivation
- Fully optimized profile

Includes:
- Delta comparison
- Visual improvement bars
- Baseline reference marker

---

### 🔹 3. Advanced Analytics
Interactive sensitivity analysis:
- Study hours curve
- Attendance curve
- Sleep curve
- Tutoring curve
- 2D heatmap (Study Hours × Attendance)

Helps understand:
- Feature impact magnitude
- Score response behavior
- Diminishing returns zones

---

## 🧠 Model Architecture

- Algorithm: Ridge Regression (α = 1.0)
- Preprocessing: StandardScaler
- Training Data: 5,000 synthetic records (seed = 42)
- Output Range: Clipped between 40–100
- Features: 12 total inputs

Feature Categories:
- Academic behavior
- Lifestyle habits
- Environmental support
- Socioeconomic indicators

---

## 🛠 Tech Stack

- Python
- Streamlit
- scikit-learn
- Plotly
- NumPy
- Joblib

---

## 📊 Input Factors (12)

| Feature | Type |
|---------|------|
| Study Hours | Numeric |
| Attendance | Numeric |
| Sleep | Numeric |
| Previous Score | Numeric |
| Tutoring | Numeric |
| Physical Activity | Numeric |
| Parental Involvement | Ordinal |
| Access to Resources | Ordinal |
| Motivation | Ordinal |
| Internet Access | Binary |
| Teacher Quality | Ordinal |
| Family Income | Ordinal |

---

## 🎯 Project Goals

- Demonstrate applied ML deployment
- Build interactive decision-support system
- Translate regression output into actionable intelligence
- Showcase feature sensitivity and behavioral impact

---

## ▶ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
