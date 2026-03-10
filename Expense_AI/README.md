# 💰 FinTrack – AI Personal Expense & Savings Advisor

> An AI-powered personal finance management system built with Python, Streamlit, and Scikit-learn.

---

## 📌 Project Info

| Field        | Details                        |
|--------------|--------------------------------|
| Domain       | FinTech / Personal Finance     |
| Method       | Machine Learning               |
| Technologies | Python, Streamlit, Scikit-learn, Plotly |
| Type         | Mini Project (CSE - AI & ML)   |

---

## 🚀 Setup & Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 🧠 ML Components

### 1. `ExpenseClassifier` (ml_engine.py)
- **Phase 1**: Keyword-based category prediction (works with 0 samples)
- **Phase 2**: Auto-upgrades to TF-IDF + Logistic Regression when ≥30 labelled samples exist
- **Input**: Expense description text
- **Output**: Predicted category (Food, Transport, Entertainment, etc.)

### 2. `SavingsAdvisor` (ml_engine.py)
- Implements the **50-30-20 budgeting rule**
- Analyses spending patterns across categories
- Generates personalised savings recommendations
- Computes a **Savings Score (0–100)**

### 3. `SpendingForecaster` (ml_engine.py)
- Uses **Linear Regression** on monthly spending totals
- Forecasts next 3 months of spending
- Falls back to moving average when data is insufficient

---

## 📁 Project Structure

```
fintrack/
├── app.py              # Main Streamlit UI
├── ml_engine.py        # ML models (Classifier, Advisor, Forecaster)
├── requirements.txt    # Python dependencies
├── fintrack_data.json  # Auto-created data store (JSON)
└── README.md
```

---

## 🖥️ Features

| Feature                | Description                                     |
|------------------------|-------------------------------------------------|
| 💸 Expense Tracking    | Log daily expenses with auto-category suggestion|
| 💼 Income Tracking     | Record multiple income sources                  |
| 🎯 Savings Goals       | Set goals, track progress, add deposits         |
| 🧾 Bill Reminders      | Add recurring bills, get due-date alerts        |
| 🤖 AI Advisor          | Personalised tips + spending forecast           |
| 📊 Dashboard           | Charts, pie charts, trend analysis              |

---

## 📊 Streamlit Pages

1. **🏠 Dashboard** – Overview metrics, monthly chart, upcoming bills
2. **💸 Expenses** – Add/view expenses with AI category suggestion
3. **💼 Income** – Log income from multiple sources
4. **🎯 Savings Goals** – Create goals and deposit savings
5. **🧾 Bills** – Manage recurring bills with reminders
6. **🤖 AI Advisor** – ML recommendations + 3-month spending forecast

---

## 🎓 Academic Details

- **Project Title**: AI-Based Personal Expense Tracking and Savings Recommendation System
- **Short Name**: FinTrack
- **Branch**: CSE (Artificial Intelligence & Machine Learning)
- **Tech Stack**: Python · Streamlit · Scikit-learn · Plotly · Pandas
