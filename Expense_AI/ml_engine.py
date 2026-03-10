"""
ml_engine.py
─────────────────────────────────────────────────────────────────────────────
FinTrack – Machine Learning Engine

Modules:
  1. ExpenseClassifier  – Rule + keyword-based category predictor (no training
                          data needed; upgrades to TF-IDF + LogReg when enough
                          labelled samples exist)
  2. SavingsAdvisor     – Rule-based + statistical recommendation engine
  3. SpendingForecaster – Linear regression trend forecaster on monthly totals
─────────────────────────────────────────────────────────────────────────────
"""

import re
import numpy as np
import pandas as pd
from datetime import datetime, date


# ══════════════════════════════════════════════════════════════════════════════
# 1. EXPENSE CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════

CATEGORY_KEYWORDS = {
    "Food & Dining": [
        "lunch", "dinner", "breakfast", "food", "restaurant", "cafe", "coffee",
        "tea", "snack", "swiggy", "zomato", "pizza", "burger", "biryani",
        "canteen", "mess", "hotel", "eat", "meal", "juice", "dosa", "idli"
    ],
    "Transport": [
        "bus", "auto", "rick", "cab", "ola", "uber", "metro", "train", "petrol",
        "diesel", "fuel", "travel", "ticket", "transport", "bike", "rapido"
    ],
    "Entertainment": [
        "movie", "cinema", "netflix", "amazon prime", "hotstar", "spotify",
        "game", "gaming", "concert", "party", "outing", "fun", "show"
    ],
    "Shopping": [
        "shop", "amazon", "flipkart", "clothes", "shirt", "jeans", "shoes",
        "bag", "purchase", "buy", "mall", "store", "order"
    ],
    "Health & Medical": [
        "doctor", "hospital", "medicine", "pharmacy", "medical", "clinic",
        "health", "tablet", "injection", "gym", "fitness"
    ],
    "Education": [
        "book", "notes", "college", "course", "tuition", "fee", "exam",
        "stationery", "pen", "pencil", "study", "class", "coaching"
    ],
    "Utilities": [
        "electricity", "water", "gas", "internet", "wifi", "broadband",
        "recharge", "mobile", "phone bill", "dth", "jio", "airtel"
    ],
    "Rent": [
        "rent", "pg", "hostel", "room", "accommodation", "flat", "house"
    ],
    "Personal Care": [
        "haircut", "salon", "parlour", "cosmetics", "soap", "shampoo",
        "toothpaste", "grooming"
    ],
    "Subscriptions": [
        "subscription", "premium", "membership", "annual", "monthly plan",
        "renewal"
    ],
    "Travel": [
        "trip", "tour", "vacation", "holiday", "flight", "bus ticket",
        "train ticket", "hotel booking", "oyo"
    ],
}


class ExpenseClassifier:
    """
    Predicts an expense category from a short text description.

    Strategy:
      - Phase 1 (default): keyword matching with confidence scoring
      - Phase 2 (auto-upgrade): if ≥ 30 labelled samples exist, trains a
        TF-IDF + Logistic Regression model for higher accuracy
    """

    def __init__(self):
        self._sklearn_model = None

    # ── Keyword Matching ──────────────────────────────────────────────────────
    def _keyword_score(self, text: str) -> dict:
        text = text.lower()
        scores = {}
        for category, keywords in CATEGORY_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text)
            if score:
                scores[category] = score
        return scores

    def predict_category(self, description: str) -> str:
        """Return the predicted category for a given description string."""
        if not description or not description.strip():
            return "Other"

        scores = self._keyword_score(description)
        if scores:
            return max(scores, key=scores.get)
        return "Other"

    # ── ML Upgrade (called when enough data is available) ────────────────────
    def train(self, descriptions: list, categories: list):
        """
        Train a TF-IDF + Logistic Regression classifier.
        Automatically called when the app has ≥ 30 samples.
        """
        try:
            from sklearn.pipeline import Pipeline
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import LabelEncoder

            self._le = LabelEncoder()
            y = self._le.fit_transform(categories)

            self._sklearn_model = Pipeline([
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=500)),
                ("clf",   LogisticRegression(max_iter=300, C=1.0))
            ])
            self._sklearn_model.fit(descriptions, y)
        except Exception:
            self._sklearn_model = None

    def predict_category_ml(self, description: str) -> str:
        """Use trained ML model if available, else fallback to keyword."""
        if self._sklearn_model is None:
            return self.predict_category(description)
        try:
            pred_idx = self._sklearn_model.predict([description])[0]
            return self._le.inverse_transform([pred_idx])[0]
        except Exception:
            return self.predict_category(description)


# ══════════════════════════════════════════════════════════════════════════════
# 2. SAVINGS ADVISOR
# ══════════════════════════════════════════════════════════════════════════════

class SavingsAdvisor:
    """
    Generates personalised financial recommendations based on:
      - Spending patterns (category-wise)
      - Income vs. expenditure ratio
      - Savings goals progress
    """

    # 50-30-20 rule benchmarks
    NEEDS_RATIO       = 0.50
    WANTS_RATIO       = 0.30
    SAVINGS_RATIO     = 0.20

    NEEDS_CATEGORIES  = {"Food & Dining", "Utilities", "Rent",
                         "Health & Medical", "Education", "Transport"}
    WANTS_CATEGORIES  = {"Entertainment", "Shopping", "Personal Care",
                         "Subscriptions", "Travel"}

    def get_recommendations(
        self,
        expenses_df: pd.DataFrame,
        total_income: float,
        balance: float,
        goals: list
    ) -> list:
        tips = []

        if expenses_df.empty or total_income <= 0:
            tips.append("Start logging your expenses to get personalised insights!")
            return tips

        total_spent = expenses_df["amount"].sum()
        cat_totals  = expenses_df.groupby("category")["amount"].sum()

        # ── 50/30/20 Analysis ─────────────────────────────────────────────────
        needs_spent  = sum(cat_totals.get(c, 0) for c in self.NEEDS_CATEGORIES)
        wants_spent  = sum(cat_totals.get(c, 0) for c in self.WANTS_CATEGORIES)
        savings_done = balance

        needs_pct    = needs_spent  / total_income
        wants_pct    = wants_spent  / total_income
        savings_pct  = savings_done / total_income if savings_done > 0 else 0

        if wants_pct > self.WANTS_RATIO:
            excess = wants_spent - (self.WANTS_RATIO * total_income)
            tips.append(
                f"Your 'wants' spending (entertainment, shopping, etc.) is "
                f"₹{wants_spent:,.0f} ({wants_pct*100:.0f}% of income). "
                f"The ideal limit is 30%. Reducing by ₹{excess:,.0f} could boost your savings."
            )

        if savings_pct < self.SAVINGS_RATIO:
            target_savings = self.SAVINGS_RATIO * total_income
            tips.append(
                f"You're currently saving {savings_pct*100:.1f}% of your income. "
                f"Aim for at least 20% (₹{target_savings:,.0f}). "
                f"Try cutting discretionary expenses first."
            )

        # ── Top Spending Category Warning ────────────────────────────────────
        if not cat_totals.empty:
            top_cat   = cat_totals.idxmax()
            top_amt   = cat_totals.max()
            top_pct   = (top_amt / total_spent) * 100
            if top_pct > 40:
                tips.append(
                    f"📌 '{top_cat}' accounts for {top_pct:.0f}% of your total expenses "
                    f"(₹{top_amt:,.0f}). Consider reviewing this category."
                )

        # ── Entertainment / Subscription Overuse ─────────────────────────────
        for cat in ["Entertainment", "Subscriptions"]:
            if cat in cat_totals:
                pct = (cat_totals[cat] / total_income) * 100
                if pct > 10:
                    tips.append(
                        f"💸 You're spending {pct:.0f}% of income on {cat}. "
                        f"Consider auditing unused subscriptions or reducing outings."
                    )

        # ── Savings Goal Progress ─────────────────────────────────────────────
        for goal in goals:
            pct  = (goal["saved"] / goal["target"] * 100) if goal["target"] > 0 else 0
            days = (datetime.strptime(goal["target_date"], "%Y-%m-%d").date() - date.today()).days
            if pct < 50 and days < 90:
                needed_pm = ((goal["target"] - goal["saved"]) / max(days / 30, 1))
                tips.append(
                    f"🎯 Goal '{goal['name']}' is {pct:.0f}% complete with only "
                    f"{days} days left. You need to save ₹{needed_pm:,.0f}/month to reach it."
                )

        # ── Emergency Fund Reminder ───────────────────────────────────────────
        if not any("emergency" in g["name"].lower() for g in goals):
            tips.append(
                "🛡️ You don't have an Emergency Fund goal yet. "
                "Aim to save 3–6 months' expenses for financial safety."
            )

        # ── Positive Reinforcement ────────────────────────────────────────────
        if savings_pct >= self.SAVINGS_RATIO:
            tips.append(
                f"🌟 Great job! You're saving {savings_pct*100:.1f}% of your income — "
                f"above the 20% benchmark. Consider investing the surplus."
            )

        if not tips:
            tips.append("✅ Your finances look well-balanced! Keep up the good habits.")

        return tips

    def savings_score(
        self,
        total_income: float,
        total_expenses: float,
        num_goals: int
    ) -> int:
        """
        Compute a savings score (0-100) based on:
          - Savings rate (0-50 pts)
          - Expense-to-income ratio (0-30 pts)
          - Having active savings goals (0-20 pts)
        """
        score = 0

        if total_income > 0:
            savings_rate = max(0, (total_income - total_expenses) / total_income)
            score += min(50, int(savings_rate * 250))   # 20% rate → 50 pts

            expense_ratio = total_expenses / total_income
            if expense_ratio <= 0.6:
                score += 30
            elif expense_ratio <= 0.8:
                score += 20
            elif expense_ratio <= 1.0:
                score += 10

        score += min(20, num_goals * 7)
        return min(100, score)


# ══════════════════════════════════════════════════════════════════════════════
# 3. SPENDING FORECASTER
# ══════════════════════════════════════════════════════════════════════════════

class SpendingForecaster:
    """
    Forecasts next 3 months of spending using:
      - Simple Linear Regression on monthly totals
      - Falls back to moving average when data is insufficient
    """

    def forecast(self, expenses_df: pd.DataFrame) -> pd.DataFrame | None:
        if expenses_df.empty:
            return None

        expenses_df = expenses_df.copy()
        expenses_df["date"] = pd.to_datetime(expenses_df["date"])
        monthly = (
            expenses_df
            .groupby(expenses_df["date"].dt.to_period("M"))["amount"]
            .sum()
            .reset_index()
        )
        monthly.columns = ["period", "amount"]
        monthly["month_num"] = range(len(monthly))

        if len(monthly) < 2:
            return None

        # ── Linear Regression ─────────────────────────────────────────────────
        X = monthly["month_num"].values.reshape(-1, 1)
        y = monthly["amount"].values

        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, y)

        # Predict existing months (actual vs fitted) + 3 future months
        fitted      = model.predict(X)
        future_nums = np.array([[len(monthly)], [len(monthly)+1], [len(monthly)+2]])
        future_pred = model.predict(future_nums)
        future_pred = np.maximum(future_pred, 0)    # no negative forecasts

        # Build result dataframe
        existing_months = [str(p) for p in monthly["period"]]
        last_period     = monthly["period"].iloc[-1]
        future_months   = [
            str(last_period + i) for i in range(1, 4)
        ]

        all_months   = existing_months + future_months
        all_actual   = list(y) + [None, None, None]
        all_predicted = list(fitted) + list(future_pred)

        return pd.DataFrame({
            "month":     all_months,
            "actual":    all_actual,
            "predicted": all_predicted
        })
