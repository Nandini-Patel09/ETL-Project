import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date, timedelta
import json, os, io
from ml_engine import ExpenseClassifier, SavingsAdvisor, SpendingForecaster

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="FinTrack – AI Expense Advisor",
                   page_icon="💰", layout="wide",
                   initial_sidebar_state="expanded")

# ─── Data Storage ──────────────────────────────────────────────────────────────
DATA_FILE = "fintrack_data.json"

def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE) as f:
            d = json.load(f)
        # ── Fix: deduplicate savings_goals on load ──────────────────────────
        seen, unique = set(), []
        for g in d.get("savings_goals", []):
            key = (g["name"].strip().lower(), g["target"], g["target_date"])
            if key not in seen:
                seen.add(key)
                unique.append(g)
        d["savings_goals"] = unique
        # ── Ensure budget_limits key exists ────────────────────────────────
        d.setdefault("budget_limits", {})
        return d
    return {"expenses":[],"incomes":[],"savings_goals":[],"bills":[],"budget_limits":{}}

def save_data(d):
    with open(DATA_FILE,"w") as f:
        json.dump(d, f, indent=2, default=str)

# ─── Session State (must come BEFORE CSS so theme is available) ────────────────
if "data"         not in st.session_state: st.session_state.data         = load_data()
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "editing_bill" not in st.session_state: st.session_state.editing_bill = None
if "theme"        not in st.session_state: st.session_state.theme        = "dark"

# ─── CSS (theme-aware) ─────────────────────────────────────────────────────────
_dark = st.session_state.theme == "dark"
_bg        = "linear-gradient(135deg,#0f0c29,#1a1a2e,#16213e)" if _dark else "linear-gradient(135deg,#f0f2f6,#e8ecf3,#dde3ed)"
_text      = "#e0e0e0"   if _dark else "#1a1a2e"
_sidebar   = "rgba(255,255,255,0.04)" if _dark else "rgba(255,255,255,0.7)"
_metric    = "rgba(255,255,255,0.06)" if _dark else "rgba(255,255,255,0.8)"
_metric_b  = "rgba(255,255,255,0.1)"  if _dark else "rgba(108,99,255,0.15)"
_input_bg  = "rgba(255,255,255,.07)"  if _dark else "rgba(255,255,255,0.9)"
_input_b   = "rgba(255,255,255,.15)"  if _dark else "rgba(108,99,255,0.3)"
_tab_bg    = "rgba(255,255,255,.04)"  if _dark else "rgba(255,255,255,0.5)"
_tab_col   = "#aaa"      if _dark else "#555"
_chat_bot  = "rgba(255,255,255,.07)"  if _dark else "rgba(255,255,255,0.85)"
_chat_botb = "rgba(255,255,255,.12)"  if _dark else "rgba(108,99,255,0.2)"
_chat_meta = "#888"      if _dark else "#666"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&display=swap');
html,body,[class*="css"]{{font-family:'Sora',sans-serif;}}
.stApp{{background:{_bg};color:{_text};}}
[data-testid="stSidebar"]{{background:{_sidebar};border-right:1px solid rgba(255,255,255,0.08);}}
[data-testid="metric-container"]{{background:{_metric};border:1px solid {_metric_b};border-radius:16px;padding:16px;backdrop-filter:blur(10px);}}
h1,h2,h3{{font-family:'Sora',sans-serif;font-weight:700;color:{_text};}}
.stButton>button{{background:linear-gradient(135deg,#6c63ff,#4ecdc4);color:white;border:none;border-radius:10px;font-weight:600;padding:.5rem 1.5rem;transition:all .3s ease;}}
.stButton>button:hover{{transform:translateY(-2px);box-shadow:0 8px 25px rgba(108,99,255,.4);}}
.stTextInput>div>div>input,.stNumberInput>div>div>input,.stSelectbox>div>div{{background:{_input_bg};border:1px solid {_input_b};border-radius:10px;color:{_text};}}
.stTabs [data-baseweb="tab-list"]{{gap:8px;background:{_tab_bg};border-radius:12px;padding:4px;}}
.stTabs [data-baseweb="tab"]{{border-radius:8px;color:{_tab_col};font-weight:600;}}
.stTabs [aria-selected="true"]{{background:linear-gradient(135deg,#6c63ff,#4ecdc4);color:white!important;}}
.goal-card{{background:rgba(78,205,196,.1);border:1px solid rgba(78,205,196,.25);border-radius:12px;padding:12px 16px;margin:6px 0;}}
.chat-user{{background:linear-gradient(135deg,#6c63ff,#4a44cc);border-radius:18px 18px 4px 18px;padding:12px 18px;margin:8px 0 4px 20%;color:white;font-size:.95rem;}}
.chat-bot{{background:{_chat_bot};border:1px solid {_chat_botb};border-radius:18px 18px 18px 4px;padding:12px 18px;margin:8px 20% 4px 0;color:{_text};font-size:.95rem;}}
.chat-meta{{font-size:.75rem;color:{_chat_meta};margin:2px 8px 8px;}}
.budget-ok{{color:#4ecdc4;font-weight:700;}}
.budget-warn{{color:#f7b731;font-weight:700;}}
.budget-over{{color:#ff6b6b;font-weight:700;}}
</style>
""", unsafe_allow_html=True)

data = st.session_state.data

# ─── Global computed variables ─────────────────────────────────────────────────
expenses_df = (pd.DataFrame(data["expenses"]) if data["expenses"]
               else pd.DataFrame(columns=["amount","category","date","description","note"]))
income_df   = (pd.DataFrame(data["incomes"])  if data["incomes"]
               else pd.DataFrame(columns=["amount","source","date","note"]))

total_income   = float(income_df["amount"].sum())   if not income_df.empty else 0.0
total_expenses = float(expenses_df["amount"].sum()) if not expenses_df.empty else 0.0
balance        = total_income - total_expenses
savings_rate   = (balance / total_income * 100)     if total_income > 0 else 0.0

CATEGORIES = ["Food & Dining","Transport","Entertainment","Shopping",
              "Health & Medical","Education","Utilities","Rent",
              "Personal Care","Subscriptions","Travel","Other"]

# ─── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 💰 FinTrack")
st.sidebar.markdown("*AI Personal Finance Advisor*")

# ── Theme Toggle ──────────────────────────────────────────────────────────────
theme_icon = "☀️ Light Mode" if st.session_state.theme == "dark" else "🌙 Dark Mode"
if st.sidebar.button(theme_icon, use_container_width=True):
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
    st.rerun()

st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate",
    ["🏠 Dashboard","💸 Expenses","💼 Income","🎯 Savings Goals","🧾 Bills","💰 Budget Limits","🤖 AI Chatbot"],
    label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.metric("💵 Balance",  f"Rs.{balance:,.0f}")
st.sidebar.metric("📥 Income",   f"Rs.{total_income:,.0f}")
st.sidebar.metric("📤 Expenses", f"Rs.{total_expenses:,.0f}")

# ── CSV Export ────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("### 📥 Export Data")

def _to_csv(df): return df.to_csv(index=False).encode("utf-8")

if not expenses_df.empty:
    st.sidebar.download_button("⬇️ Export Expenses CSV",
        data=_to_csv(expenses_df), file_name="fintrack_expenses.csv",
        mime="text/csv", use_container_width=True)
if not income_df.empty:
    st.sidebar.download_button("⬇️ Export Income CSV",
        data=_to_csv(income_df), file_name="fintrack_income.csv",
        mime="text/csv", use_container_width=True)
if data["savings_goals"]:
    goals_df = pd.DataFrame(data["savings_goals"])
    st.sidebar.download_button("⬇️ Export Goals CSV",
        data=_to_csv(goals_df), file_name="fintrack_goals.csv",
        mime="text/csv", use_container_width=True)

# ─── Local fallback chatbot ────────────────────────────────────────────────────
def _local_chatbot(q, income, expenses, bal, srate, cats, tips, d):
    q = q.lower()
    if any(w in q for w in ["overspend","most","highest","top","where"]):
        return f"Category breakdown:\n{cats}\n\nFocus on reducing the largest category first!"
    if any(w in q for w in ["save","saving","tip","advice","suggest","how"]):
        return "Personalised tips:\n" + "\n".join(tips[:3])
    if any(w in q for w in ["score","health","status","rate"]):
        score = SavingsAdvisor().savings_score(income, expenses, len(d["savings_goals"]))
        return (f"Savings Score: {score}/100\nRate: {srate:.1f}%\nBalance: Rs.{bal:,.0f}\n"
                + ("Excellent! Keep it up!" if score >= 70 else "Keep going, small changes add up!"))
    if any(w in q for w in ["goal","target","reach","when"]):
        if d["savings_goals"]:
            g = d["savings_goals"][0]
            pct = (g["saved"]/g["target"]*100) if g["target"] > 0 else 0
            return f"Goal '{g['name']}': {pct:.0f}% complete\nRs.{g['saved']:,.0f}/Rs.{g['target']:,.0f}"
        return "No goals yet. Go to Savings Goals to create one!"
    if any(w in q for w in ["bill","due","pay"]):
        if d["bills"]:
            today = date.today()
            lines = [f"- {b['name']}: Rs.{b['amount']:,.0f} due in "
                     f"{(datetime.strptime(b['due_date'],'%Y-%m-%d').date()-today).days} days"
                     for b in d["bills"]]
            return "Bills:\n" + "\n".join(lines)
        return "No bills yet."
    if any(w in q for w in ["balance","left","remain","how much"]):
        return f"Balance: Rs.{bal:,.0f} (Income Rs.{income:,.0f} minus Expenses Rs.{expenses:,.0f})"
    return "Ask me: 'Where am I overspending?', 'Give saving tips', 'My savings score', 'Bill status'"


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.markdown("# 🏠 Dashboard")
    st.markdown("Your financial snapshot at a glance.")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Income",   f"Rs.{total_income:,.0f}")
    c2.metric("Total Expenses", f"Rs.{total_expenses:,.0f}",
              delta=f"-Rs.{total_expenses:,.0f}", delta_color="inverse")
    c3.metric("Net Balance",    f"Rs.{balance:,.0f}")
    c4.metric("Savings Rate",   f"{savings_rate:.1f}%")
    st.markdown("---")

    col_l, col_r = st.columns([3,2])
    with col_l:
        st.markdown("### Monthly Spending Trend")
        if not expenses_df.empty:
            edf = expenses_df.copy()
            edf["date"] = pd.to_datetime(edf["date"])
            monthly = edf.groupby(edf["date"].dt.strftime("%Y-%m"))["amount"].sum().reset_index()
            monthly.columns = ["Month","Amount"]
            fig = px.bar(monthly, x="Month", y="Amount",
                         color="Amount", color_continuous_scale=["#6c63ff","#4ecdc4"])
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",
                              font_color="#e0e0e0",showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No expense data yet.")
    with col_r:
        st.markdown("### By Category")
        if not expenses_df.empty:
            cat_data = expenses_df.groupby("category")["amount"].sum().reset_index()
            fig2 = px.pie(cat_data, values="amount", names="category",
                          hole=0.45, color_discrete_sequence=px.colors.sequential.Plasma_r)
            fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",
                               font_color="#e0e0e0")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No data yet.")

    st.markdown("### Upcoming Bills")
    if data["bills"]:
        today = date.today()
        shown = 0
        for bill in data["bills"]:
            due = datetime.strptime(bill["due_date"],"%Y-%m-%d").date()
            days_left = (due - today).days
            if days_left <= 7:
                icon = "🔴" if days_left <= 2 else "🟡"
                st.warning(f"{icon} **{bill['name']}** — Rs.{bill['amount']:,.0f} due in **{days_left} days** ({bill['due_date']})")
                shown += 1
        if shown == 0:
            st.success("No bills due in the next 7 days.")
    else:
        st.info("No bills added yet.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EXPENSES — Voice Entry via st.markdown (NOT iframe)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💸 Expenses":
    st.markdown("# 💸 Expense Tracker")
    tab1, tab2 = st.tabs(["➕ Add Expense", "📋 View & Analyse"])

    with tab1:
        # ── VOICE ENTRY ── rendered directly in page (not inside iframe) ──────
        st.markdown("### 🎙️ Voice Entry")
        st.caption("Click **Start Recording**, speak your expense clearly, then click **Use This Text** to fill the Description field. Works in **Chrome / Edge** only.")

        st.markdown("""
<div id="voiceWidget" style="background:rgba(78,205,196,0.08);border:2px dashed rgba(78,205,196,0.5);
     border-radius:14px;padding:20px 24px;margin-bottom:16px;">
  <div id="voiceDisplay" style="min-height:40px;font-size:1.05rem;font-weight:600;
       color:#4ecdc4;margin-bottom:14px;word-break:break-word;text-align:center;
       font-style:italic;">Your spoken text will appear here...</div>
  <div style="display:flex;gap:10px;justify-content:center;flex-wrap:wrap;">
    <button id="voiceBtn" onclick="toggleVoice()"
      style="background:linear-gradient(135deg,#6c63ff,#4ecdc4);color:white;border:none;
             border-radius:10px;padding:10px 26px;font-size:0.95rem;cursor:pointer;font-weight:600;">
      🎤 Start Recording
    </button>
    <button id="fillBtn" onclick="fillDescription()" disabled
      style="background:rgba(255,255,255,0.1);color:#ccc;border:1px solid rgba(255,255,255,0.2);
             border-radius:10px;padding:10px 22px;font-size:0.95rem;cursor:not-allowed;font-weight:600;">
      ✅ Use This Text
    </button>
  </div>
  <p id="voiceStatus" style="margin-top:12px;color:#aaa;font-size:0.82rem;text-align:center;min-height:18px;"></p>
</div>

<script>
(function() {
  var recog = null;
  var listening = false;
  var capturedText = "";

  window.toggleVoice = function() {
    var SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) {
      document.getElementById('voiceStatus').innerText = "Not supported — please open in Google Chrome or Edge.";
      return;
    }
    if (listening) { recog.stop(); return; }

    capturedText = "";
    recog = new SR();
    recog.lang = 'en-IN';
    recog.continuous = true;
    recog.interimResults = true;

    recog.onstart = function() {
      listening = true;
      document.getElementById('voiceBtn').innerText = "⏹ Stop";
      document.getElementById('voiceBtn').style.background = "linear-gradient(135deg,#ff6b6b,#ee5a24)";
      document.getElementById('voiceStatus').innerText = "Listening... speak your expense now";
      document.getElementById('voiceDisplay').innerText = "";
      document.getElementById('fillBtn').disabled = true;
      document.getElementById('fillBtn').style.cursor = "not-allowed";
      document.getElementById('fillBtn').style.color = "#ccc";
    };

    recog.onresult = function(e) {
      var interim = "";
      for (var i = e.resultIndex; i < e.results.length; i++) {
        if (e.results[i].isFinal) capturedText += e.results[i][0].transcript + " ";
        else interim = e.results[i][0].transcript;
      }
      document.getElementById('voiceDisplay').innerText = (capturedText + interim).trim();
    };

    recog.onerror = function(e) {
      document.getElementById('voiceStatus').innerText = "Error: " + e.error + " — allow mic access and try again.";
      listening = false;
      document.getElementById('voiceBtn').innerText = "🎤 Start Recording";
      document.getElementById('voiceBtn').style.background = "linear-gradient(135deg,#6c63ff,#4ecdc4)";
    };

    recog.onend = function() {
      listening = false;
      document.getElementById('voiceBtn').innerText = "🎤 Start Recording";
      document.getElementById('voiceBtn').style.background = "linear-gradient(135deg,#6c63ff,#4ecdc4)";
      if (capturedText.trim()) {
        document.getElementById('voiceStatus').innerText = "Done! Click 'Use This Text' to fill the description below.";
        document.getElementById('fillBtn').disabled = false;
        document.getElementById('fillBtn').style.cursor = "pointer";
        document.getElementById('fillBtn').style.color = "white";
        document.getElementById('fillBtn').style.background = "linear-gradient(135deg,#4ecdc4,#2ecc71)";
      } else {
        document.getElementById('voiceStatus').innerText = "Nothing captured — try again.";
      }
    };
    recog.start();
  };

  window.fillDescription = function() {
    var text = capturedText.trim();
    if (!text) return;
    // Target the first text input (Description field) in the Streamlit app
    var inputs = document.querySelectorAll('input[type="text"]');
    for (var i = 0; i < inputs.length; i++) {
      var nativeSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
      nativeSetter.call(inputs[i], text);
      inputs[i].dispatchEvent(new Event('input', { bubbles: true }));
      document.getElementById('voiceStatus').innerText = "Text filled in Description field below!";
      break;
    }
  };
})();
</script>
""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 📝 Expense Details")
        col1, col2 = st.columns(2)
        with col1:
            exp_desc   = st.text_input("Description *", placeholder="e.g., Lunch at college canteen")
            exp_amount = st.number_input("Amount (Rs.) *", min_value=0.01, step=10.0,
                                          value=None, placeholder="Enter amount")
            exp_date   = st.date_input("Date", value=date.today())
        with col2:
            exp_cat  = st.selectbox("Category", CATEGORIES)
            exp_note = st.text_area("Note (optional)", placeholder="Any extra details...")
            if exp_desc and exp_desc.strip():
                clf       = ExpenseClassifier()
                suggested = clf.predict_category(exp_desc)
                st.info(f"🤖 AI suggests: **{suggested}**")

        if st.button("💾 Save Expense", use_container_width=True):
            if not exp_desc or not exp_desc.strip():
                st.error("Please enter a description.")
            elif exp_amount is None or exp_amount <= 0:
                st.error("Please enter a valid amount greater than 0.")
            else:
                data["expenses"].append({
                    "description": exp_desc.strip(),
                    "amount":      float(exp_amount),
                    "category":    exp_cat,
                    "date":        str(exp_date),
                    "note":        exp_note or ""
                })
                save_data(data)
                st.session_state.data = data
                st.success(f"Saved Rs.{exp_amount:,.0f} for '{exp_desc.strip()}'!")
                st.rerun()

    with tab2:
        if not expenses_df.empty:
            edf = expenses_df.copy()
            edf["date"] = pd.to_datetime(edf["date"])
            c1,c2 = st.columns(2)
            with c1:
                filter_cat = st.selectbox("Filter by Category", ["All"] + CATEGORIES)
            with c2:
                months_avail = sorted(edf["date"].dt.strftime("%Y-%m").unique().tolist(), reverse=True)
                filter_month = st.selectbox("Filter by Month", ["All"] + months_avail)
            filtered = edf.copy()
            if filter_cat != "All":
                filtered = filtered[filtered["category"] == filter_cat]
            if filter_month != "All":
                filtered = filtered[filtered["date"].dt.strftime("%Y-%m") == filter_month]
            st.dataframe(filtered[["date","description","category","amount"]].sort_values("date",ascending=False),
                         use_container_width=True, hide_index=True)
            st.markdown(f"**Total: Rs.{filtered['amount'].sum():,.0f}**")
            if not filtered.empty:
                cat_sum = filtered.groupby("category")["amount"].sum().reset_index()
                fig = px.bar(cat_sum, x="category", y="amount",
                             color="amount", color_continuous_scale="Purples")
                fig.update_layout(plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",font_color="#e0e0e0")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No expenses recorded yet.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: INCOME
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💼 Income":
    st.markdown("# 💼 Income Tracker")
    tab1, tab2 = st.tabs(["➕ Add Income","📋 View Income"])
    SOURCES = ["Salary","Freelance","Part-time Job","Pocket Money",
               "Stipend","Business","Investment","Gift","Other"]
    with tab1:
        c1,c2 = st.columns(2)
        with c1:
            inc_source = st.selectbox("Source", SOURCES)
            inc_amount = st.number_input("Amount (Rs.) *", min_value=0.01, step=100.0,
                                          value=None, placeholder="Enter amount")
        with c2:
            inc_date = st.date_input("Date", value=date.today())
            inc_note = st.text_input("Note", placeholder="e.g., Monthly stipend")
        if st.button("💾 Save Income", use_container_width=True):
            if inc_amount is None or inc_amount <= 0:
                st.error("Please enter a valid amount.")
            else:
                data["incomes"].append({
                    "source": inc_source, "amount": float(inc_amount),
                    "date": str(inc_date), "note": inc_note or ""
                })
                save_data(data)
                st.session_state.data = data
                st.success(f"Income of Rs.{inc_amount:,.0f} saved!")
                st.rerun()
    with tab2:
        if not income_df.empty:
            idf = income_df.copy()
            idf["date"] = pd.to_datetime(idf["date"])
            st.dataframe(idf[["date","source","amount","note"]].sort_values("date",ascending=False),
                         use_container_width=True, hide_index=True)
            src_sum = idf.groupby("source")["amount"].sum().reset_index()
            fig = px.pie(src_sum, values="amount", names="source",
                         title="Income Sources", color_discrete_sequence=px.colors.sequential.Teal)
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",font_color="#e0e0e0")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No income recorded yet.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SAVINGS GOALS  — fixed validation
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Savings Goals":
    st.markdown("# 🎯 Savings Goals")
    tab1, tab2 = st.tabs(["➕ New Goal","📊 Track Goals"])

    with tab1:
        st.markdown("### Create a New Savings Goal")
        c1,c2 = st.columns(2)
        with c1:
            goal_name   = st.text_input("Goal Name *", placeholder="e.g., New Laptop, Bike, Trip to Goa")
            goal_target = st.number_input("Target Amount (Rs.) *",
                                           min_value=0.0, step=500.0,
                                           value=0.0,
                                           help="Enter the total amount you want to save")
        with c2:
            goal_saved = st.number_input("Already Saved (Rs.)",
                                          min_value=0.0, step=100.0, value=0.0)
            goal_date  = st.date_input("Target Date",
                                        value=date.today() + timedelta(days=180),
                                        min_value=date.today())

        if st.button("🎯 Create Goal", use_container_width=True):
            errors = []
            if not goal_name or not goal_name.strip():
                errors.append("Goal Name cannot be empty.")
            if goal_target <= 0:
                errors.append("Target Amount must be greater than 0.")
            # ── Duplicate check ──────────────────────────────────────────────
            existing_keys = {
                (g["name"].strip().lower(), g["target"], g["target_date"])
                for g in data["savings_goals"]
            }
            new_key = (goal_name.strip().lower(), float(goal_target), str(goal_date))
            if new_key in existing_keys:
                errors.append("A goal with this name, target, and date already exists.")
            if errors:
                for e in errors:
                    st.error(f"⚠️ {e}")
            else:
                data["savings_goals"].append({
                    "name":        goal_name.strip(),
                    "target":      float(goal_target),
                    "saved":       float(goal_saved),
                    "target_date": str(goal_date),
                    "created":     str(date.today())
                })
                save_data(data)
                st.session_state.data = data
                st.success(f"Goal **'{goal_name.strip()}'** created! Target: Rs.{goal_target:,.0f}")
                st.rerun()

    with tab2:
        # Reload fresh from session state
        goals = st.session_state.data["savings_goals"]
        if goals:
            for i, goal in enumerate(goals):
                target = float(goal["target"])
                saved  = float(goal["saved"])
                pct    = min((saved / target) * 100, 100) if target > 0 else 0
                due    = datetime.strptime(goal["target_date"],"%Y-%m-%d").date()
                days_l = (due - date.today()).days
                needed = max(target - saved, 0)
                mthly  = needed / max(days_l / 30, 1) if days_l > 0 else needed

                bar_color = "#4ecdc4" if pct >= 75 else ("#f7b731" if pct >= 40 else "#ff6b6b")
                st.markdown(
                    f'<div class="goal-card">'
                    f'<b>🎯 {goal["name"]}</b> &nbsp;—&nbsp; '
                    f'Rs.{saved:,.0f} <span style="color:#aaa">/ Rs.{target:,.0f}</span>'
                    f'&nbsp;&nbsp;<span style="color:{bar_color};font-weight:700">{pct:.0f}%</span><br>'
                    f'<small>📅 {days_l} days left &nbsp;|&nbsp; '
                    f'Need Rs.{mthly:,.0f}/month to reach goal</small>'
                    f'</div>',
                    unsafe_allow_html=True)
                st.progress(pct / 100)

                gc1, gc2, gc3 = st.columns([3,1,1])
                with gc1:
                    deposit = st.number_input(
                        f"Deposit for goal {i+1}", min_value=0.0, step=100.0,
                        key=f"dep_{i}", label_visibility="collapsed",
                        placeholder=f"Add deposit to '{goal['name']}'")
                with gc2:
                    if st.button("➕ Add Deposit", key=f"addbtn_{i}", use_container_width=True):
                        if deposit > 0:
                            st.session_state.data["savings_goals"][i]["saved"] = saved + deposit
                            save_data(st.session_state.data)
                            st.success(f"Rs.{deposit:,.0f} added!")
                            st.rerun()
                        else:
                            st.warning("Enter an amount > 0")
                with gc3:
                    if st.button("🗑️ Delete", key=f"delgoal_{i}", use_container_width=True):
                        st.session_state.data["savings_goals"].pop(i)
                        save_data(st.session_state.data)
                        st.rerun()
                st.markdown("---")
        else:
            st.info("No savings goals yet. Go to ➕ New Goal tab to create one!")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: BILLS  — with Mark as Paid, Edit, Delete
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧾 Bills":
    st.markdown("# 🧾 Recurring Bills")
    tab1, tab2 = st.tabs(["➕ Add Bill","📋 Manage Bills"])

    with tab1:
        st.markdown("### Add a New Bill")
        c1,c2 = st.columns(2)
        with c1:
            bill_name   = st.text_input("Bill Name *", placeholder="e.g., WiFi, Electricity, Rent")
            bill_amount = st.number_input("Amount (Rs.) *", min_value=0.0, step=50.0,
                                           value=0.0)
        with c2:
            bill_due  = st.date_input("Next Due Date", value=date.today() + timedelta(days=7))
            bill_freq = st.selectbox("Frequency", ["Monthly","Weekly","Quarterly","Yearly","One-time"])

        if st.button("🧾 Add Bill", use_container_width=True):
            if not bill_name or not bill_name.strip():
                st.error("Please enter a bill name.")
            elif bill_amount <= 0:
                st.error("Please enter a valid amount greater than 0.")
            else:
                data["bills"].append({
                    "name":      bill_name.strip(),
                    "amount":    float(bill_amount),
                    "due_date":  str(bill_due),
                    "frequency": bill_freq,
                    "paid":      False
                })
                save_data(data)
                st.session_state.data = data
                st.success(f"Bill **'{bill_name.strip()}'** added!")
                st.rerun()

    with tab2:
        bills = st.session_state.data["bills"]
        if not bills:
            st.info("No bills added yet. Use the ➕ Add Bill tab to get started!")
        else:
            today = date.today()
            FREQ_DAYS = {"Weekly":7,"Monthly":30,"Quarterly":91,"Yearly":365,"One-time":0}

            for i, bill in enumerate(bills):
                due       = datetime.strptime(bill["due_date"],"%Y-%m-%d").date()
                days_left = (due - today).days
                paid      = bill.get("paid", False)

                if paid:
                    status_icon = "✅ PAID"
                    border = "rgba(78,205,196,0.5)"; bg = "rgba(78,205,196,0.08)"
                elif days_left < 0:
                    status_icon = "🔴 OVERDUE"
                    border = "rgba(255,80,80,0.5)";  bg = "rgba(255,80,80,0.08)"
                elif days_left <= 7:
                    status_icon = "🟡 DUE SOON"
                    border = "rgba(255,200,0,0.5)";  bg = "rgba(255,200,0,0.06)"
                else:
                    status_icon = "🟢 OK"
                    border = "rgba(78,205,196,0.3)"; bg = "rgba(78,205,196,0.04)"

                st.markdown(
                    f'<div style="background:{bg};border:1px solid {border};'
                    f'border-radius:12px;padding:14px 18px;margin:8px 0;">'
                    f'<span style="font-weight:700;font-size:1rem;">{status_icon}&nbsp;&nbsp;{bill["name"]}</span>'
                    f'&nbsp;&nbsp;<span style="color:#4ecdc4;font-weight:700;">Rs.{bill["amount"]:,.0f}</span>'
                    f'<br><small style="color:#aaa;">Due: {bill["due_date"]}&nbsp;|&nbsp;{bill["frequency"]}'
                    + (f'&nbsp;|&nbsp;<span style="color:#4ecdc4;">Marked Paid</span>' if paid else '')
                    + f'</small></div>',
                    unsafe_allow_html=True)

                bc1, bc2, bc3 = st.columns(3)

                # ── Mark Paid / Unpaid
                with bc1:
                    label = "↩️ Mark Unpaid" if paid else "✅ Mark as Paid"
                    if st.button(label, key=f"paid_{i}", use_container_width=True):
                        if not paid:
                            freq  = bill["frequency"]
                            delta = FREQ_DAYS.get(freq, 30)
                            if freq != "One-time" and delta > 0:
                                new_due = due + timedelta(days=delta)
                                st.session_state.data["bills"][i]["due_date"] = str(new_due)
                                st.session_state.data["bills"][i]["paid"]     = False
                            else:
                                st.session_state.data["bills"][i]["paid"] = True
                        else:
                            st.session_state.data["bills"][i]["paid"] = False
                        save_data(st.session_state.data)
                        st.rerun()

                # ── Edit toggle
                with bc2:
                    edit_label = "🔼 Close Edit" if st.session_state.editing_bill == i else "✏️ Edit Bill"
                    if st.button(edit_label, key=f"edit_{i}", use_container_width=True):
                        st.session_state.editing_bill = None if st.session_state.editing_bill == i else i
                        st.rerun()

                # ── Delete
                with bc3:
                    if st.button("🗑️ Delete", key=f"del_{i}", use_container_width=True):
                        st.session_state.data["bills"].pop(i)
                        save_data(st.session_state.data)
                        st.session_state.editing_bill = None
                        st.rerun()

                # ── Inline Edit Form
                if st.session_state.editing_bill == i:
                    with st.container():
                        st.markdown("**✏️ Edit Bill Details**")
                        ec1, ec2 = st.columns(2)
                        with ec1:
                            new_name   = st.text_input("Bill Name",   value=bill["name"],         key=f"en_{i}")
                            new_amount = st.number_input("Amount (Rs.)", value=float(bill["amount"]),
                                                          min_value=0.0, step=50.0,               key=f"ea_{i}")
                        with ec2:
                            new_due_v  = st.date_input("Due Date",    value=due,                  key=f"ed_{i}")
                            freq_opts  = ["Monthly","Weekly","Quarterly","Yearly","One-time"]
                            curr_idx   = freq_opts.index(bill["frequency"]) if bill["frequency"] in freq_opts else 0
                            new_freq   = st.selectbox("Frequency",    freq_opts, index=curr_idx,  key=f"ef_{i}")

                        sv1, sv2 = st.columns(2)
                        with sv1:
                            if st.button("💾 Save Changes", key=f"sv_{i}", use_container_width=True):
                                if not new_name.strip():
                                    st.error("Name cannot be empty.")
                                elif new_amount <= 0:
                                    st.error("Amount must be > 0.")
                                else:
                                    st.session_state.data["bills"][i].update({
                                        "name":      new_name.strip(),
                                        "amount":    float(new_amount),
                                        "due_date":  str(new_due_v),
                                        "frequency": new_freq
                                    })
                                    save_data(st.session_state.data)
                                    st.session_state.editing_bill = None
                                    st.success(f"Bill '{new_name}' updated!")
                                    st.rerun()
                        with sv2:
                            if st.button("❌ Cancel", key=f"cx_{i}", use_container_width=True):
                                st.session_state.editing_bill = None
                                st.rerun()

                st.markdown("---")



# ══════════════════════════════════════════════════════════════════════════════
# PAGE: BUDGET LIMITS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💰 Budget Limits":
    st.markdown("# 💰 Budget Limits")
    st.markdown("Set monthly spending limits per category and track how you're doing.")

    budget_limits = st.session_state.data.setdefault("budget_limits", {})

    # ── Current month spending per category ───────────────────────────────────
    this_month = date.today().strftime("%Y-%m")
    cat_spent_this_month = {}
    if not expenses_df.empty:
        edf_b = expenses_df.copy()
        edf_b["date"] = pd.to_datetime(edf_b["date"])
        monthly_exp = edf_b[edf_b["date"].dt.strftime("%Y-%m") == this_month]
        if not monthly_exp.empty:
            cat_spent_this_month = monthly_exp.groupby("category")["amount"].sum().to_dict()

    # ── Set / Edit limits ─────────────────────────────────────────────────────
    st.markdown("### ✏️ Set Monthly Limits")
    st.caption("Leave 0 to remove a limit for that category.")
    cols = st.columns(3)
    updated_limits = {}
    for idx, cat in enumerate(CATEGORIES):
        with cols[idx % 3]:
            current = float(budget_limits.get(cat, 0))
            new_val = st.number_input(cat, min_value=0.0, step=100.0,
                                      value=current, key=f"bl_{cat}")
            if new_val > 0:
                updated_limits[cat] = new_val

    if st.button("💾 Save Budget Limits", use_container_width=True):
        st.session_state.data["budget_limits"] = updated_limits
        save_data(st.session_state.data)
        st.success("Budget limits saved!")
        st.rerun()

    st.markdown("---")
    st.markdown(f"### 📊 This Month's Spending vs Limits  ({this_month})")

    if not updated_limits and not budget_limits:
        st.info("No budget limits set yet. Enter limits above and click Save.")
    else:
        active_limits = budget_limits if budget_limits else updated_limits
        rows = []
        for cat, limit in active_limits.items():
            spent = cat_spent_this_month.get(cat, 0.0)
            pct   = (spent / limit * 100) if limit > 0 else 0
            status = "✅ OK" if pct <= 75 else ("⚠️ Warning" if pct <= 100 else "🔴 Over!")
            rows.append({"Category": cat, "Limit (Rs.)": limit,
                         "Spent (Rs.)": spent, "Used %": round(pct, 1), "Status": status})

        if rows:
            for row in rows:
                pct   = row["Used %"]
                color = "#4ecdc4" if pct <= 75 else ("#f7b731" if pct <= 100 else "#ff6b6b")
                bar   = min(pct, 100)
                st.markdown(
                    f'<div style="background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.1);'
                    f'border-radius:12px;padding:14px 18px;margin:6px 0;">'
                    f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                    f'<span style="font-weight:600;">{row["Category"]}</span>'
                    f'<span style="color:{color};font-weight:700;">{row["Status"]}</span></div>'
                    f'<div style="margin:6px 0;font-size:0.88rem;color:#aaa;">'
                    f'Rs.{row["Spent (Rs.)"]:,.0f} / Rs.{row["Limit (Rs.)"]:,.0f} '
                    f'&nbsp;—&nbsp; <b style="color:{color}">{pct:.1f}%</b></div>'
                    f'<div style="background:rgba(255,255,255,0.08);border-radius:8px;height:8px;">'
                    f'<div style="background:{color};width:{bar}%;height:8px;border-radius:8px;'
                    f'transition:width 0.4s ease;"></div></div></div>',
                    unsafe_allow_html=True)

            # Summary chart
            st.markdown("### 📈 Overview Chart")
            chart_df = pd.DataFrame(rows)
            fig = go.Figure()
            fig.add_bar(name="Spent", x=chart_df["Category"], y=chart_df["Spent (Rs.)"],
                        marker_color="#6c63ff")
            fig.add_bar(name="Limit", x=chart_df["Category"], y=chart_df["Limit (Rs.)"],
                        marker_color="rgba(78,205,196,0.4)")
            fig.update_layout(barmode="overlay", plot_bgcolor="rgba(0,0,0,0)",
                              paper_bgcolor="rgba(0,0,0,0)", font_color=_text,
                              legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig, use_container_width=True)

            # ── Export budget report ──────────────────────────────────────────
            report_df = pd.DataFrame(rows)
            st.download_button("⬇️ Export Budget Report CSV",
                data=_to_csv(report_df), file_name=f"fintrack_budget_{this_month}.csv",
                mime="text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: AI CHATBOT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 AI Chatbot":  # noqa: E741
    st.markdown("# 🤖 FinBot — AI Finance Chatbot")
    st.markdown("*Ask me anything about your finances. I analyse your real data!*")

    advisor    = SavingsAdvisor()
    forecaster = SpendingForecaster()

    cat_breakdown = ""
    if not expenses_df.empty:
        edf_c = expenses_df.copy()
        edf_c["date"] = pd.to_datetime(edf_c["date"])
        cat_totals    = edf_c.groupby("category")["amount"].sum().to_dict()
        cat_breakdown = ", ".join([f"{k}: Rs.{v:,.0f}" for k,v in cat_totals.items()])

    goals_text = "\n".join(
        [f"- {g['name']}: Rs.{g['saved']:,.0f}/Rs.{g['target']:,.0f} "
         f"({(g['saved']/g['target']*100) if g['target']>0 else 0:.0f}%)"
         for g in data["savings_goals"]]) or "No goals set."

    bills_text = "\n".join(
        [f"- {b['name']}: Rs.{b['amount']:,.0f} due "
         f"{(datetime.strptime(b['due_date'],'%Y-%m-%d').date()-date.today()).days} days"
         for b in data["bills"]]) or "No bills."

    ai_tips = advisor.get_recommendations(
        expenses_df if not expenses_df.empty else pd.DataFrame(),
        total_income, balance, data["savings_goals"])
    tips_text = "\n".join([f"- {t}" for t in ai_tips])

    SYSTEM_PROMPT = f"""You are FinBot, a smart and friendly AI financial advisor inside FinTrack.
You have access to the user's real financial data. Give specific, personalised, concise advice.
Use Rs. for currency. Keep replies under 200 words. Be encouraging and practical.

=== USER FINANCIAL DATA ===
Income:       Rs.{total_income:,.0f}
Expenses:     Rs.{total_expenses:,.0f}
Balance:      Rs.{balance:,.0f}
Savings Rate: {savings_rate:.1f}%
Category Spending: {cat_breakdown or "No data yet."}
Savings Goals:
{goals_text}
Bills:
{bills_text}
ML Tips:
{tips_text}
==========================="""

    if not st.session_state.chat_history:
        st.markdown(
            '<div class="chat-bot">👋 Hi! I\'m <b>FinBot</b> — your AI finance advisor.<br>'
            'Try: <i>"Where am I overspending?"</i> or <i>"How can I save more?"</i></div>',
            unsafe_allow_html=True)

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-meta" style="text-align:right">{msg.get("time","")}</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bot">🤖 {msg["content"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-meta">{msg.get("time","")}</div>', unsafe_allow_html=True)

    st.markdown("---")
    ci, cs, cc = st.columns([7,1,1])
    with ci:
        user_input = st.text_input("msg", label_visibility="collapsed",
                                   placeholder="Ask FinBot about your finances...")
    with cs:
        send_clicked = st.button("Send", use_container_width=True)
    with cc:
        if st.button("Clear", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    st.markdown("**Quick questions:**")
    qcols = st.columns(4)
    quick = ["Where am I overspending?","Give me 3 saving tips",
             "What's my savings score?","How to reach my goal?"]
    triggered_prompt = None
    for idx, qp in enumerate(quick):
        if qcols[idx].button(qp, key=f"qp_{idx}"):
            triggered_prompt = qp

    final_input = triggered_prompt or (user_input if send_clicked else None)

    if final_input and final_input.strip():
        now = datetime.now().strftime("%H:%M")
        st.session_state.chat_history.append(
            {"role":"user","content":final_input.strip(),"time":now})
        api_messages = [{"role":m["role"],"content":m["content"]}
                        for m in st.session_state.chat_history
                        if m["role"] in ("user","assistant")]
        with st.spinner("FinBot is thinking..."):
            try:
                import anthropic
                client   = anthropic.Anthropic()
                response = client.messages.create(
                    model="claude-sonnet-4-20250514", max_tokens=512,
                    system=SYSTEM_PROMPT, messages=api_messages)
                bot_reply = response.content[0].text
            except Exception:
                bot_reply = _local_chatbot(
                    final_input.strip(), total_income, total_expenses,
                    balance, savings_rate, cat_breakdown, ai_tips, data)
        st.session_state.chat_history.append(
            {"role":"assistant","content":bot_reply,"time":datetime.now().strftime("%H:%M")})
        st.rerun()

    if not expenses_df.empty:
        st.markdown("---")
        cf, cs2 = st.columns(2)
        with cf:
            st.markdown("### 📈 Spending Forecast")
            edf_f = expenses_df.copy()
            edf_f["date"] = pd.to_datetime(edf_f["date"])
            forecast = forecaster.forecast(edf_f)
            if forecast is not None:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=forecast["month"], y=forecast["actual"],
                                          mode="lines+markers", name="Actual",
                                          line=dict(color="#6c63ff",width=2)))
                fig.add_trace(go.Scatter(x=forecast["month"], y=forecast["predicted"],
                                          mode="lines+markers", name="Forecast",
                                          line=dict(color="#4ecdc4",width=2,dash="dash")))
                fig.update_layout(plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",
                                   font_color="#e0e0e0",margin=dict(t=20))
                st.plotly_chart(fig, use_container_width=True)
        with cs2:
            st.markdown("### 🏆 Financial Health")
            score = advisor.savings_score(total_income, total_expenses, len(data["savings_goals"]))
            m1,m2 = st.columns(2)
            m1.metric("Savings Score", f"{score}/100")
            m2.metric("Savings Rate",  f"{savings_rate:.1f}%")
            if score >= 70:   st.success("Excellent financial discipline!")
            elif score >= 40: st.warning("Good — small tweaks can improve your score.")
            else:             st.error("Ask FinBot for personalised advice!")