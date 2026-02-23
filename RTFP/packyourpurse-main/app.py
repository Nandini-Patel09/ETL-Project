import requests
from flask import Flask, render_template, redirect, url_for, request, session
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

API_KEY ='fsq3lJ5rZy9QG+jQIqex8R2pynfAOywb9JBkYXjU0RdSPkQ='

# Mock databases
users = {}
user_trips = {}
user_expenses = {}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'])
        users[username] = password
        user_trips[username] = {}
        user_expenses[username] = []
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user_password = users.get(username)

        if user_password and check_password_hash(user_password, password):
            session['username'] = username

            # 🛠️ Initialize user data if not present
            if username not in user_trips:
                user_trips[username] = {}

            if username not in user_expenses:
                user_expenses[username] = []

            return redirect(url_for('dashboard'))
        else:
            return "Invalid credentials. Try again."
    return render_template('login.html')


@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))

    trip = user_trips.get(session['username'])
    expenses = user_expenses.get(session['username'], [])

    total_expenses = 0
    if expenses:
        total_expenses = sum(float(expense['amount']) for expense in expenses)

    remaining_budget = 0
    if trip and trip.get('budget'):
        remaining_budget = float(trip['budget']) - total_expenses


    suggestions = []
    if trip and trip.get('destination'):
        suggestions = get_travel_suggestions(trip['destination'])

    return render_template('dashboard.html', trip=trip, expenses=expenses, suggestions=suggestions, total_expenses=total_expenses, remaining_budget=remaining_budget)



@app.route('/trip', methods=['GET', 'POST'])
def trip():
    if 'username' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        destination = request.form['destination']
        budget = request.form['budget']
        travelers = request.form['travelers']
        days = request.form['days']

        # Save trip
        user_trips[session['username']] = {
            'destination': destination,
            'budget': budget,
            'travelers': travelers,
            'days': days
        }
        return redirect(url_for('dashboard'))
    
    return render_template('trip.html')

@app.route('/expenses', methods=['GET', 'POST'])
def expenses():
    if 'username' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        category = request.form['category']
        amount = request.form['amount']

        # 🛠️ Fix: check if user's expenses exist
        if session['username'] not in user_expenses:
            user_expenses[session['username']] = []

        user_expenses[session['username']].append({
            'category': category,
            'amount': amount
        })
        return redirect(url_for('dashboard'))
    
    return render_template('expenses.html')

def get_travel_suggestions(destination):
    url = "https://api.foursquare.com/v3/places/search"
    headers = {
        "Accept": "application/json",
        "Authorization": API_KEY
    }
    params = {
        "near": destination,
        "limit": 5
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()

        suggestions = []
        for place in data.get("results", []):
            name = place.get("name")
            fsq_id = place.get("fsq_id")
            categories = place.get("categories", [])
            description = categories[0]["name"] if categories else ""

            # Get place photo
            photo_url = None
            if fsq_id:
                photo_response = requests.get(
                    f"https://api.foursquare.com/v3/places/{fsq_id}/photos",
                    headers=headers
                )
                if photo_response.status_code == 200:
                    photos = photo_response.json()
                    if photos:
                        first_photo = photos[0]
                        prefix = first_photo.get("prefix")
                        suffix = first_photo.get("suffix")
                        if prefix and suffix:
                            photo_url = f"{prefix}original{suffix}"

            suggestions.append({
                "name": name,
                "description": description,
                "photo_url": photo_url
            })

        return suggestions
    except Exception as e:
        print(f"Error fetching travel suggestions: {e}")
        return []



@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('home'))

if __name__ == "__main__":
    app.run(debug=True)