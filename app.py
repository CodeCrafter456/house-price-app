import os, joblib, numpy as np  # type: ignore
from functools import wraps
from flask import (Flask, render_template, request, redirect,
                   url_for, session, flash, g)
from werkzeug.security import generate_password_hash, check_password_hash
from database import get_db, init_db

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-in-production")

MODEL_PATH = os.path.join("models", "house_price_model.joblib")
model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("Model loaded")
else:
    print("Model not found – run train_model.py first")

@app.teardown_appcontext
def close_db(error):
    db = g.pop("db", None)
    if db: db.close()

def db():
    if "db" not in g:
        g.db = get_db()
    return g.db

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to continue.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if session.get("username") != "admin":
            flash("Admin access only.", "danger")
            return redirect(url_for("dashboard"))
        return f(*args, **kwargs)
    return login_required(decorated)

@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))

@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "POST":
        username  = request.form["username"].strip()
        email     = request.form["email"].strip().lower()
        password  = request.form["password"]
        full_name = request.form.get("full_name","").strip()
        if not username or not email or not password:
            flash("All fields are required.", "danger")
            return render_template("register.html")
        pw_hash = generate_password_hash(password)
        try:
            db().execute(
                "INSERT INTO users (username,email,password_hash,full_name) "
                "VALUES (?,?,?,?)",
                (username, email, pw_hash, full_name)
            )
            db().commit()
            flash("Account created! Please log in.", "success")
            return redirect(url_for("login"))
        except Exception:
            flash("Username or email already exists.", "danger")
    return render_template("register.html")

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]
        user = db().execute(
            "SELECT * FROM users WHERE username=?", (username,)
        ).fetchone()
        if user and check_password_hash(user["password_hash"], password):
            session["user_id"]  = user["id"]
            session["username"] = user["username"]
            session["email"]    = user["email"]
            session["full_name"]= user["full_name"] or username
            flash(f"Welcome back, {session['full_name']}!", "success")
            return redirect(url_for("dashboard"))
        flash("Invalid username or password.", "danger")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))

@app.route("/dashboard")
@login_required
def dashboard():
    history = db().execute(
        "SELECT * FROM predictions_history WHERE user_id=? ORDER BY created_at DESC LIMIT 10",
        (session["user_id"],)
    ).fetchall()
    total = db().execute(
        "SELECT COUNT(*) FROM predictions_history WHERE user_id=?",
        (session["user_id"],)
    ).fetchone()[0]
    return render_template("dashboard.html", history=history, total=total)

ZIP_CHOICES = [
    (10001,"10001 – New York, NY"),
    (10002,"10002 – New York, NY (Lower East Side)"),
    (90210,"90210 – Beverly Hills, CA"),
    (94102,"94102 – San Francisco, CA"),
    (60601,"60601 – Chicago, IL"),
    (30301,"30301 – Atlanta, GA"),
    (77001,"77001 – Houston, TX"),
    (85001,"85001 – Phoenix, AZ"),
    (98101,"98101 – Seattle, WA"),
    (2101, "02101 – Boston, MA"),
]

@app.route("/predict", methods=["GET","POST"])
@login_required
def predict():
    prediction = None
    form_data  = {}
    if request.method == "POST":
        try:
            sqft       = int(request.form["sqft"])
            bedrooms   = int(request.form["bedrooms"])
            bathrooms  = int(request.form["bathrooms"])
            year_built = int(request.form["year_built"])
            zip_code   = int(request.form["zip_code"])
            form_data  = dict(sqft=sqft, bedrooms=bedrooms,
                              bathrooms=bathrooms, year_built=year_built,
                              zip_code=zip_code)
            if model is None:
                flash("ML model not loaded. Run train_model.py first.", "danger")
            else:
                features   = np.array([[sqft, bedrooms, bathrooms,
                                         year_built, zip_code]])
                prediction = round(float(model.predict(features)[0]), 2)
                db().execute(
                    "INSERT INTO predictions_history "
                    "(user_id,sqft,bedrooms,bathrooms,year_built,zip_code,predicted_price) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (session["user_id"], sqft, bedrooms, bathrooms,
                     year_built, zip_code, prediction)
                )
                db().commit()
        except (ValueError, KeyError) as e:
            flash(f"Invalid input: {e}", "danger")
            return render_template("predict.html",
                           prediction=prediction,
                           form_data=form_data,
                           zip_choices=ZIP_CHOICES)
    
    return render_template("predict.html",
                           prediction=prediction,
                           form_data=form_data,
                           zip_choices=ZIP_CHOICES)

@app.route("/profile", methods=["GET","POST"])
@login_required
def profile():
    if request.method == "POST":
        full_name = request.form.get("full_name","").strip()
        email     = request.form.get("email","").strip().lower()
        new_pw    = request.form.get("new_password","")
        row = db().execute(
            "SELECT * FROM users WHERE id=?", (session["user_id"],)
        ).fetchone()
        pw_hash = row["password_hash"]
        if new_pw:
            if len(new_pw) < 6:
                flash("Password must be at least 6 characters.", "danger")
                return redirect(url_for("profile"))
            pw_hash = generate_password_hash(new_pw)
        try:
            db().execute(
                "UPDATE users SET full_name=?,email=?,password_hash=? WHERE id=?",
                (full_name, email, pw_hash, session["user_id"])
            )
            db().commit()
            session["full_name"] = full_name
            session["email"]     = email
            flash("Profile updated!", "success")
        except Exception:
            flash("Email already in use.", "danger")
        return redirect(url_for("profile"))
    user = db().execute(
        "SELECT * FROM users WHERE id=?", (session["user_id"],)
    ).fetchone()
    return render_template("profile.html", user=user)

@app.route("/admin")
@admin_required
def admin():
    users = db().execute(
        "SELECT u.id, u.username, u.email, u.full_name, u.created_at, "
        "COUNT(p.id) as pred_count "
        "FROM users u LEFT JOIN predictions_history p ON p.user_id=u.id "
        "GROUP BY u.id ORDER BY u.created_at DESC"
    ).fetchall()
    return render_template("admin.html", users=users)

if __name__ == "__main__":
    init_db()
    app.run(debug=True, port=5000)