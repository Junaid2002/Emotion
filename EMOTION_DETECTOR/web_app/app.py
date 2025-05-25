import os
import sys
from flask import Flask, render_template, request
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from predict import preprocess_and_predict

app = Flask(__name__, template_folder='../templates', static_folder='../static')
app.secret_key = 'a_random_secure_key_1234567890'

@app.route('/')
def index():
    return render_template('index.html', prediction=None)

@app.route('/', methods=['POST'])
def analyze():
    user_input = request.form['user_input']
    try:
        prediction = preprocess_and_predict(user_input)
    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")
    return render_template('index.html', prediction=prediction)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']
        return render_template('contact.html', message_submitted=True)
    return render_template('contact.html', message_submitted=False)

@app.route('/report', methods=['POST'])
def report():
    problem_description = request.form['problem_description']
    return render_template('index.html', report_submitted=True)

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)