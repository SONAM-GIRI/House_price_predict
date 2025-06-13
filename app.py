from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import io
import base64
import os

app = Flask(__name__)

# Load and train model
df = pd.read_csv('data.csv')
X = df[['Size', 'Bedrooms', 'Age', 'LocationScore']]
y = df['Price']
model = LinearRegression().fit(X, y)

# Generate actual vs predicted chart
def plot_prediction_chart(y_true, y_pred):
    plt.figure(figsize=(6, 4))
    plt.scatter(y_true, y_pred, color='teal', alpha=0.6)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted House Prices")
    plt.grid(True)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close()
    return img_base64

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        size = int(request.form['size'])
        bedrooms = int(request.form['bedrooms'])
        age = int(request.form['age'])
        score = int(request.form['location'])
        features = [[size, bedrooms, age, score]]
        prediction = int(model.predict(features)[0])

        y_pred = model.predict(X)
        chart = plot_prediction_chart(y, y_pred)

        return render_template('predict.html', prediction=prediction, chart=chart)
    return render_template('predict.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
