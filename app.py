from flask import Flask, jsonify
from flask_cors import CORS
import joblib
from sentiment_analysis import find_sentiment
from risk_analysis import predict_risk


app = Flask(__name__)
CORS(app)

# Load model
try:
    model = joblib.load('model/risk_rf_model.pkl')
    print(" Model loaded")
except:
    model = None
    print("Failed to load model")

@app.route("/sentiment")
def sentiment_analyser(DataFrame , ticker):
    return find_sentiment(DataFrame , ticker); 

@app.route('/risk', methods=['POST'])
def risk_analyser():
    if not model:
        return jsonify({'error': 'Model not loaded'}),

    data = request.get_json()
    ticker = data.get('ticker')

    if not ticker:
        return jsonify({'error': 'Ticker not provided'}),

    try:
        results = predict_risk(ticker, model)
        if results is None:
            return jsonify({'error': f'Data fetch failed for {ticker}'}),

        formatted = [{
            'date': str(date),
            'price': float(row['Close']),
            'risk': int(row['Predicted_Risk'])
        } for date, row in results.iterrows()]

        return jsonify({'ticker': ticker, 'predictions': formatted}),

    except Exception as e:
        return jsonify({'error': str(e)}),


@app.route("/")
def hello_world():
    return "<p>Hello World</p>"

if __name__ == "__main__":
    app.run(debug=True)