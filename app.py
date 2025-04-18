from flask import Flask
from sentiment_analysis import find_sentiment

app = Flask(__name__)

@app.route("/sentiment")
def sentiment_analyser(DataFrame , ticker):
    return find_sentiment(DataFrame , ticker); 

@app.route("/")
def hello_world():
    return "<p>Hello World</p>"

if __name__ == "__main__":
    app.run(debug=True)
