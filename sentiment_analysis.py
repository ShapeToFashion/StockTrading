import ast
from transformers import pipeline , AutoTokenizer , AutoModelForSequenceClassification
from transformers import AutoConfig

from datetime import datetime, timedelta
from alpaca_trade_api import REST , Stream
import torch.nn.functional as F
import pandas as pd

config = AutoConfig.from_pretrained(
  "bert-base-uncased",
  num_labels=2,
  id2label = {0: "positive" , 1:"negative"},
  label2id = {"positive":0 , "negative":1}
)
#find the sentiment_model.zip in the whatsapp chat and extract it in the pwd 
model_path = "./sentiment_model"

model = AutoModelForSequenceClassification.from_pretrained(model_path , config=config)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

model.eval()

def extract_content(news_string):
    cleaned = news_string.replace("NewsV2(", "").rstrip(")")
    try:
        parsed = ast.literal_eval(cleaned)
        return parsed.get('headline')
    except:
        return None


def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_label = logits.argmax(dim=1).item()
    probs = F.softmax(logits, dim=1)  
    return probs[0][0].item()


def find_sentiment(dataFrame , ticker):
    #ticker = "AAPL"
    API_KEY = 'enter the alpaca api key'
    API_SECRET = 'enter the alpaca secret'
    time_df = pd.DataFrame(dataFrame['Date'])
    sentiment_scores = []

    for index, item in time_df.iterrows():
        print(item['Date'])
        date = item['Date']
        start = datetime.combine(date, datetime.min.time())
        end = start + timedelta(days=1)

        res_client = REST(API_KEY, API_SECRET, base_url='https://paper-api.alpaca.markets')
        news = res_client.get_news(ticker, start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'), limit=10)
        news_dicts = [n._raw for n in news]
        news_df = pd.DataFrame(news_dicts)

        if not news_df.empty:
            news_df['sentiment'] = news_df['headline'].apply(predict_sentiment)
            daily_sentiment = news_df['sentiment'].mean()
        else:
            daily_sentiment = None  # Or set to 0.0 or neutral value if preferred

        sentiment_scores.append(daily_sentiment)

    dataFrame['sentiment_score'] = sentiment_scores
    return dataFrame

