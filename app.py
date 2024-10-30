import warnings
warnings.filterwarnings('ignore')
from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import os
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

app = Flask(__name__)

# Load and preprocess data
data = pd.read_csv('AmazonReview.csv')
data.dropna(inplace=True)
data.loc[data['Sentiment'] <= 3, 'Sentiment'] = 0
data.loc[data['Sentiment'] > 3, 'Sentiment'] = 1
stp_words = stopwords.words('english')

def clean_review(review):
    return " ".join(word for word in review.split() if word not in stp_words)

data['Review'] = data['Review'].apply(clean_review)

# Text Vectorization and Model Training
cv = TfidfVectorizer(max_features=2500)
X = cv.fit_transform(data['Review']).toarray()
x_train, x_test, y_train, y_test = train_test_split(X, data['Sentiment'], test_size=0.25, random_state=42)

model = LogisticRegression()
model.fit(x_train, y_train)
pred = model.predict(x_test)
accuracy = accuracy_score(y_test, pred)
cm = confusion_matrix(y_test, pred)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = ""
    if request.method == 'POST':
        user_review = request.form['review']
        user_review_vector = cv.transform([clean_review(user_review)]).toarray()
        prediction = "Positive" if model.predict(user_review_vector)[0] == 1 else "Negative"
    return render_template('index.html', prediction=prediction, accuracy=accuracy)

if __name__ == '__main__':
    app.run(debug=True)
