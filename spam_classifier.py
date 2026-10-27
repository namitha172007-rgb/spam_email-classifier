import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import urllib.request, zipfile, io
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix

# Data download (UCI SMS Spam Collection)
url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(req).read()))
df = pd.read_csv(z.open("SMSSpamCollection"), sep="\t", header=None, names=["label", "text"])
df["label"] = df["label"].map({"ham": 0, "spam": 1})
print(df.head())
print(df["label"].value_counts())

# Split + TF-IDF
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42)
tfidf = TfidfVectorizer(stop_words="english", lowercase=True)
X_train_t = tfidf.fit_transform(X_train)
X_test_t = tfidf.transform(X_test)

# Train 3 models
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": LinearSVC(),
}
for name, m in models.items():
    m.fit(X_train_t, y_train)
    pred = m.predict(X_test_t)
    print(name, "| Accuracy:", round(accuracy_score(y_test, pred), 4),
          "| Precision:", round(precision_score(y_test, pred), 4))

# Real-time check
best = models["SVM"]
def check(msg):
    return "SPAM" if best.predict(tfidf.transform([msg]))[0] == 1 else "HAM"

print(check("Congratulations! You won a free iPhone, click now"))
print(check("Hi, are we meeting at 5 pm today?"))

# Confusion matrix graph
cm = confusion_matrix(y_test, best.predict(X_test_t))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.show()

# Top spam words graph
lr = models["Logistic Regression"]
words = np.array(tfidf.get_feature_names_out())
top = np.argsort(lr.coef_[0])[-15:]
plt.barh(words[top], lr.coef_[0][top])
plt.title("Top spam words"); plt.show()