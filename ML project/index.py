import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report



df = pd.read_csv("spam.csv", encoding="latin-1")
df = df[["v1","v2"]]

df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(df['message'])
y = df['label']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)
# print("Training Accuracy:", train_acc)
# print("Test Accuracy:", test_acc)
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print(cm)

print(classification_report(y_test, y_pred))
