import re
from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer


def format_prediction(func):
    def wrapper(*args, **kwargs):
        print("=" * 40)
        print(" PREDICTION RESULT ".center(40, "="))
        print("=" * 40)
        result = func(*args, **kwargs)
        print("=" * 40)
        return result
    return wrapper

class EmotionMatcher:

    def __init__(self, file_path):
        self.file_path = file_path
        self.emoji_dict = {
            "joy": "😁", 
            "fear": "😱", 
            "anger": "😠", 
            "sadness": "😢", 
            "disgust": "😒", 
            "shame": "😳", 
            "guilt": "😬"
        }
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 3), stop_words='english', max_features=5000)
        self.clf = None
        self.emotions = ["joy", "fear", "anger", "sadness", "disgust", "shame", "guilt"]

    def read_data(self, file):
        data = []
        with open(file, 'r') as f:
            for line in f:
                line = line.strip()
                label = ' '.join(line[1:line.find("]")].strip().split())
                text = line[line.find("]")+1:].strip()
                data.append([label, text])
        return data

    def create_feature(self, text):
        text = text.lower()
        text_alphanum = re.sub('[^a-z0-9#]', ' ', text)
        return text_alphanum

    def convert_label(self, item, name):
        items = list(map(float, item.split()))
        label = ""
        for idx in range(len(items)):
            if items[idx] == 1:
                label += name[idx] + " "
        return label.strip()

    def train_test(self, clf, X_train, X_test, y_train, y_test):
        clf.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, clf.predict(X_train))
        test_acc = accuracy_score(y_test, clf.predict(X_test))
        return train_acc, test_acc

    def run_tool(self):
        data = self.read_data(self.file_path)
        X_all = []
        y_all = []
        for label, text in data:
            y_all.append(self.convert_label(label, self.emotions))
            X_all.append(self.create_feature(text))

        
        X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.1, random_state=123)

        # Vectorize the text data using TF-IDF
        X_train = self.vectorizer.fit_transform(X_train)
        X_test = self.vectorizer.transform(X_test)

        # Hyperparameter tuning for LinearSVC
        param_grid = {'C': [0.11]}
        self.clf = GridSearchCV(LinearSVC(random_state=123), param_grid, cv=5)
        
        train_acc, test_acc = self.train_test(self.clf, X_train, X_test, y_train, y_test)
        print(f"Training Accuracy: {train_acc:.4f}, Testing Accuracy: {test_acc:.4f}")
        print(f"Best Parameters: {self.clf.best_params_}")

    @format_prediction
    def predict_emotion(self, text):
        features = self.create_feature(text)
        features = self.vectorizer.transform([features])
        prediction = self.clf.predict(features)[0]
        emoji = self.emoji_dict.get(prediction, "🤔")
        print(f"{text} {emoji}")
    
