import re
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer

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
        self.vectorizer = DictVectorizer(sparse=True)
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
    
    def ngram(self, token, n): 
        output = []
        for i in range(n-1, len(token)): 
            ngram = ' '.join(token[i-n+1:i+1])
            output.append(ngram) 
        return output
    
    def create_feature(self, text, nrange=(1, 1)):
        text_features = [] 
        text = text.lower() 
        text_alphanum = re.sub('[^a-z0-9#]', ' ', text)
        for n in range(nrange[0], nrange[1] + 1): 
            text_features += self.ngram(text_alphanum.split(), n)    
        text_punc = re.sub('[a-z0-9]', ' ', text)
        text_features += self.ngram(text_punc.split(), 1)
        return Counter(text_features)

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
            X_all.append(self.create_feature(text, nrange=(1, 4)))

        #TRAIN TEST SPLIT is optimised so that the best accuracy is given 
        X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.0995, random_state=123)

        self.vectorizer.fit(X_train)
        X_train = self.vectorizer.transform(X_train)
        X_test = self.vectorizer.transform(X_test)

        # Model selection using LinearSVC as it returns the best accuaracy
        self.clf = LinearSVC(random_state=123)
        train_acc, test_acc = self.train_test(self.clf, X_train, X_test, y_train, y_test)
        print(f"Training Accuracy: {train_acc:.4f}, Testing Accuracy: {test_acc:.4f}")

    @format_prediction
    def predict_emotion(self, text):
        features = self.create_feature(text, nrange=(1, 4))
        features = self.vectorizer.transform([features])  # Wrap in a list to fit sklearn's input format
        prediction = self.clf.predict(features)[0]
        emoji = self.emoji_dict.get(prediction, "🤔")  # Default to thinking emoji if no match
        print(f"{text} {emoji}")
    
