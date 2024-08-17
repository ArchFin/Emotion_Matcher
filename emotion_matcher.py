import re 
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer

class EmotionMatcher(file_path):

    def __init__(self, file_path):

        self.file_path = file_path

    #we are now wrangling the data that has been given into the form we require cant just use a copy paste code 
    def read_data(self, file):
        data = []
        with open(file, 'r')as f:
            for line in f:
                line = line.strip()
                label = ' '.join(line[1:line.find("]")].strip().split())
                text = line[line.find("]")+1:].strip()
                data.append([label, text])
        return data
    
    #tokenisation of the phrases 
    def ngram(self, token, n): 
        output = []
        for i in range(n-1, len(token)): 
            ngram = ' '.join(token[i-n+1:i+1])
            output.append(ngram) 
        return output
    

    def create_feature(self, text, nrange=(1, 1)):
        text_features = [] 
        text = text.lower() 
        #This skips out some part of the emotion focusing more on the words as we all know captial letters could help identify certain aspects of speech
        text_alphanum = re.sub('[^a-z0-9#]', ' ', text)
        for n in range(nrange[0], nrange[1]+1): 
            text_features += self.ngram(text_alphanum.split(), n)    
        text_punc = re.sub('[a-z0-9]', ' ', text)
        text_features += self.ngram(text_punc.split(), 1)
        return Counter(text_features)

    #creating the labels of emotions

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
        emotions = ["joy", 'fear', "anger", "sadness", "disgust", "shame", "guilt"]
        # equivilant to [1,0,0,0,0,0,0] = "joy"
        X_all = []
        y_all = []
        for label, text in data:
            y_all.append(self.convert_label(label, emotions))
            X_all.append(self.create_feature(text, nrange=(1, 4)))
            
        #training the data set on the data we have 
        X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size = 0.0995, random_state = 123)

        
        vectorizer = DictVectorizer(sparse = True)
        X_train = vectorizer.fit_transform(X_train)
        X_test = vectorizer.transform(X_test)

        #tests to show the effectivness of classifaction 
        svc = SVC()
        lsvc = LinearSVC(random_state=123)
        rforest = RandomForestClassifier(random_state=123)
        dtree = DecisionTreeClassifier()

        clifs = [lsvc, svc, rforest, dtree] #removed all others as they just were not as effective

        for clf in clifs: 
            clf_name = clf.__class__.__name__
            train_acc, test_acc = self.train_test(clf, X_train, X_test, y_train, y_test)
            print("| {:25} | {:17.7f} | {:13.7f} |".format(clf_name, train_acc, test_acc))
            
            
        l = ["joy", 'fear', "anger", "sadness", "disgust", "shame", "guilt"]
        l.sort()
        label_freq = {}
        for label, _ in data: 
            label_freq[label] = label_freq.get(label, 0) + 1

        print("{:10}({})  {}".format(self.convert_label(l, emotions), l, label_freq[l]))

# print the labels and their counts in sorted order 
#for l in sorted(label_freq, key=label_freq.get, reverse=True):
    








#Actual run code 

# test_size = 0.0995, test_accuracy = 0.5865772 best so far using linear SVC






#train and test them 
#print("| {:25} | {} | {} |".format("Classifier", "Training Accuracy", "Test Accuracy"))
#print("| {} | {} | {} |".format("-"*25, "-"*17, "-"*13))

    
#Trial
# emoji_dict = {"joy":"😁", "fear":"😱", "anger":"😠", "sadness":"😢", "disgust":"😒", "shame":"😳", "guilt":"😬"}
# t1 = "I hate physics"
# t2 = "I have a fear of dogs"
# t3 = "My dog died yesterday"
# t4 = "I don't love you anymore..!"
# t5 = "I am hungry for human flesh"
# t6 = "I am happy"
# t7 = "I am angry"

# texts = [t1]

# for text in texts: 
#     features = create_feature(text, nrange=(1, 4))
#     features = vectorizer.transform(features)
#     prediction = clf.predict(features)[0]
#     print(text, emoji_dict[prediction])    
