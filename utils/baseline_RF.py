from sklearn.ensemble import RandomForestClassifier

class Baseline(object):

    def __init__(self, language):
        self.language = language
        # from 'Multilingual and Cross-Lingual Complex Word Identification' (Yimam et. al, 2017)
        if language == 'english':
            self.avg_word_length = 5.3
        else:  # spanish
            self.avg_word_length = 6.2

        self.model = RandomForestClassifier()

    def extract_wdLen(self, word):
        len_chars = len(word) / self.avg_word_length
        len_tokens = len(word.split(' '))
        return [len_chars, len_tokens]

    def train(self, trainset):
        X = []
        y = []
        for sent in trainset:
            features = self.extract_wdLen(sent['target_word'])
            X.append(features)
            y.append(sent['gold_label'])
        self.model.fit(X, y)
    
    def test(self, testset):
        X = []
        for sent in testset:
            features = self.extract_wdLen(sent['target_word'])
            X.append(features)
#        print(self.model.n_classes_)
        return self.model.predict(X)



