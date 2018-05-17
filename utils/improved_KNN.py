from sklearn.neighbors import KNeighborsClassifier

import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import cess_esp
from nltk.corpus import brown

import pyphen
import re
from collections import Counter
from scipy import stats
#import pandas as pd
#from gensim.models import Word2Vec
#import numpy as np
#import os, random

class ImprovedSys(object):

    def __init__(self, language):
        self.language = language
        # from 'Multilingual and Cross-Lingual Complex Word Identification' (Yimam et. al, 2017)
        if language == 'english':
            self.avg_word_length = 5.3
        else:  # spanish
            self.avg_word_length = 6.2

        self.model = KNeighborsClassifier(n_neighbors=40)

# ***** Processing corpora/datasets for bag-of-words & simple-word lexicon ***** #
    def create_BoW_data(self, trainset, devset, testset): # training, developing & testing sets
        BoW = Counter()
        wordRE = re.compile('[^\W_]+', re.UNICODE) # account for all characters including accents

        for sent in trainset:
            wdList = wordRE.findall(sent['sentence'])
            for word in wdList:
                BoW[word] += 1.
        for sent in devset:
            wdList = wordRE.findall(sent['sentence'])
            for word in wdList:
                BoW[word] += 1.
        for sent in testset:
            wdList = wordRE.findall(sent['sentence'])
            for word in wdList:
                BoW[word] += 1.
        return BoW
# -------- English corpora --------
    def create_engBoWLexicon_wiki(self): # the Wikipedia corpus from http://www.cs.upc.edu/~nlp/wikicorpus/
#        # create file list to iterate through the entire directory (too time-consuming)
#        dir_list = []
#        for subdir, dirs, files in os.walk("datasets/english/raw.en"):
#            for file in files:
#                dir_list.append(file)
#        # filter out files with no content
#        valid_list = []
#        for file in dir_list:
#            if os.stat("datasets/english/raw.en/"+file).st_size != 0:
#                valid_list.append(file)
#        # randomly sample 5 files to reduce processing time (takes about 15 min)
#        random.seed(31415926)
#        sample_list = random.sample(valid_list, 5)
#        
#        wdDict = Counter()
#        wordRE = re.compile('[^\W_]+', re.UNICODE)

#        for filename in sample_list:
#                with open("datasets/english/raw.en/"+filename, encoding="ISO-8859-1") as file:
#                    next(file)
#                    for line in file:
#                        l = wordRE.findall(line)
#                        for word in l:
#                            wdDict[word] += 1.
        wiki_path = "datasets/english/raw.en/englishText_1230000_1240000" # choose the largest file
        wordRE = re.compile('[^\W_]+', re.UNICODE)

        wdDict = Counter()
        with open(wiki_path, encoding="ISO-8859-1") as file:
            next(file)
            for line in file:
                l = wordRE.findall(line)
                for word in l:
                    wdDict[word] += 1.
        lexicon = {}
        threshold = stats.scoreatpercentile(list(wdDict.values()), 40)
        for word, count in wdDict.items():
            if count >= threshold:
                lexicon[word] = count
        return wdDict, lexicon
    
    def create_engBoW_brown(self): # nltk brown corpus
        BoW = Counter()
        for word in brown.words():
            BoW[word] += 1.
        return BoW

    def create_engLexicon_brown(self): # nltk brown corpus
        engAll = Counter()
        engDict = {}
        for word in brown.words():
            engAll[word] += 1.
        threshold = stats.scoreatpercentile(list(engAll.values()), 20)
        for word, count in engAll.items():
            if count >= threshold:
                engDict[word] = count
        return engDict

    def create_engLexicon_ogden(self): # Ogden's vocabulary
        ogden_path = "datasets/english/Ogden_vocab.tsv"
        wordRE = re.compile('[^\W_]+', re.UNICODE)

        wdDict = Counter()
        with open(ogden_path, encoding="utf8") as file:
            for line in file:
                l = wordRE.findall(line)
                for word in l:
                    wdDict[word] += 1.
        return wdDict

    def create_engLexicon_original(self, dataset1, dataset2, dataset3): # News, WikiNews & Wikipedia
        lexicon = Counter()
        for sent in dataset1:
            if sent['gold_label'] == "0":
                for word in sent['target_word'].split():
                    lexicon[word] += 1.
        for sent in dataset2:
            if sent['gold_label'] == "0":
                for word in sent['target_word'].split():
                    lexicon[word] += 1.
        for sent in dataset3:
            if sent['gold_label'] == "0":
                for word in sent['target_word'].split():
                    lexicon[word] += 1.
        return lexicon
# -------- Spanish corpora --------
    def create_espBoW_wiki(self): # the Wikipedia corpus from http://www.cs.upc.edu/~nlp/wikicorpus/
#        # create file list to iterate through the entire directory (too time-consuming)
#        dir_list = []
#        for subdir, dirs, files in os.walk("datasets/spanish/raw.es"):
#            for file in files:
#                dir_list.append(file)
#        # filter out files with no content
#        valid_list = []
#        for file in dir_list:
#            if os.stat("datasets/spanish/raw.es/"+file).st_size != 0:
#                valid_list.append(file)
#        # randomly sample 5 files to reduce processing time (takes about 15 min)
#        random.seed(31415926)
#        sample_list = random.sample(valid_list, 5)
#        
#        wdDict = Counter()
#        wordRE = re.compile('[^\W_]+', re.UNICODE)

#        for filename in sample_list:
#                with open("datasets/spanish/raw.es/"+filename, encoding="ISO-8859-1") as file:
#                    next(file)
#                    for line in file:
#                        l = wordRE.findall(line)
#                        for word in l:
#                            wdDict[word] += 1.
        wiki_path = "datasets/spanish/raw.es/spanishText_20000_25000" # choose the largest file
        wordRE = re.compile('[^\W_]+', re.UNICODE)

        wdDict = Counter()
        with open(wiki_path, encoding="ISO-8859-1") as file:
            next(file)
            for line in file:
                l = wordRE.findall(line)
                for word in l:
                    wdDict[word] += 1.
        return wdDict

    def create_espBoWLexicon(self): # CESS esp corpus
        BoW = Counter()
        lexicon = {}
        for word in cess_esp.words():
            BoW[word] += 1.
        threshold = stats.scoreatpercentile(list(BoW.values()), 10)
        for word, count in BoW.items():
            if count >= threshold:
                lexicon[word] = count
        return BoW, lexicon

# Feature 1 & 2: number of characters & number of tokens
    def extract_wdLen(self, targetWord):
        len_chars = len(targetWord) / self.avg_word_length
        wordRE = re.compile('[^\W_]+', re.UNICODE)

        len_tokens = len(wordRE.findall(targetWord))
        return [len_chars, len_tokens]

# Feature 3: number of vowels per word
    def extract_nVow(self, targetWord): 
        n = 0.
        wordRE = re.compile('[^\W_]+', re.UNICODE)

        for word in wordRE.findall(targetWord):
            c = 0.
            for char in word:
                if char in "aeiouAEIOUáéíóú": # Spanish and English have the same vowels
                    c += 1
            n += c
        len_token = len(wordRE.findall(targetWord))
        if len_token == 0:
            return 0
        else:
            return n/len_token

# Feature 4: number of syllables per word
    def extract_eng_nSyl(self, targetWord):
        dic = pyphen.Pyphen(lang='en')
        wordRE = re.compile('[^\W_]+', re.UNICODE)

        n = 0.
        for word in wordRE.findall(targetWord):
            n += len(dic.inserted(word).split("-"))
        len_token = len(wordRE.findall(targetWord))
        if len_token == 0:
            return 0
        else:
            return n/len_token

    def extract_esp_nSyl(self, targetWord):
        dic = pyphen.Pyphen(lang='es')
        wordRE = re.compile('[^\W_]+', re.UNICODE)

        n = 0.
        for word in wordRE.findall(targetWord):
            n += len(dic.inserted(word).split("-"))
        len_token = len(wordRE.findall(targetWord))
        if len_token == 0:
            return 0
        else:
            return n/len_token

# Feature 5: word frequency
    def extract_unigramProb(self, targetWord, BoW):
        count = []
        wordRE = re.compile('[^\W_]+', re.UNICODE)

        for word in wordRE.findall(targetWord):
            if word in list(BoW.keys()):
                count.append(BoW[word])
            else:
                count.append(1)
        if len(count) == 0:
            return 0
        else:
            return min(count)/len(BoW)

# Feature 6: simple word lexicon
    def extract_simWd(self, targetWord, lexicon):
        wordRE = re.compile('[^\W_]+', re.UNICODE)

        for word in wordRE.findall(targetWord):
            score = 0.
            if word not in lexicon.keys():
                score += 1
        len_token = len(wordRE.findall(targetWord))
        if len_token == 0:
            return 0
        else:
            return score


# Feature 7: number of word senses
    def extract_nSense(self, targetWord):
        n = []
        wordRE = re.compile('[^\W_]+', re.UNICODE)

        for word in wordRE.findall(targetWord):
            n.append(len(wn.synsets(word)))
        len_token = len(wordRE.findall(targetWord))
        if len_token == 0:
            return 0
        else:
            return min(n)

## Feature: Word age-of-acquisition (English) (unused)
#    def create_aoaDict(self):
#        aoa_path = 'datasets/english/AoA_ratings_Kuperman.csv'
#        df = pd.read_csv(aoa_path, sep=',', header=0, index_col=0)
#        
#        aoa_dict = {}
#        for word, row in df.iterrows():
#            if row['Rating.Mean'] >= 1:
#                aoa_dict[word] = row['Rating.Mean']
#        
#        sumv = 0.
#        count = 0.
#        for k,v in aoa_dict.items():
#            sumv += v
#            count += 1
#        mean_rate = sumv/count
#        return aoa_dict, mean_rate
#    
#    def extract_aoaRate(self, targetWord, aoa_dict, mean_rate):
#        wordRE = re.compile('[^\W_]+', re.UNICODE)
#        rate = []
#        for word in wordRE.findall(targetWord):
#            if word in list(aoa_dict.keys()):
#                rate.append(aoa_dict[word])
#            else:
#                rate.append(mean_rate)
#        len_token = len(wordRE.findall(targetWord))
#        if len_token == 0:
#            return 0
#        else:
#            return max(rate)
#
## Feature: pos tags (unused)
#    def create_posWeight(self, trainset):
#        complex_words = []
#        wordRE = re.compile('[^\W_]+', re.UNICODE)
#
#        for sent in trainset:
#            if sent['gold_label'] == "1":
#                for word in wordRE.findall(sent['target_word']):
#                    complex_words.append(word)
#        tagged = nltk.tag.pos_tag(complex_words)
#        tag_fd = nltk.FreqDist(tag for (word, tag) in tagged)
#        pos_weight = {pos: count for (pos, count) in tag_fd.most_common()}
#        return pos_weight
#    
#    def extract_posWeight(self, targetWord, pos_weight):
#        weight = 0.
#        wordRE = re.compile('[^\W_]+', re.UNICODE)
#
#        for word in wordRE.findall(targetWord):
#            postag = nltk.tag.pos_tag([word])[0][1]
#            if postag in list(pos_weight.keys()):
#                weight += pos_weight[postag]
#        return weight
#
## Feature: number of annotators that deemed the word to be complex (unused)
#    def extract_nAnnotator(self, sentence):
#        return [sentence['native_complex'], sentence['nonnative_complex']]
#
## Feature: word embeddings (unused)
#    def create_word2vec(self, dataset):
#        sents = []
#        wordRE = re.compile('[^\W_]+', re.UNICODE)
#
#        for sent in dataset:
#            sents.append(wordRE.findall(sent['sentence'].replace("'","")))
#        model = Word2Vec(sents, size=10, window=5, min_count=1, workers=4)
#        return model
#    
#    def extract_wordEmbed(self, model, targetWord):
#        vec = np.zeros(10)
#        wordRE = re.compile('[^\W_]+', re.UNICODE)
#
#        for word in wordRE.findall(targetWord):
#            if word not in list(model.wv.vocab):
#                actual_word = [s for s in list(model.wv.vocab) if word in s][0]
#                vec = vec + model.wv.__getitem__(actual_word)
#            else:
#                vec = vec + model.wv.__getitem__(word)
#        return list(vec)
    
    
# ***** Training ***** #
#    def train_eng(self, trainset, BoW, lexicon, pos_weight):
    def train_eng(self, trainset, BoW, lexicon):
        X = []
        y = []
#        model = self.create_word2vec(trainset)
        for sent in trainset:
            features = (self.extract_wdLen(sent['target_word']))
            features.append(self.extract_nVow(sent['target_word']))
            features.append(self.extract_eng_nSyl(sent['target_word']))
            features.append(self.extract_unigramProb(sent['target_word'], BoW))
            features.append(self.extract_simWd(sent['target_word'],lexicon))
            features.append(self.extract_nSense(sent['target_word']))
#            features.append(self.extract_aoaRate(sent['target_word'], aoa_dict, mean_rate))
##            features += self.extract_nAnnotator(sent)
##            features += self.extract_wordEmbed(model, sent['target_word'])
#            features.append(self.extract_posWeight(sent['target_word'], pos_weight))
            X.append(features)
            y.append(sent['gold_label'])
        self.model.fit(X, y)

#    def train_esp(self, trainset, BoW, lexicon, pos_weight):
    def train_esp(self, trainset, BoW, lexicon):
        X = []
        y = []
#        model = self.create_word2vec(trainset)
        for sent in trainset:
            features = (self.extract_wdLen(sent['target_word']))
            features.append(self.extract_nVow(sent['target_word']))
            features.append(self.extract_esp_nSyl(sent['target_word']))
            features.append(self.extract_unigramProb(sent['target_word'], BoW))
            features.append(self.extract_simWd(sent['target_word'],lexicon))
            features.append(self.extract_nSense(sent['target_word']))
##            features += self.extract_nAnnotator(sent)
##            features += self.extract_wordEmbed(model, sent['target_word'])
#            features.append(self.extract_posWeight(sent['target_word'], pos_weight))
            X.append(features)
            y.append(sent['gold_label'])
        self.model.fit(X, y)

# ***** Testing ***** #
#    def test_eng(self, testset, BoW, lexicon, pos_weight):
#    def test_eng(self, testset, BoW, lexicon, aoa_dict, mean_rate):
    def test_eng(self, testset, BoW, lexicon):
        X = []
#        model = self.create_word2vec(testset)
        for sent in testset:
            features = (self.extract_wdLen(sent['target_word']))
            features.append(self.extract_nVow(sent['target_word']))
            features.append(self.extract_eng_nSyl(sent['target_word']))
            features.append(self.extract_unigramProb(sent['target_word'], BoW))
            features.append(self.extract_simWd(sent['target_word'],lexicon))
            features.append(self.extract_nSense(sent['target_word']))
#            features.append(self.extract_aoaRate(sent['target_word'], aoa_dict, mean_rate))
##            features += self.extract_nAnnotator(sent)
##            features += self.extract_wordEmbed(model, sent['target_word'])
#            features.append(self.extract_posWeight(sent['target_word'], pos_weight))
            X.append(features)
#        print(self.model.n_classes_)
        return self.model.predict(X)
    
#    def test_esp(self, testset, BoW, lexicon, pos_weight):
    def test_esp(self, testset, BoW, lexicon):
        X = []
#        model = self.create_word2vec(testset)
        for sent in testset:
            features = (self.extract_wdLen(sent['target_word']))
            features.append(self.extract_nVow(sent['target_word']))
            features.append(self.extract_esp_nSyl(sent['target_word']))
            features.append(self.extract_unigramProb(sent['target_word'], BoW))
            features.append(self.extract_simWd(sent['target_word'],lexicon))
            features.append(self.extract_nSense(sent['target_word']))
##            features += self.extract_nAnnotator(sent)
##            features += self.extract_wordEmbed(model, sent['target_word'])
#            features.append(self.extract_posWeight(sent['target_word'], pos_weight))
            X.append(features)
#        print(self.model.n_classes_)
        return self.model.predict(X)




