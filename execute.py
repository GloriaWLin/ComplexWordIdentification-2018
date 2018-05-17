import time
start = time.time()

from utils.dataset import Dataset
from utils.scorer import report_score
from utils.improved_LR import ImprovedSys

def execute_demo(language):
    data = Dataset(language)

    trainset_small = []
    for i in range(int(5*len(data.trainset)/5)):
        trainset_small.append(data.trainset[i])
        
#    trainset_small = []
#    for i in range(1000):
#        trainset_small.append(data.trainset[i])
        
# ***** Improved system ***** #
    system = ImprovedSys(language)
    
    if language == 'english':
        
        BoW, lexicon = system.create_engBoWLexicon_wiki()
#        BoW = system.create_BoW_data(data.trainset, data.devset, data.testset)
#        BoW = system.create_engBoW_brown()
#        lexicon = system.create_engLexicon_original(data.tnewsset, data.twikinewsset, data.twikiset)
#        pos_weight = system.create_posWeight(data.trainset)
#        aoa_dict, mean_rate = system.create_aoaDict()
        
        system.train_eng(trainset_small, BoW, lexicon)
        dev_predictions = system.test_eng(data.devset, BoW, lexicon)
        test_predictions = system.test_eng(data.testset, BoW, lexicon)

    if language == 'spanish':
        
        BoW, lexicon = system.create_espBoWLexicon()
#        BoW = system.create_BoW_data(data.trainset, data.devset, data.testset)
#        BoW = system.create_espBoW_wiki()
#        pos_weight = system.create_posWeight(data.trainset)
        
        system.train_esp(trainset_small, BoW, lexicon)
        dev_predictions = system.test_esp(data.devset, BoW, lexicon)
        test_predictions = system.test_esp(data.testset, BoW, lexicon)

    dev_gold_labels = [sent['gold_label'] for sent in data.devset]
    test_gold_labels = [sent['gold_label'] for sent in data.testset]
    
    print("{}: {} training - {} dev".format(language, len(trainset_small), 
          len(data.devset)))
    report_score(dev_gold_labels, dev_predictions)
    print("{}: {} training - {} test".format(language, len(trainset_small), 
          len(data.testset)))
    report_score(test_gold_labels, test_predictions)
    
#    print("Gold labels:")
#    print(test_gold_labels)
#    print("Predicted labels:")
#    print(list(test_predictions))
    

if __name__ == '__main__':
    execute_demo('english')
    execute_demo('spanish')
    print('Runtime: {0:0.1f} seconds'.format(time.time() - start))


















