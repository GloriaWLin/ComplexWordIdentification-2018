import csv

class Dataset(object):

    def __init__(self, language):
        self.language = language

        trainset_path = "datasets/{}/{}_Train.tsv".format(language, language.capitalize())
        devset_path = "datasets/{}/{}_Dev.tsv".format(language, language.capitalize())
        testset_path = "datasets/{}/{}_Test.tsv".format(language, language.capitalize())

        self.trainset = self.read_dataset(trainset_path)
        self.devset = self.read_dataset(devset_path)
        self.testset = self.read_dataset(testset_path)

        # datasets for the English lexicon
        tnews_path = "datasets/english/original/News_Train.tsv"
        twikiNews_path = "datasets/english/original/WikiNews_Train.tsv"
        twiki_path = "datasets/english/original/Wikipedia_Train.tsv"
        
        self.tnewsset = self.read_dataset(tnews_path)
        self.twikinewsset = self.read_dataset(twikiNews_path)
        self.twikiset = self.read_dataset(twiki_path)

    def read_dataset(self, file_path):
        with open(file_path, encoding="utf8") as file:
            fieldnames = ['hit_id', 'sentence', 'start_offset', 'end_offset', 'target_word', 'native_annots',
                          'nonnative_annots', 'native_complex', 'nonnative_complex', 'gold_label', 'gold_prob']
            reader = csv.DictReader(file, fieldnames=fieldnames, delimiter='\t')

            dataset = [sent for sent in reader]

        return dataset
    
        

            
