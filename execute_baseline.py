import time
start = time.time()

from utils.dataset import Dataset
from utils.baseline_MLPC import Baseline
from utils.scorer import report_score

def execute_demo(language):
    data = Dataset(language)

# ***** Baseline system ***** #
    system = Baseline(language)

    system.train(data.trainset)

    predictions = system.test(data.devset)

    gold_labels = [sent['gold_label'] for sent in data.devset]

    print("{}: {} training - {} dev".format(language, len(data.trainset), len(data.devset)))
    report_score(gold_labels, predictions)


if __name__ == '__main__':
    execute_demo('english')
    execute_demo('spanish')
    print('Runtime: {0:0.1f} seconds'.format(time.time() - start))


















