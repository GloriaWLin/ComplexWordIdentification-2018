# ComplexWordIdentification-2018

This work presents two systems for complex word identification using datasets from CWI Share Task 2018. Both systems were based on random forest classifiers, and they were trained with a word's length, frequency, the number of synonyms and a corpus--based lexicon. The final system produced F1 scores of 0.80 and 0.74 for the English and Spanish datasets respectively.

### Implementation

Use execute.py or execute_baseline.py to call the dataset reader (dataset.py), the result evaluator (scorer.py) and the relevant systems (baseline_{algorithm name}.py or improved_{algorithm name}.py). Before implementing the codes, make sure the relevant datasets are included under a subfolder named "datasets".

All codes are compatible with Python 3. Additional package, Pyphen, is required to be installed.


