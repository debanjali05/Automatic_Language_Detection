# Automatic Language Detection

This is a N-gram based Language Detection model trained to classify 5 languages: English, French, German, Italian and Dutch.

## Dataset
Wortschatz Leipzig Corpora Collection ([here](https://wortschatz.uni-leipzig.de/en/download)) is used for training and testing.
For training, 30K sentences for each language was used while for testing 10K sentences for each language was used.

## Requirement
Python 3.6.2 and NLTK 3.2.4

## Running 
To train the model
```bash
python language_detection_training
```
To test the model
```bash
python language_detection_testing
```

Note: The model checkpoints are saved in "checkpoints" folder. Create this folder and update the directory path in [utils.py](https://github.com/debanjali05/Automatic_Language_Detection/blob/master/utils.py) before running.
Also, the correct directory path for the Dataset should be updates in this file. Adding/Changing language and setting the value of N can be done in this file.
