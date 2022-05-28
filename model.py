#Importing Libraries
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import names


#Creating a function that returns the dictionary that contains the positive, negative and neutral words 
def word_feats(words):
    return dict([(word, True) for word in words])
#Telling the machine different types of positive, negative and neutral words
positive_vocab = [ 'awesome', 'outstanding', 'fantastic', 'terrific', 'good', 'nice', 'great', ':)' ]
negative_vocab = [ 'bad', 'terrible','useless', 'hate', ':(' ,'disgusting','ugh','not','never','sad','bad','abuse']
neutral_vocab = [ 'movie','the','sound','was','is','actors','did','know','words','I' ]


#Adding them to the dictionary accordingly
positive_features = [(word_feats(pos), 'pos') for pos in positive_vocab]
negative_features = [(word_feats(neg), 'neg') for neg in negative_vocab]
neutral_features = [(word_feats(neu), 'neu') for neu in neutral_vocab]

#training
train_set = negative_features + positive_features + neutral_features

#Naive Bayes classifiers are a collection of classification algorithms based on Bayes’ Theorem. 
# It is not a single algorithm but a family of algorithms where all of them share a common principle, 
# i.e. every pair of features being classified is independent of each other.
#.train--->training the classifier
classifier = NaiveBayesClassifier.train(train_set) 

# Predict
neg = 0
pos = 0
#asks for user input
sentence = input("Enter your sentence: ")
sentence = sentence.lower()
words = sentence.split(' ')
for word in words:
    #.classify--->return: the most appropriate label for the given featureset.
    classResult = classifier.classify( word_feats(word))
    if classResult == 'neg':
        neg = neg + 1
    if classResult == 'pos':
        pos = pos + 1

print('Positive: ' + str(float(pos)/len(words)))
print('Negative: ' + str(float(neg)/len(words)))
