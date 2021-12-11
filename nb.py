from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report

# we use pickle to store the intermediate results of conditional probabilities
def loadObject(name):
    with open(name, "rb") as f:
        return pickle.load(f)


def saveObject(name, model):
    with open(name, "wb") as f:
        return pickle.dump(model, f)


data = pd.read_table("test.csv", sep=",", names=["label", "text"])
# Tokenize the text using a lambda function
data["split_data"] = data["text"].apply(
    lambda x: ["#S"] + word_tokenize(str(x)) + ["#E"]
)
# Construct corpus dictionary for the entire dataset
dictionary = {}
for index, row in data.iterrows():
    text = set(row["split_data"])
    for word in text:
        # Parameters for naive bayes model, not as good as MultinomialNB, didn't use eventually
        # I leave it here for further optimization if interested in the future
        dictionary[word] = dictionary.get(word, {1: 0, 2: 0})
        dictionary[word][row["label"]] += 1

# Construct two dictionaries for conditional probability for each individual word, given positive sentiment or negative sentiment
tempdict1 = {}
tempdict2 = {}
sentences1 = data[data["label"] == 1]["split_data"].values
sentences2 = data[data["label"] == 2]["split_data"].values
for i in range(len(sentences1)):
    tempdict1[i] = {}
    for t in range(len(sentences1[i])):
        word = sentences1[i][t]
        # if there is no word, fill in an empty array
        tempdict1[i][word] = tempdict1[i].get(word, [])
        tempdict1[i][word].append(t)

for i in range(len(sentences2)):
    tempdict2[i] = {}
    for t in range(len(sentences2[i])):
        word = sentences2[i][t]
        tempdict2[i][word] = tempdict2[i].get(word, [])
        tempdict2[i][word].append(t)


def getConditionalProbability(word, tempdict, sentences):
    d = {}
    allNextWords = 0
    for i, value in tempdict.items():
        nextArr = value.get(word, [])
        for wordIndex in nextArr:
            try:
                nexWord = sentences[i][wordIndex + 1]
                d[nexWord] = d.get(nexWord, 0)
                d[nexWord] += 1
                allNextWords += 1
            except:
                pass
    d1 = {}
    for key, value in d.items():
        d1[key] = value / allNextWords
    return d1


# Calculate conditional probability for each word in negative sentiment
conditionalProbDictLabel1 = {}
for word in dictionary:
    conditionalProbDictLabel1[word] = getConditionalProbability(
        word, tempdict1, sentences1
    )

# Calculate conditional probability for each word in positve sentiment
conditionalProbDictLabel2 = {}
for word in dictionary:
    conditionalProbDictLabel2[word] = getConditionalProbability(
        word, tempdict2, sentences2
    )

# save
saveObject("conditionalProbDictLabel1.pkl", conditionalProbDictLabel1)
saveObject("conditionalProbDictLabel2.pkl", conditionalProbDictLabel2)

#
np.random.seed(0)


def getRandomWord(conditionalProbDictLabel1, word):
    d = conditionalProbDictLabel1[word]
    p = np.array(list(d.values()))
    # Look at np.random.choice: Given the prob distribution of next word, we random select word based on given probability
    nextWord = np.random.choice(list(d.keys()), p=p.ravel())
    return nextWord


# generate text for label1 and label2
generateLabel1 = []
for i in range(5000):
    nextWord = getRandomWord(conditionalProbDictLabel1, "#S")
    text = ["#S", nextWord]
    for t in range(500):  # sentence max lengh
        nextWord = getRandomWord(conditionalProbDictLabel1, nextWord)
        text.append(nextWord)
        if nextWord == "#E":
            break
    generateLabel1.append(text)


generateLabel2 = []
for i in range(5000):
    nextWord = getRandomWord(conditionalProbDictLabel2, "#S")
    text = ["#S", nextWord]
    for t in range(500):  #
        nextWord = getRandomWord(conditionalProbDictLabel2, nextWord)
        text.append(nextWord)
        if nextWord == "#E":
            break
    generateLabel2.append(text)

# Store them in the form we want
generateLabelText = []
for item in generateLabel1:
    if item[0] == "#S":
        item = item[1:]
    if item[-1] == "#E":
        item = item[:-1]
    generateLabelText.append({"label": 1, "text": " ".join(item)})

for item in generateLabel2:
    if item[0] == "#S":
        item = item[1:]
    if item[-1] == "#E":
        item = item[:-1]
    generateLabelText.append({"label": 2, "text": " ".join(item)})

generateLabelTextDf = pd.DataFrame(generateLabelText)


def calculLossFunction(data):
    data["predictProbaLabel1"] = 1
    data["predictProbaLabel2"] = 1
    data[["predictProbaLabel1", "predictProbaLabel2"]] = mnb.predict_proba(
        cv.transform(data["text"])
    )

    def calculLoss(x):
        if x["label"] == 1:
            return -np.log(x["predictProbaLabel1"])
        else:
            return -np.log(x["predictProbaLabel2"])

    data["loss"] = data.apply(calculLoss, axis=1)
    return data["loss"].mean()


# train NB model
cv = CountVectorizer()
X_train_count = cv.fit_transform(data["text"])
mnb = MultinomialNB().fit(X_train_count, data["label"])
data["predict"] = mnb.predict(X_train_count)

# Look at the result for fitting original data
# calcul f1 acc recall
print(classification_report(data["label"], data["predict"]))
# calcul mean logloss
print(calculLossFunction(data))
data[["label", "text", "predict", "loss"]].to_csv("test_preict.csv", index=None)

# Test the result for fitting the generated sentences(data)
generateLabelTextDf["predict"] = mnb.predict(cv.transform(generateLabelTextDf["text"]))
# calcul f1 acc recall
print(
    classification_report(generateLabelTextDf["label"], generateLabelTextDf["predict"])
)
# calcul mean logloss
calculLossFunction(generateLabelTextDf)
generateLabelTextDf[["label", "text", "predict", "loss"]].to_csv(
    "generateLabelText_preict.csv", index=None
)

# Test the result for real world data
reviews = pd.read_csv("Reviews.csv")
reviews["text"] = reviews["Text"]
reviews["label"] = reviews["Score"].apply(lambda x: 2 if x == 5 else 1)
# calcul f1 acc recall
reviews["predict"] = mnb.predict(cv.transform(reviews["text"]))
print(classification_report(reviews["label"], reviews["predict"]))
# calcul mean loss
calculLossFunction(reviews)

reviews[["label", "text", "predict", "loss"]].to_csv("Reviews_preict.csv", index=None)
