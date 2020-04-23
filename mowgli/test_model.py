import csv

import models, datasets

labelfile = "../resources/labels.csv"
testfile  = "../resources/test.csv"
# testfile  = "../resources/train.csv"

with open(labelfile, newline='') as csvfile:
    data = list(csv.reader(csvfile))

label_dic = {}

for s in data:
    label_dic[s[0]] = s[1]

# print (label_dic)

with open(testfile, newline='') as csvfile:
    data = list(csv.reader(csvfile))

total   = 0
correct = 0

for s in data:
    right_label = label_dic[s[0]]
    message     = s[1]

    intent, probability = models.classify_intent(
        models.get_classifier(),
        models.get_vectorizer(),
        datasets.labels('../resources/labels.csv'),
        message
    )

    if intent == right_label:
        correct += 1
    else:
        print ("%s recognized as %s, correct is %s" % (message, intent, right_label))
    total += 1

print ("Results: %d from %d correct = %4.2f percent" % (correct, total, (correct/total)*100.0))
