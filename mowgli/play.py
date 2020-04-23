import models, datasets

message = "oki doki"

intent, probability = models.classify_intent(
    models.get_classifier(),
    models.get_vectorizer(),
    datasets.labels('../resources/labels.csv'),
    message
)

print (intent, probability)
