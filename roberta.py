from transformers import pipeline
classifier = pipeline('zero-shot-classification', model='roberta-large-mnli')
sequence_to_classify = "one day I will see the world"
candidate_labels = ['travel', 'cooking', 'dancing']
a = classifier(sequence_to_classify, candidate_labels)
print(a)