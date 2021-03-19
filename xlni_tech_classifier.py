import json
from tqdm.autonotebook import tqdm

from transformers import pipeline
classifier = pipeline("zero-shot-classification",
                      model="joeddav/xlm-roberta-large-xnli", device=0)
# print(next(classifier.parameters()).is_cuda)

candidate_labels = ["quantity", "person", "organization", "brand", "facility", "location", "product", "event", "work" "of" "art", "technology", "animal"]

tech_paths = ["wiki_abstracts/abtract_tech.json", "wiki_abstracts/empty_tech_intro.json"]
non_tech_paths = ["wiki_abstracts/abtract_non_tech.json", "wiki_abstracts/empty_non_tech_intro.json"]

import pdb; pdb.set_trace()
tech_predicts = []
for path in tech_paths:
    js = json.load(open(path))
    for term in tqdm(js):
        abstract = js[term]
        if abstract != '':
            labels = classifier(js[term], candidate_labels)['labels']
            scores = classifier(js[term], candidate_labels)['scores']
            if labels[0] == 'technology' or labels[1] == 'technology':
                tech_predicts.append(1)
            else:
                tech_predicts.append(0)
        else:
            tech_predicts.append(-1)
# import pdb; pdb.set_trace()

non_tech_predicts = []            
for path in non_tech_paths:
    js = json.load(open(path))
    for term in tqdm(js):
        abstract = js[term]
        if abstract != '':
            labels = classifier(js[term], candidate_labels)['labels']
            scores = classifier(js[term], candidate_labels)['scores']
            if labels[0] == 'technology' or labels[1] == 'technology':
                non_tech_predicts.append(1)
            else:
                non_tech_predicts.append(0)
        else:
            non_tech_predicts.append(-1)            

import pdb; pdb.set_trace()