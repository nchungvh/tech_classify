# tech_classify

## Download abstracts
Use db_extract_abstract.py to download dbpedia abstracts

## Construct a tech classifier and use the classifier to filter entity terms from graph_dicts
1. Run tech_classify.py to train a Lightning module for classifi **Wiki** abstracts
2. Run classify_abstracts.py to classify **DBpedia** abstracts and save them for use in next step
3. Run tech_filter_from_abstract_tech.py to filter graph dicts using classified abstracts in the previous step
