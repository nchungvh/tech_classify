# tech_classify

## Extract entities from texts then store as graph_dict: {com1: {term1: count_term1, term2: count_term2, ...}, com2:....}

## Convert count graph_dict to tfidf graph_dict
python count_to_tfidf.py -d 1 -t 2 -i count_terms_234/count_term_indeed.json -o tfidf_234/tfidf_indeed.json -b entity_reject_field.json

## Download abstracts for tfidf graph_dict
Use db_extract_abstract.py to download dbpedia abstracts
python db_extract_abstract.py -p tfidf_234/weighted_indeed.json -o indeed -f tfidf_234/

## Construct a tech classifier and use the classifier to filter entity terms from graph_dicts
1. Run tech_classify.py to train a Lightning module for classifying **Wiki** abstracts
2. Run classify_abstracts.py to classify **DBpedia** abstracts and save them for use in next step
    python classify_abstracts.py -xy abstract_embs.npz -ck lightning_logs/version_14/checkpoints/epoch\=83-step\=175139.ckpt -gd indeed -ab tfidf_234/abstracts/ --out tfidf_234/
3. Standardize company names according to crunchbase
    python standardize_comp_names.py -i tfidf_234/ -o tfidf_234/
4. Run tech_filter_from_abstract_tech.py to filter graph dicts using classified abstracts in the previous step
    python tech_filter_from_abstract_tech.py -gd hashtag -gp tfidf_234/ -ab tfidf_234/ -o tfidf_234/
5. Merge all tech_filtered_graph_dicts to a json bipartie graph
    python merge.py -i tfidf_234/ -in 1 -p 1 -c 1 -t 1 -ht 1 -w 1 -o tfidf_234/