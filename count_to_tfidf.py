import argparse
from tqdm import tqdm
import json
from collections import defaultdict
import math

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dfvar', dest='dfvar', type=int, help='Which variant of idf computation is used? (0 or 1)')
parser.add_argument('-t','--tfvar', dest='tfvar', type=int, help='Which variant of tf compuration is used 0, 1 or 2')
parser.add_argument('-b', '--black', dest='black', type=str, help='Path to a list of blacklist terms')
parser.add_argument('-i', '--input', dest='input', type=str, help='Path to input count_term')
parser.add_argument('-o', '--out', dest='output', type=str, help='Path to output folder')
args = parser.parse_args()


with open(args.black, 'r', encoding = 'utf-8') as f:
    tab = set(json.load(f))
    taboo = set()
    for n in tab:
        a = n.replace(" ","_")
        taboo.add(a)

def clean_dbpedia_name(name):
    return name.split("/")[-1]

def cal_df(graph_dict):
    df = defaultdict(int)
    for com, vals in graph_dict.items():
        terms = set()
        if type(vals) == list:
            for v in vals:
                name = clean_dbpedia_name(v[0])
                if name not in taboo:
                    terms.add(name)
        else:
            for name in vals.keys():
                if name not in taboo:
                    terms.add(name)
        for term in terms:
            df[term] +=1
    return df

def cal_idf(df, N, variant=0):
    idf = {}
    for term, f in df.items():
        if variant == 0:
            idf[term] = math.log(N/df[term], math.e)
        elif variant == 1:
            idf[term] = math.log(1 + N/df[term], math.e)
        else:
            print("Unknown variant")
    return idf

def cal_tf(graph_dict, variant=0, alpha=0.4):
    com_tf = {}
    for com, vals in graph_dict.items():
        tf = defaultdict(int)
        max_tf = 0
        if type(vals) == list:
            for v in vals:
                term = clean_dbpedia_name(v[0])
                if term in taboo:
                    continue
                val = v[1]
                tf[term] += val
                if val >= max_tf:
                    max_tf = val
        else:
            for term, val in vals.items():
                if term in taboo:
                    continue
                tf[term] += val
                if val >= max_tf:
                    max_tf = val
        if variant == 0:
            com_tf[com] = tf
        elif variant == 1:
            for term, val in tf.items():
                if val > 0:
                    tf[term] = 1 + math.log(val)
                else:
                    tf[term] = 0
            com_tf[com] = tf
        elif variant == 2:
            for term, val in tf.items():
                tf[term] = alpha + (1-alpha)*val/max_tf
            com_tf[com] = tf
        else:
            print('Unknown variant')
    return com_tf

def cal_tfidf(tf_com, idf):
    result = {}
    for com, tf_dict in tf_com.items():
        tfidf = {}
        for term, tf in tf_dict.items():
            tfidf[term] = tf*idf[term]
        result[com] = tfidf
    return result

if __name__ == '__main__':
    with open(args.input, 'r') as f:
        graph_dict = json.load(f)
    df = cal_df(graph_dict)
    idf = cal_idf(df, len(graph_dict), args.dfvar)
    tf = cal_tf(graph_dict, variant=args.tfvar)
    tfidf = cal_tfidf(tf, idf)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(tfidf, f, ensure_ascii=False)