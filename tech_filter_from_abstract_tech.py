import json
import re
import numpy as np
import torch
import glob 
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-gd', '--graph_dict', dest='gd', type=str, help='Graph dict type')
parser.add_argument('-gp', '--graph_dict_path', dest='gp', type=str, help='Path to graph dict')
parser.add_argument('-ab', '--abstract', dest='ab', type=str, help='Path to abstract_tech_ path')
parser.add_argument('-o', '--out', dest='out', type=str, help='Path to output folder')
args = parser.parse_args()


def load_graph_dict(graph_dict_type):
    # if graph_dict_type == 'hashtag':
    #     with open(args.gp + "/weighted_twitter_" + graph_dict_type + "_temp.json") as f:
    #         return json.load(f)
    # else:
    #     with open(args.gp + "/weighted_filtered_" + graph_dict_type + "_temp.json") as f:
    #         return json.load(f)
    with open(args.gp + "/standard_" + graph_dict_type + ".json") as f:
        return json.load(f)

def load_abstract_tech(graph_dict_type):
    with open(args.ab + "/abstract_tech_" + graph_dict_type + ".json") as f:
        return json.load(f)

if __name__ == '__main__':
    # import pdb;pdb.set_trace()
    graph_dict = load_graph_dict(args.gd)
    print("#### Graph dict loaded")
    term2tech = load_abstract_tech(args.gd)

    new_graph_dict = {}
    for com in graph_dict:
        d = {}
        for term in graph_dict[com]:
            if term in term2tech and term2tech[term] == 1:
                d[term] = graph_dict[com][term]
        new_graph_dict[com] = d

    with open(args.out + '/tech_filtered_' + args.gd + '.json', 'w', encoding = 'utf-8') as f:
        json.dump(new_graph_dict,f, ensure_ascii=False)   
                