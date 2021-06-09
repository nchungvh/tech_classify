import json
import networkx as nx 
from networkx.readwrite import json_graph
import sys
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-i' ,'--input', dest='input', type=str, help='Input path to graph dicts')
# parser.add_argument('-d' ,'--dbpedia', dest='dbpedia', type=float, help='Weight for weighted_filtered_dbpedia')
parser.add_argument('-in' ,'--indeed', dest='indeed', type=float, help='Weight for tech_filtered_indeed')
parser.add_argument('-p' ,'--patent', dest='patent', type=float, help='Weight for tech_filtered_patent')
parser.add_argument('-c','--crunch', dest='crunch', type=float, help='Weight for weighted_filtered_crunch')
parser.add_argument('-t','--twitter', dest='twitter', type=float, help='Weight for weighted_filtered_twitter')
parser.add_argument('-ht','--hashtag', dest='hashtag', type=float, help='Weight for weighted_filtered_hashtag')
parser.add_argument('-w','--website', dest='website', type=float, help='Weight for weighted_filtered_website')
parser.add_argument('-o','--output', dest='output', type=str, help='Output path', default='.')
parser.add_argument('--avg', dest='avg', action='store_true')
parser.add_argument('--no-avg', dest='avg', action='store_false')
parser.set_defaults(avg=True)
args = parser.parse_args()

w = [args.indeed, args.patent, args.crunch, args.twitter, args.hashtag, args.website]
print('weight for graph:\tDBpedia:{}\tEnt-Fishing:{}\tCrunchBase:{}'.format(w[0],w[1],w[2]))

# db = json.load(open('weighted_filtered_dbpedia.json'))
# ef = json.load(open('weighted_filtered_entfishing_tfidf_english.json'))
# cb = json.load(open('weighted_filtered_crunch.json'))
# tw = json.load(open('weighted_filtered_twitter.json'))
# twh = json.load(open('weighted_twitter_hashtag.json'))
# web = json.load(open('weighted_filtered_website.json'))

# db = json.load(open(args.input + '/tech_filtered_dbpedia.json'))
ind = json.load(open(args.input + '/tech_filtered_indeed.json'))
pa = json.load(open(args.input + '/tech_filtered_patent.json'))
cb = json.load(open(args.input + '/tech_filtered_crunch.json'))
tw = json.load(open(args.input + '/tech_filtered_twitter.json'))
twh = json.load(open(args.input + '/tech_filtered_hashtag.json'))
web = json.load(open(args.input + 'tech_filtered_website.json'))

print(len(ind), len(pa), len(cb), len(tw), len(twh), len(web))

merge = nx.Graph()
count = 0
com_dict = {}
ent_dict = {}

def add_graph(json_file, weight):
    global merge 
    global count 
    global com_dict
    global ent_dict
    curr = count + 0

    json_file = {u:v for (u,v) in json_file.items() if len(v) > 0}
    print('num company in input graph: {}'.format(len(json_file)))
    merge_node = 0
    for com in json_file:
        if com not in com_dict:
            merge.add_node(count, label = [0], content = [com], content_detail = [[com]])
            com_dict[com] = count 
            count += 1
            merge_node += 1
        for ent in json_file[com]:
            if ent not in ent_dict:
                name = ent.replace("_"," ")
                merge.add_node(count, label = [1], content = [name], content_detail = [[name]])
                ent_dict[ent] = count 
                count += 1
            try:
                merge[com_dict[com]][ent_dict[ent]]['weight'] += json_file[com][ent] * weight
            except:
                merge.add_edge(com_dict[com], ent_dict[ent], weight = json_file[com][ent] * weight)
    
    print('num company full/merged in input graph:  {} {}'.format(len(json_file), len(json_file) - merge_node ))
    print('num after/before node of merge graph: {} {}'.format(curr, count))

# def add_ent_graph(json_file, weight):
#     global merge 
#     global count 
#     global com_dict
#     global ent_dict
#     curr = count + 0

#     json_file = {u:v for (u,v) in json_file.items() if len(v) > 0}
#     print('num company in input graph: {}'.format(len(json_file)))
#     merge_node = 0
#     for com in json_file:
#         if com not in com_dict:
#             merge.add_node(count, label = [0], content = [com], content_detail = [[com]])
#             com_dict[com] = count 
#             count += 1
#             merge_node += 1
#         for ent in json_file[com]:
#             if ent not in ent_dict:
#                 merge.add_node(count, label = [1], content = [ent], content_detail = [[ent]])
#                 ent_dict[ent] = count 
#                 count += 1
#             try:
#                 merge[com_dict[com]][ent_dict[ent]]['weight'] += 1 * weight
#             except:
#                 merge.add_edge(com_dict[com], ent_dict[ent], weight = 1 * weight)
    
#     print('num company full/merged in input graph:  {} {}'.format(len(json_file), len(json_file) - merge_node ))
#     print('num after/before node of merge graph: {} {}'.format(curr, count))

# add_graph(db, w[0])
if args.indeed != 0:
    add_graph(ind, args.indeed)

if args.patent != 0:
    add_graph(pa, args.patent)
if args.crunch != 0:
    add_graph(cb, args.crunch)
    
if args.twitter != 0:
    add_graph(tw, args.twitter)
if args.hashtag != 0:
    add_graph(twh, args.hashtag)
if args.website != 0:
    add_graph(web, args.website)

# Remove edges with weight 0 
print(nx.info(merge))
removed = []
for u,v,a in merge.edges(data=True):
    if a['weight'] == 0:
        removed.append((u,v))

for u, v in removed:
    merge.remove_edge(u,v)

# Remove isolated nodes
merge.remove_nodes_from(list(nx.isolates(merge)))

print(nx.info(merge))

print('num nodes: {}'.format(len(merge.nodes)))
from networkx.readwrite import json_graph
res = json_graph.node_link_data(merge)
res['nodes'] = [
    {
        'id': node['id'],
        'label': node['label'],
        'content': node['content'],
        'content_detail': node['content_detail'],
        'train': True,
        'test': False
    }
    for node in res['nodes']]

if args.avg:
    res['links'] = [
        {
            'source': link['source'],
            'target': link['target'],
            'weight': link['weight']/sum(w),
            'test_removed': False,
            'train_removed': False
        }
        for link in res['links']]
else:
    res['links'] = [
        {
            'source': link['source'],
            'target': link['target'],
            'weight': link['weight'],
            'test_removed': False,
            'train_removed': False
        }
        for link in res['links']]


w_str = "_".join([str(e) for e in w])

try:
    os.mkdir(args.output)
except OSError:
    print ("Creation of the directory %s failed" % args.output)
else:
    print ("Successfully created the directory %s " % args.output)

with open("tfidf_234/merge-G_" + args.output + ".json", 'w') as outfile:
        json.dump(res, outfile)




