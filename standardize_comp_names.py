import json
import pickle as pkl
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i' ,'--input', dest='input', type=str, help='Input path to graph dicts')
parser.add_argument('-o','--output', dest='output', type=str, help='Output path', default='.')
args = parser.parse_args()

with open(args.input + "/tfidf_website.json", 'r', encoding = 'utf-8') as f:
    web = json.load(f)
web_coms = set(web.keys())

with open(args.input +"/tfidf_indeed.json", 'r', encoding = 'utf-8') as f:
    indeed = json.load(f)
in_coms = set(indeed.keys())

with open(args.input +"/tfidf_patent.json", 'r', encoding = 'utf-8') as f:
    patent = json.load(f)
pa_coms = set(patent.keys())

with open(args.input +"/tfidf_hashtag.json", 'r', encoding = 'utf-8') as f:
    ha = json.load(f)
ha_coms = set(ha.keys())

with open(args.input +"/tfidf_crunch.json", 'r', encoding = 'utf-8') as f:
    crunch = json.load(f)
cb_coms = set(crunch.keys())

with open(args.input +"/tfidf_twitter.json", 'r', encoding='utf-8') as f:
    tw = json.load(f)
tw_coms = set(tw.keys())

coms = set()
coms.update(cb_coms)
coms.update(in_coms)
coms.update(pa_coms)
coms.update(tw_coms)
coms.update(web_coms)
coms.update(ha_coms)

com_list = sorted(list(coms))

mapping = {}
for com in coms:
    for suffix in [' AG', ' SA', ' GmbH', ' GA', ' S.A.', ' Sàrl',' G.A',' Ltd', ' ag', ' sa', ' gmbh', ' ga', ' s.a.', ' sàrl', ' g.a.', ' ltd', ' gmbH']:
        if com + suffix in coms:
            mapping[com] = com + suffix

def standardize_graph_dict(graph_dict, mapping):
    st_dict = {}
    for com in graph_dict:
        if com in mapping:
            st_dict[mapping[com]] = graph_dict[com]
        else:
            st_dict[com] = graph_dict[com]
    return st_dict
    
new_crunch = standardize_graph_dict(crunch, mapping)
new_in = standardize_graph_dict(indeed, mapping)
new_pa = standardize_graph_dict(patent, mapping)
new_ha = standardize_graph_dict(ha, mapping)
new_tw = standardize_graph_dict(tw, mapping)
new_web = standardize_graph_dict(web, mapping)

with open(args.output + '/standard_crunch.json','w', encoding='utf8') as outfile:
    json.dump(new_crunch, outfile, ensure_ascii=False)
with open(args.output + '/standard_website.json','w', encoding='utf8') as outfile:
    json.dump(new_web, outfile, ensure_ascii=False)
with open(args.output + '/standard_indeed.json','w', encoding='utf8') as outfile:
    json.dump(new_in, outfile, ensure_ascii=False)
with open(args.output + '/standard_patent.json','w', encoding='utf8') as outfile:
    json.dump(new_pa, outfile, ensure_ascii=False)
with open(args.output + '/standard_hashtag.json','w', encoding='utf8') as outfile:
    json.dump(new_ha, outfile, ensure_ascii=False)
with open(args.output + '/standard_twitter.json','w', encoding='utf8') as outfile:
    json.dump(new_tw, outfile, ensure_ascii=False)
