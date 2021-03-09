import json
import os
import requests
from tqdm.notebook import tqdm
import multiprocessing.pool as Pool
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('-p','--path', dest='path', type=str, help='Path to a graph dict')
parser.add_argument('-o','--out', dest='out', type=str, help='Name suffix for output')
args = parser.parse_args()

missing = []

def extract_abstract(entity):
    url = "https://dbpedia.org/page/" + entity
    response = requests.get(url)
    try:
        x = response.text.split('class="lead"')[1].split('</p>')[0][1:]
        response.close()
        return x
    except:
        return ''
    
sub_graph = json.load(open(args.path)) #graph_dict
entities = []
for com in sub_graph:
    entities.extend(list(sub_graph[com].keys()))
entities = list(set(entities))
print('number of entities: {}'.format(len(entities)))
result = []

num_procs = os.cpu_count()
print(num_procs)
# eventlet.monkey_patch()
results = []
for subindex in range(0,len(entities), 1000):
    temp = entities[subindex:subindex+1000]
    re = []
    pool = Pool.ThreadPool(processes=num_procs)
    for tech in temp:
        re.append(pool.apply_async(extract_abstract, args = (tech,)))
    pool.close()
    pool.join()
    re = [r.get() for r in re]
    results.extend(re)
    output = {}
    for i in range(len(results)):
        output[entities[i]] = results[i]
    print(len(results))
    with open("abstracts/{}_dbabstract_{}.json".format(args.out, subindex), 'w', encoding='utf8') as outfile:
        json.dump(output, outfile, ensure_ascii=False)

if len(missing) > 0:
    with open("missing_" + args.out +".pkl", 'wb') as handle:
        pickle.dump(missing, handle, protocol=pickle.HIGHEST_PROTOCOL)