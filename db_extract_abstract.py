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

def get_url(entity):  
    try:
        params = {'text':entity}
        headers = {'accept': 'application/json'}
        try:
            url = 'https://api.dbpedia-spotlight.org/en/annotate'
            response = requests.get(url, params = params, headers = headers)
            url = json.loads(response.text)['Resources'][0]['@URI']
            response.close()
            response = requests.get(url)
            x = response.text.split('class="lead"')[1].split('</p>')[0]
            response.close()
            return x
        except:
            try:
                url = 'https://api.dbpedia-spotlight.org/de/annotate'
                response = requests.get(url, params = params, headers = headers)
                url = json.loads(response.text)['Resources'][0]['@URI']
                response.close()
                response = requests.get(url)
                x = response.text.split('class="lead"')[1].split('</p>')[0][1:]
                response.close()
                return x
            except:
                url = 'https://api.dbpedia-spotlight.org/fr/annotate'
                response = requests.get(url, params = params, headers = headers)
                url = json.loads(response.text)['Resources'][0]['@URI']
                response.close()
                response = requests.get(url)
                x = response.text.split('class="lead"')[1].split('</p>')[0][1:]
                response.close()
                return x
    except:
        missing.append(entity)
        return ''
#     finally:
#         return ''
#     return url

def extract_abstract(entity):
    entity = entity.replace(" ", "_").replace("#","").capitalize()
    url = "https://dbpedia.org/page/" + entity
    response = requests.get(url)
    x = response.text.split('class="lead"')[1].split('</p>')[0][1:]
    response.close()
    return x
    
sub_graph = json.load(open(args.path)) #graph_dict
entities = []
for com in sub_graph:
    entities.extend(list(sub_graph[com].keys()))
entities = list(set(entities))
entities = [i.replace('#','').replace('_',' ').lower() for i in entities]
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
        re.append(pool.apply_async(get_url, args = (tech,)))
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