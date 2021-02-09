
# python read_json.py 5
# python read_json.py 6
# python read_json.py 7
# python read_json.py 8
# python read_json.py 9
# python read_json.py 10
# python read_json.py 11
# python read_json.py 12


import json
import os
import glob 
import sys
import requests
import eventlet
import multiprocessing.pool as Pool

fs = glob.glob('*e55e')
output = []
for f in fs:
    with open(f,'r', encoding = 'utf8')as f:
        for line in f:
            output.append(json.loads(line))

index = int(sys.argv[1])
files = output[index*100000:(index+1) * 100000]
# temp = output[100:200]
def get_data(url):
    response = requests.get(url)
    x = response.text.split('class="lead"')[1].split('</p>')[0]
    response.close()
    return x
num_procs = os.cpu_count()
print(num_procs)
# eventlet.monkey_patch()
results = []
for subindex in range(0,len(files), 1000):
    temp = files[subindex:subindex+1000]
    re = []
    pool = Pool.ThreadPool(processes=num_procs)
    for tech in temp:
        re.append(pool.apply_async(get_data, args = (tech['uri'],)))
    pool.close()
    pool.join()
    re = [r.get() for r in re]
    results.extend(re)

    for i in range(len(results)):
        files[i]['abtract'] = results[i]

    with open("output_{}_{}.json".format(index,subindex), 'w') as outfile:
        json.dump(files, outfile)