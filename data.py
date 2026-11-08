#depreciated, unused concept file

import csv, json
files=['1.tsv','2.tsv','3.tsv','4.tsv']
data={}
for file in files:
    with open(file) as f:
        data[file]=list(csv.DictReader(f, delimiter='\t'))
with open('feeder.json','w') as f:
    json.dump(data,f)
