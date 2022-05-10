import os
import json
from sklearn import metrics
labels = []
data = "Cameras"
with open("data/er_magellan/Structured/" + data + "/test.txt", "r") as f:
    for i in f.readlines():
        j = i.split("\t")
        j = j[2].split("\n")
        labels.append(j[0])
orders = [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]
for order in range(6):
    g = open("data/er_magellan/Structured/" + data + "/test.txt", "w")
    with open("data/er_magellan/Structured/" + data + "/test2.txt", "r") as f:
        for i in f.readlines():
            i = i.split("\t")
            for index in range(2):
                j = i[index].split("COL")
                temp1 = j[orders[order][0]]
                temp2 = j[orders[order][1]]
                temp3 = j[orders[order][2]]
                j[1] = orders[order][0]
                j[2] = orders[order][1]
                j[3] = orders[order][2]
                i[index] = 'COL'.join(j)
            i = '\t'.join(i)
            g.writelines(i)
    g.close()
    os.system("!CUDA_VISIBLE_DEVICES=0 python matcher.py   --task Structured/" + data + "   --input_path "
              "data/er_magellan/Structured/" + data + "/test.txt   --output_path output/output_small.jsonl   --lm "
              "roberta   --max_len 128   --use_gpu   --fp16   --checkpoint_path checkpoints/")
    predicts = []
    with open('output/output_small.jsonl', 'r', encoding='utf8') as fp:
        for line in fp.readlines():
            js_l = json.loads(line)
            predicts.append(js_l["match"])
    print(orders)
    print(metrics.f1_score(labels, predicts))
