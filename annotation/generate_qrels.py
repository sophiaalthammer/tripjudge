#
# Based on Sebastian Hofstätter code published in https://github.com/sebastian-hofstaetter/fira-trec-19-dataset
#
# Create qrels (trec-format) from the FiRA raw annotation file, either for the doc snippets or aggregated per doument
# -------------------------------
#

import argparse
import os
import sys
sys.path.append(os.getcwd())
import csv
from collections import Counter
import pandas as pd

#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--annotations', action='store',  dest='annotations', required=True)
parser.add_argument('--qrels-out', action='store',  dest='out_file', required=True)
parser.add_argument('--mode', action='store',  dest='mode', required=True, choices=['4class','2class'])

parser.add_argument('--old-qrels', action='store',  dest='old_qrels', required=False)

args = parser.parse_args()


old_qrels = {}
if args.old_qrels != None:
    with open(args.old_qrels,"r",encoding="utf8") as in_file:
        for l in in_file:
            l = l.split(" ")
            old_qrels[(l[0],l[2])] = int(l[3])

#########################

annotations_tsv = csv.reader(open(args.annotations), delimiter="\t", quotechar='"')

queries_remove_path = './preprocessing/queries_remove_cleaned.tsv'
queries_remove = pd.read_csv(queries_remove_path, sep='\t')

label_0 = '0_NOT_RELEVANT'
label_1 = '1_TOPIC_RELEVANT_DOES_NOT_ANSWER'
label_2 = '2_GOOD_ANSWER'
label_3 = '3_PERFECT_ANSWER'

if args.mode == '4class':
    label_to_int = {
    '0_NOT_RELEVANT':0,
    '1_TOPIC_RELEVANT_DOES_NOT_ANSWER':1,
    '2_GOOD_ANSWER':2,
    '3_PERFECT_ANSWER':3}
else:
    label_to_int = {
    'non_rel':0,
    'rel':1}


def heuristic_majority_voting(annotations):
    # if only one annotation we remove the query-document pair
    if len(annotations) == 1:
        return None

    labels = [l[1] for l in annotations]
    label_agg = Counter(labels)

    # perfect agreement (for all len > 1)
    if len(label_agg) == 1:
        return labels[0]
    else:

        maxval = label_agg.most_common()[0][1]
        pool = []
        for item, count in label_agg.most_common():
            if count == maxval:
                pool.append(item)
            else:
                break

        if len(pool) == 1:
            return pool[0]
        else:
            # lowest wins
            if label_0 in pool:
                return label_0
            elif label_1 in pool:
                return label_1
            elif label_2 in pool:
                return label_2
            else:
                return label_3


def heuristic_majority_voting2class(annotations):
    # if only one annotation we remove the query-document pair
    if len(annotations) == 1:
        return None

    labels = ["rel" if (l[1] == label_3 or l[1] == label_2) else "non_rel" for l in data]
    label_agg = Counter(labels)

    # perfect agreement (for all len > 1)
    if len(label_agg) == 1:
        return labels[0]
    else:
        maxval = label_agg.most_common()[0][1]
        pool = []
        for item, count in label_agg.most_common():
            if count == maxval:
                pool.append(item)
            else:
                break

        if len(pool) == 1:
            return pool[0]
        else:
            # heuristic lowest wins
            return 'non_rel'

#########################

annotated_pairs = {}
for row in list(annotations_tsv)[1:]:
    pair = (row[8], row[7])
    if pair not in annotated_pairs:
        annotated_pairs[pair]=[]
    annotated_pairs[pair].append(row)

queries_remove_ids = list(queries_remove['query_id'])
for key in list(annotated_pairs.keys()):
    if key[0] in queries_remove_ids:
        annotated_pairs.pop(key)


out_pairs = set()

out_file_path = args.out_file
mode = args.mode

with open(out_file_path,"w") as out_file:
    labels_per_pair = {}

    for pair,data in annotated_pairs.items():
        if mode == '4class':
            label = heuristic_majority_voting(data)
        else:
            label = heuristic_majority_voting2class(data)
        if label:
            qd_pair = (pair[0],pair[1])
            if qd_pair not in labels_per_pair:
                labels_per_pair[qd_pair] = []
            labels_per_pair[qd_pair].append(label_to_int[label])

    for pair,data in labels_per_pair.items():
        label = max(data)

        out_file.write("{} Q0 {} {}\n".format(pair[0],pair[1],label))
        out_pairs.add((pair[0],pair[1]))

    # if args.old_qrels:
    #     for pair,data in old_qrels.items():
    #         if data <= 1:
    #             if (pair[0],pair[1]) in out_pairs:
    #                 print("found duplicate",(pair[0],pair[1]))
    #                 continue
    #             out_file.write("{} Q0 {} {}\n".format(pair[0],pair[1],data))
