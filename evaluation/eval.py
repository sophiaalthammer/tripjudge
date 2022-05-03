#
# Based on Sebastian Hofstätter code published in https://github.com/sebastian-hofstaetter/matchmaker
#

import os
import copy
import time
import glob
from typing import Dict, Tuple, List
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import scipy.stats as stats

from evaluation.utils import *
import time


def read_bm25(test_results_path):
    with open(test_results_path, 'r') as file:
       lines = file.readlines()
       test_results = {}
       for line in lines:
           splitted_line = line.split(' ')
           query_id = splitted_line[0]
           doc_id = splitted_line[2]
           score = float(splitted_line[4])
           if test_results.get(query_id):
               doc_list = test_results.get(query_id)
               doc_list.append((doc_id, score))
               test_results.update({query_id:doc_list})
           else:
               test_results.update({query_id:[(doc_id, score)]})
    return test_results

def read_run(test_results_path):
    with open(test_results_path, 'r') as file:
        lines = file.readlines()
        test_results = {}
        for line in lines:
            splitted_line = line.split('\t')
            query_id = splitted_line[0]
            doc_id = splitted_line[1]
            score = float(splitted_line[3])
            if test_results.get(query_id):
                doc_list = test_results.get(query_id)
                doc_list.append((doc_id, score))
                test_results.update({query_id:doc_list})
            else:
                test_results.update({query_id:[(doc_id, score)]})
    return test_results


def eval_run(test_results_path, qrels_path, binarization_point, qrels_with_irrel=False, with_j=False):
    if 'bm25' in test_results_path:
        test_results = read_bm25(test_results_path)
    else:
        test_results = read_run(test_results_path)

    metrics = None
    ranked_results = unrolled_to_ranked_result(test_results)

    if with_j:
        # for j coverage we need all the irrelevant and relevant judgements
        qrels = load_qrels_with_irrel(qrels_path)
        ranked_results_j = {}
        for query_id, value in qrels.items():
            if ranked_results.get(query_id):
                doc_ids_ranked_list = ranked_results.get(query_id)
                ranked_list_per_query_id = []
                for doc_id in doc_ids_ranked_list:
                    if doc_id in list(value.keys()):
                        ranked_list_per_query_id.append(doc_id)
                ranked_results_j.update({query_id: ranked_list_per_query_id})

        ranked_results = ranked_results_j
        # but for the evaluation of nDCG only the positive judgements should be in the qrels set
        qrels = load_qrels(qrels_path)
    elif qrels_with_irrel:
        qrels = load_qrels_with_irrel(qrels_path)
    else:
        qrels = load_qrels(qrels_path)

    metrics = calculate_metrics_plain(ranked_results,qrels,binarization_point)

    if 'dctr' in qrels_path:
        qrels = 'dctr'
    elif 'raw' in qrels_path:
        qrels = 'raw'
    else:
        qrels = 'annotation'

    metric_file_path = os.path.join('/'.join(test_results_path.split('/')[:-1]), 'test_top200_rerank_head_{}_joption'.format(qrels)+"-metrics.csv")
    save_fullmetrics_oneN(metric_file_path, metrics, -1, -1)
    return metrics


def get_metrics_for_multiple_qrels_runs(qrels, runs, path, binarization_point):
    metrics = {}
    for qrel in qrels:
        if 'dctr' in qrel:
            qrel_name = 'dctr'
        elif 'raw' in qrel:
            qrel_name = 'raw'
        else:
            qrel_name = 'annotation'
        metrics.update({qrel_name: {}})
        for run in runs:
            test_results_path = os.path.join(path, run)
            run_name = run.split('/')[0]

            run_metrics = eval_run(test_results_path, qrel, binarization_point)
            metrics.get(qrel_name).update({run_name: run_metrics})
    return metrics


def compute_kendalls_tau_between_collections(pairs, measures, metrics, path):
    with open(os.path.join(path, 'kendalltau.txt'), 'w') as f:
        for pair in pairs:
            for measure in measures:
                runs = metrics.get(pair[0])
                measure_numbers = []
                for value in runs.values():
                    measure_numbers.append(value.get(measure))
                runs_per_measure = dict(zip(runs.keys(), measure_numbers))
                runs_per_measure_sorted_1 = dict(
                    sorted(runs_per_measure.items(), key=lambda item: item[1], reverse=True))

                runs = metrics.get(pair[1])
                measure_numbers = []
                for value in runs.values():
                    measure_numbers.append(value.get(measure))
                runs_per_measure = dict(zip(runs.keys(), measure_numbers))
                runs_per_measure_sorted_2 = dict(
                    sorted(runs_per_measure.items(), key=lambda item: item[1], reverse=True))

                tau, p_value = stats.kendalltau(list(runs_per_measure_sorted_1.keys()),
                                                list(runs_per_measure_sorted_2.keys()))
                print('pair {} measure {} and kendall tau {} pvalue {}'.format(pair, measure, tau, p_value))

                f.write('pair {} measure {} and kendall tau {} pvalue {}\n'.format(pair, measure, tau, p_value))
        f.close()


def compute_kendalls_tau_between_metrics(pairs, tests, metrics, path):
    with open(os.path.join(path, 'kendalltau_withinstability_ndcg3.txt'), 'w') as f:
        for test in tests:
            for pair in pairs:
                runs = metrics.get(test)
                measure_numbers = []
                for value in runs.values():
                    measure_numbers.append(value.get(pair[0]))
                runs_per_measure = dict(zip(runs.keys(), measure_numbers))
                runs_per_measure_sorted_1 = dict(
                    sorted(runs_per_measure.items(), key=lambda item: item[1], reverse=True))

                runs = metrics.get(test)
                measure_numbers = []
                for value in runs.values():
                    measure_numbers.append(value.get(pair[1]))
                runs_per_measure = dict(zip(runs.keys(), measure_numbers))
                runs_per_measure_sorted_2 = dict(
                    sorted(runs_per_measure.items(), key=lambda item: item[1], reverse=True))

                tau, p_value = stats.kendalltau(list(runs_per_measure_sorted_1.keys()),
                                                list(runs_per_measure_sorted_2.keys()))
                print('pair {} test {} and kendall tau {} pvalue {}'.format(pair, test, tau, p_value))

                f.write('pair {} test {} and kendall tau {} pvalue {}\n'.format(pair, test, tau, p_value))
        f.close()

if __name__ == "__main__":
    test_results_path = 'path_to/ensemble-output.txt'
    qrels_path_anno = './data/qrels_2class.txt'
    qrels_path_dctr = 'path_to/qrels.dctr.head.test.txt'
    qrels_path_raw = 'path_to/qrels.raw.head.test.txt'
    binarization_point = 1

    runs = ['bm25/bm25_top1k_head.test.txt',
           'scibert_dot/top1k_head_dctr-output.txt',
           'pubmedbert_dot/top1k_head_dctr-output.txt',
           'colbert_scibert/test_top200_rerank_head_dctr-output.txt',
           'colbert_pubmedbert/test_top200_rerank_head_dctr-output.txt',
           'bert_cat/test_top200_rerank_head_dctr-output.txt',
           '3bert_fix_ensemble_avg/top200_rerank_head_dctr-ensemble-output.txt']

    path = 'output_path/'
    qrels = [qrels_path_anno, qrels_path_dctr, qrels_path_raw]

    # evaluate all runs again with the irrelevant ones to find coverage!
    for qrel in qrels:
        for run in runs:
            test_results_path = os.path.join(path, run)
            eval_run(test_results_path, qrel, binarization_point, qrels_with_irrel=True, with_j=True)

    # evaluate annotated runs with joption
    for run in runs:
        test_results_path = os.path.join(path, run)
        eval_run(test_results_path, qrels_path_anno, binarization_point, qrels_with_irrel=True, with_j=True)


    # evaluate multiple runs for kendalls tau
    metrics = get_metrics_for_multiple_qrels_runs(qrels, runs, path, binarization_point)
    measures = ['Recall@100', 'MRR@10', 'nDCG@5', 'nDCG@10', 'nDCG@3']
    pairs = [('annotation', 'dctr'), ('annotation', 'raw'), ('dctr', 'raw')]

    compute_kendalls_tau_between_collections(pairs, measures, metrics, path)

    # stability of test collection: kendalls tau between metrics
    measures = ['Recall@100', 'MRR@10', 'nDCG@5', 'nDCG@10', 'nDCG@3']
    pairs = [('nDCG@3', 'Recall@100'), ('nDCG@3', 'MRR@10'), ('nDCG@3', 'nDCG@5')]
    tests = ['annotation', 'raw', 'dctr']
    compute_kendalls_tau_between_metrics(pairs, tests, metrics, path)


