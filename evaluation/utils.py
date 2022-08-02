#
# Based on Sebastian Hofstätter code published in https://github.com/sebastian-hofstaetter/matchmaker
#

import os
import copy
import time
import glob
from typing import Dict, Tuple, List
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from allennlp.nn.util import move_to_device
from allennlp.common import Params, Tqdm

import time
import numpy as np
from rich.console import Console
from rich.table import Table
from rich import box


global_metric_config = {
    "MRR+Recall@":[10,20,100,200,1000], # multiple allowed
    "nDCG@":[3,5,10,20,1000], # multiple allowed
    "MAP@":1000, #only one allowed
}

def calculate_metrics_plain(ranking, qrels, binarization_point=1.0, return_per_query=False):
    '''
    calculate main evaluation metrics for the given results (without looking at candidates),
    returns a dict of metrics
    '''

    ranked_queries = len(ranking)
    ap_per_candidate_depth = np.zeros((ranked_queries))
    coverage_per_candidate_depth = np.zeros((len(global_metric_config["nDCG@"]), ranked_queries))
    rr_per_candidate_depth = np.zeros((len(global_metric_config["MRR+Recall@"]), ranked_queries))
    rank_per_candidate_depth = np.zeros((len(global_metric_config["MRR+Recall@"]), ranked_queries))
    recall_per_candidate_depth = np.zeros((len(global_metric_config["MRR+Recall@"]), ranked_queries))
    ndcg_per_candidate_depth = np.zeros((len(global_metric_config["nDCG@"]), ranked_queries))
    evaluated_queries = 0

    for query_index, (query_id, ranked_doc_ids) in enumerate(ranking.items()):
        if query_id in qrels:
            evaluated_queries += 1

            relevant_ids = np.array(list(qrels[query_id].keys()))  # key, value guaranteed in same order
            relevant_grades = np.array(list(qrels[query_id].values()))
            sorted_relevant_grades = np.sort(relevant_grades)[::-1]

            num_relevant = relevant_ids.shape[0]
            np_rank = np.array(ranked_doc_ids)
            relevant_mask = np.in1d(np_rank, relevant_ids)  # shape: (ranking_depth,) - type: bool

            binary_relevant = relevant_ids[relevant_grades >= binarization_point]
            binary_num_relevant = binary_relevant.shape[0]
            binary_relevant_mask = np.in1d(np_rank, binary_relevant)  # shape: (ranking_depth,) - type: bool

            # check if we have a relevant document at all in the results -> if not skip and leave 0
            if np.any(binary_relevant_mask):

                # now select the relevant ranks across the fixed ranks
                ranks = np.arange(1, binary_relevant_mask.shape[0] + 1)[binary_relevant_mask]

                #
                # ap
                #
                map_ranks = ranks[ranks <= global_metric_config["MAP@"]]
                ap = np.arange(1, map_ranks.shape[0] + 1) / map_ranks
                ap = np.sum(ap) / binary_num_relevant
                ap_per_candidate_depth[query_index] = ap

                # mrr only the first relevant rank is used
                first_rank = ranks[0]

                for cut_indx, cutoff in enumerate(global_metric_config["MRR+Recall@"]):

                    curr_ranks = ranks.copy()
                    curr_ranks[curr_ranks > cutoff] = 0

                    recall = (curr_ranks > 0).sum(axis=0) / binary_num_relevant
                    recall_per_candidate_depth[cut_indx, query_index] = recall

                    #
                    # mrr
                    #

                    # ignore ranks that are out of the interest area (leave 0)
                    if first_rank <= cutoff:
                        rr_per_candidate_depth[cut_indx, query_index] = 1 / first_rank
                        rank_per_candidate_depth[cut_indx, query_index] = first_rank

            if np.any(relevant_mask):

                # now select the relevant ranks across the fixed ranks
                ranks = np.arange(1, relevant_mask.shape[0] + 1)[relevant_mask]

                grades_per_rank = np.ndarray(ranks.shape[0], dtype=int)
                for i, id in enumerate(np_rank[relevant_mask]):
                    grades_per_rank[i] = np.where(relevant_ids == id)[0]

                grades_per_rank = relevant_grades[grades_per_rank]

                #
                # ndcg = dcg / idcg
                #
                for cut_indx, cutoff in enumerate(global_metric_config["nDCG@"]):
                    #
                    # get idcg (from relevant_ids)
                    idcg = (sorted_relevant_grades[:cutoff] / np.log2(1 + np.arange(1, min(num_relevant, cutoff) + 1)))

                    curr_ranks = ranks.copy()
                    curr_ranks[curr_ranks > cutoff] = 0

                    coverage_per_candidate_depth[cut_indx, query_index] = (curr_ranks > 0).sum() / float(cutoff)

                    with np.errstate(divide='ignore', invalid='ignore'):
                        c = np.true_divide(grades_per_rank, np.log2(1 + curr_ranks))
                        c[c == np.inf] = 0
                        dcg = np.nan_to_num(c)

                    nDCG = dcg.sum(axis=-1) / idcg.sum()

                    ndcg_per_candidate_depth[cut_indx, query_index] = nDCG

    avg_coverage = coverage_per_candidate_depth.sum(axis=-1) / evaluated_queries
    mrr = rr_per_candidate_depth.sum(axis=-1) / evaluated_queries
    relevant = (rr_per_candidate_depth > 0).sum(axis=-1)
    non_relevant = (rr_per_candidate_depth == 0).sum(axis=-1)

    avg_rank = np.apply_along_axis(lambda v: np.mean(v[np.nonzero(v)]), -1, rank_per_candidate_depth)
    avg_rank[np.isnan(avg_rank)] = 0.

    median_rank = np.apply_along_axis(lambda v: np.median(v[np.nonzero(v)]), -1, rank_per_candidate_depth)
    median_rank[np.isnan(median_rank)] = 0.

    map_score = ap_per_candidate_depth.sum(axis=-1) / evaluated_queries
    recall = recall_per_candidate_depth.sum(axis=-1) / evaluated_queries
    nDCG = ndcg_per_candidate_depth.sum(axis=-1) / evaluated_queries

    local_dict = {}

    for cut_indx, cutoff in enumerate(global_metric_config["MRR+Recall@"]):
        local_dict['MRR@' + str(cutoff)] = mrr[cut_indx]
        local_dict['Recall@' + str(cutoff)] = recall[cut_indx]
        local_dict['QueriesWithNoRelevant@' + str(cutoff)] = non_relevant[cut_indx]
        local_dict['QueriesWithRelevant@' + str(cutoff)] = relevant[cut_indx]
        local_dict['AverageRankGoldLabel@' + str(cutoff)] = avg_rank[cut_indx]
        local_dict['MedianRankGoldLabel@' + str(cutoff)] = median_rank[cut_indx]

    for cut_indx, cutoff in enumerate(global_metric_config["nDCG@"]):
        local_dict['Avg_coverage@' + str(cutoff)] = avg_coverage[cut_indx]
        local_dict['nDCG@' + str(cutoff)] = nDCG[cut_indx]

    local_dict['QueriesRanked'] = evaluated_queries
    local_dict['MAP@' + str(global_metric_config["MAP@"])] = map_score

    if return_per_query:
        return local_dict, rr_per_candidate_depth, ap_per_candidate_depth, recall_per_candidate_depth, ndcg_per_candidate_depth
    else:
        return local_dict


def load_qrels(path):
    with open(path,'r') as f:
        qids_to_relevant_passageids = {}
        for l in f:
            try:
                l = l.strip().split()
                qid = l[0]
                if float(l[3]) > 0.0001:
                    if qid not in qids_to_relevant_passageids:
                        qids_to_relevant_passageids[qid] = {}
                    qids_to_relevant_passageids[qid][l[2]] = float(l[3])
            except:
                raise IOError('\"%s\" is not valid format' % l)
        return qids_to_relevant_passageids

def load_qrels_with_irrel(path):
    with open(path,'r') as f:
        qids_to_relevant_passageids = {}
        for l in f:
            try:
                l = l.strip().split()
                qid = l[0]
                if qid not in qids_to_relevant_passageids:
                    qids_to_relevant_passageids[qid] = {}
                qids_to_relevant_passageids[qid][l[2]] = float(l[3])
            except:
                raise IOError('\"%s\" is not valid format' % l)
        return qids_to_relevant_passageids

def unrolled_to_ranked_result(unrolled_results):
    ranked_result = {}
    for query_id, query_data in unrolled_results.items():
        local_list = []
        # sort the results per query based on the output
        for (doc_id, output_value) in sorted(query_data, key=lambda x: x[1], reverse=True):
            local_list.append(doc_id)
        ranked_result[query_id] = local_list
    return ranked_result

def save_fullmetrics_oneN(file, metrics, epoch_number, batch_number):
    # write csv header once
    if not os.path.isfile(file):
        with open(file, "w") as metric_file:
            metric_file.write("sep=,\nEpoch,After_Batch," + ",".join(k for k, v in metrics.items())+"\n")
    # append single row
    with open(file, "a") as metric_file:
        metric_file.write(str(epoch_number) + "," +str(batch_number) + "," + ",".join(str(v) for k, v in metrics.items())+"\n")
