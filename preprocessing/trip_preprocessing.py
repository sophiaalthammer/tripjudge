import os
import csv
from allennlp.data.tokenizers import PretrainedTransformerTokenizer

def read_in_queries(queries_path):
    with open(queries_path, 'r') as f:
        lines = f.readlines()
        queries = []
        for line in lines:
            line_split = line.split('\t')
            query_id = line_split[0]
            text = line_split[1]
            queries.append([query_id, text.strip('\n')])
    return queries

def write_run(runs, file_path, name, cut_off, id_identifier):
    with open(os.path.join(file_path, name), 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        #tsv_writer.writerow(['query_id', 'doc_id', 'priority'])
        for run in runs:
            for query_id, value in run.items():
                for doc_id, rank in value.items():
                    if rank <= cut_off:
                        tsv_writer.writerow([id_identifier + query_id, id_identifier + doc_id, int(rank)])


def get_pairs_from_run(run):
    run_pairs_cutoff = []
    for query_id, value in run.items():
        for doc_id, rank in value.items():
            run_pairs_cutoff.append((query_id, doc_id))
    return run_pairs_cutoff


def run_intersection(run_scibert, run_bertcat, cut_off):
    run_scibert = run_cut_off(run_scibert, cut_off)
    run_bertcat = run_cut_off(run_bertcat, cut_off)

    run_pairs_scibert = get_pairs_from_run(run_scibert)
    run_pairs_bertcat = get_pairs_from_run(run_bertcat)

    run_scibert_intersect = list(set(run_pairs_scibert).intersection(set(run_pairs_bertcat)))

    # remove the pairs from the intersection from bertcat
    # keep the pair where it has the highest ranking!
    for (query_id, doc_id) in run_scibert_intersect:
        rank_scibert = run_scibert.get(query_id).get(doc_id)
        rank_bertcat = run_bertcat.get(query_id).get(doc_id)
        if rank_scibert <= rank_bertcat:
            run_bertcat.get(query_id).pop(doc_id)
        else:
            run_scibert.get(query_id).pop(doc_id)
    return run_scibert, run_bertcat


def run_cut_off(run, cut_off):
    run_cutted = {}
    for query_id, value in run.items():
        run_cutted.update({query_id:{}})
        for doc_id, rank in value.items():
            if rank <= cut_off:
                run_cutted.get(query_id).update({doc_id:rank})
    return run_cutted

def read_run_neural(run_file):
    with open(run_file, 'r') as f:
        lines = f.readlines()

        run = {}
        for line in lines:
            line_split = line.split('\t')
            query_id = line_split[0]
            doc_id = line_split[1]
            rank = int(line_split[2])

            if run.get(query_id):
                run.get(query_id).update({doc_id:rank})
            else:
                run.update({query_id:{}})
                run.get(query_id).update({doc_id: rank})
    return run


def read_run_bm25(run_file):
    with open(run_file, 'r') as f:
        lines = f.readlines()

        run = {}
        for line in lines:
            line_split = line.split(' ')
            query_id = line_split[0]
            doc_id = line_split[2]
            rank = int(line_split[3])

            if run.get(query_id):
                run.get(query_id).update({doc_id:rank})
            else:
                run.update({query_id:{}})
                run.get(query_id).update({doc_id: rank})
    return run


def read_in_runs(queries_path):
    with open(queries_path, 'r') as f:
        lines = f.readlines()
        queries = []
        for line in lines:
            line_split = line.split('\t')
            query_id = line_split[0]
            doc_id = line_split[1]
            priority = line_split[2]
            queries.append([query_id, doc_id, priority.strip('\n')  if priority.strip('\n') == 'priority' else priority.strip('\n')])
    return queries


def filter_collection_from_run(run_trip, col_trip):
    run_docs = [line[1] for line in run_trip if not line[1] == 'doc_id']
    print('Number of unique documents in the run is {}'.format(len(set(run_docs))))
    col_trip_filtered = []
    for doc in col_trip:
        if doc[0] in run_docs:
            col_trip_filtered.append(doc)
    print('Number of documents in filtered collection is {}'.format(len(col_trip_filtered)))
    return col_trip_filtered


def untokenize_sentence(text_tokenized):
    pretok_sent = ""
    for tok in text_tokenized:
        if tok.text.startswith("##"):
            pretok_sent += tok.text[2:]
        else:
            pretok_sent += " " + tok.text
    return pretok_sent.strip()


def truncate_collection(col_trip_filtered):
    tokenizer = PretrainedTransformerTokenizer('bert-base-uncased', max_length=512)
    coll_short = []
    for line in col_trip_filtered:
        text_tokenized = tokenizer.tokenize(line[1])[1:-1]
        text_untokenized = untokenize_sentence(text_tokenized)
        coll_short.append([line[0], text_untokenized])

    coll_short[0] = ['doc_id', 'doc_text']

    counter = 0
    counter_trip = 0
    for i in range(len(coll_short)):
        counter_trip += 1
        truncated = coll_short[i][1]
        original = col_trip_filtered[i][1]

        if len(truncated.split(' ')) + 10 <= len(original.split(' ')):
            counter += 1
    print('share of truncated document: {}', format(counter / counter_trip))
    return coll_short


def write_collection(coll_short, outfile_path):
    with open(outfile_path, 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow(['doc_id', 'doc_text'])
        for line in coll_short:
            tsv_writer.writerow(line)


if __name__ == "__main__":
    file_path = 'folder_path_to_runs/'

    # read the multiple runs
    run_bm25 = read_run_bm25(os.path.join(file_path, 'bm25/bm25_top1k_head.test.txt'))
    run_scibert = read_run_neural(os.path.join(file_path, 'scibert_dot/top1k_head_dctr-output.txt'))
    run_bertcat = read_run_neural(os.path.join(file_path,
                                               '3bert_fix_ensemble_avg/top200_rerank_head_dctr-ensemble-output.txt'))

    cut_off = 10

    assert set(run_scibert.keys()) == set(run_bm25.keys())
    assert set(run_scibert.keys()) == set(run_bertcat.keys())

    # create the pool by removing the tuples which are in the intersection, remove here the pair with the lower rank
    run_scibert_intersected, run_bertcat_intersected = run_intersection(run_scibert, run_bertcat, cut_off)
    run_scibert_intersected, run_bm25_intersected = run_intersection(run_scibert_intersected, run_bm25, cut_off)
    run_bm25_intersected, run_bertcat_intersected = run_intersection(run_bertcat_intersected, run_bm25_intersected,
                                                                     cut_off)
    write_run([run_bm25_intersected, run_scibert_intersected, run_bertcat_intersected], file_path,
              'judgements_pairs_trip.tsv', cut_off, '')

    # filter collection file so that only document in the query-document pairs are included and truncate length of
    # documents
    file_path = 'out_file/judgement_pairs_trip.tsv'
    collection_path = 'path_to/collection.tsv'
    outfile_path = 'out_file/documents_trip_short.tsv'

    col_trip = read_in_queries(collection_path)
    run_trip = read_in_runs(file_path)
    # filter then with the runs, only the documents in the collection which are also in the judgement pairs!
    col_trip_filtered = filter_collection_from_run(run_trip, col_trip)
    # truncate documents text to 512 subword tokens
    coll_short = truncate_collection(col_trip_filtered)
    write_collection(coll_short, outfile_path)


