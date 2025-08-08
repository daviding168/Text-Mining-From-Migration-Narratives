# coding = utf-8
import copy
import numpy as np
import pandas as pd
from .utils import computeStatistics, contained_in_other_keywords, count_words
from numpy import array
from numpy.linalg import norm



def c_value(text, patterns):
    general_stats, _ = computeStatistics(patterns, text=text, use_with_method="c_value")
    # loop through each word in dict general_stats
    for word in general_stats:
        count_in, sum_new_freq = contained_in_other_keywords(word, general_stats)
        # sum of all frequencies of longer terms that a word appeared in
        general_stats[word]["sum_new_freq"] = sum_new_freq
        # total number of longer terms that a word appeared in
        general_stats[word]["longertermwithword"] = count_in
        # w(A), modified to avoid null value
        rank = np.log2(count_words(word) + 1)
        # word frequency
        word_freq = general_stats[word]["freq"]
        # nested term
        if count_in > 0:
            if word_freq - (sum_new_freq / count_in) <= 0:
                # negative values
                rank = 0
            else:
                rank = rank * (word_freq - (sum_new_freq / count_in))
        # not nested term
        else:
            rank = rank * word_freq
        general_stats[word]["rank"] = rank

    return general_stats


def tf_idf(corpus, patterns, opt="AVG"):
    general_stats, stats_per_doc = computeStatistics(patterns, corpus=corpus, use_with_method="tf_idf")
    # set of documents (corpus)
    len_doc = len(corpus)
    # loop through statistics for each document
    for ix, doc in stats_per_doc.items():
        total_words = len(doc)
        max_rank, min_rank = 0., 9999.0
        try:  # find max freq among all terms in a doc
            max_freq = max([doc[k]["freq"] for k in doc])
        except:
            max_freq = 0
        list_sum_square_each_doc = []
        # loop through dict of each term
        for term, val in doc.items():
            # modified the normal term frequency
            norm_tf = val["freq"] / max_freq
            # original formula
            # norm_tf = val["freq"] / total_words
            idf = np.log10(len_doc / (general_stats[term]["num_doc"]))
            rank = norm_tf * idf
            list_sum_square_each_doc.append(rank)
            # rank per document
            stats_per_doc[ix][term]["rank"] = rank
        # convert to array
        array_sum_square_each_doc = array(list_sum_square_each_doc)
        # l2 norm
        l2_norm = norm(array_sum_square_each_doc)
        # normalization each term in each document
        for term, val in doc.items():
            stats_per_doc[ix][term]["rank_norm"] = stats_per_doc[ix][term]["rank"] / l2_norm

    return apply_opt(general_stats, stats_per_doc, opt)


def f_tfidf_c(corpus, patterns, opt="AVG"):
    stats_tfidf = tf_idf(corpus, patterns, opt)
    # C_value for a set of documents we need to merge to a single text
    stats_c_value = c_value(pd.concat(corpus), patterns)

    stats_final = copy.deepcopy(stats_tfidf)
    for term in stats_tfidf:
        rank_tfidf = stats_tfidf[term]["rank"]
        rank_c_value = stats_c_value[term]["rank"]
        stats_final[term]["rank"] = (2 * (rank_tfidf * rank_c_value)) / (rank_tfidf + rank_c_value)

    return stats_final


def apply_opt(general_stats, stats_per_doc, opt):
    if not opt in ["MAX", "SUM", "AVG"]:
        raise ValueError("opt must be equal to MAX or SUM or AVG")
    # loop through statistics of each document
    for ix, doc in stats_per_doc.items():
        # loop through each term in each document
        for term in doc:
            # add key "rank" into dictionary general_stats
            if "rank" not in general_stats[term]:
                # take the maximum rank between corpus statistics and per document statistics
                general_stats[term]["rank"] = 0
            if opt == "MAX":
                general_stats[term]["rank"] = max(general_stats[term]["rank"], doc[term]["rank_norm"])
                # general_stats[term]["rank"] = max(general_stats[term]["rank"], doc[term]["rank"])
            if opt in ["SUM", "AVG"]:
                # loop through all documents and sum rank_norm of each document
                general_stats[term]["rank"] += doc[term]["rank_norm"]
                # general_stats[term]["rank"] += doc[term]["rank"]
    if opt == "AVG":
        general_stats[term]["rank"] = general_stats[term]["rank"] / general_stats[term]["num_doc"]

    return general_stats
