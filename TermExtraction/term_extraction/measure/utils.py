# coding = utf-8

from ..pattern import Pattern
import re
import json


def count_words(term):
    return len(term.split(" "))


def computeStatistics(pattern_instance: Pattern, text=None, corpus=None, use_with_method="tf_idf"):
    """
    Return two dict, one that store overall corpus statistics of terms identified, the second store statistics
    of each term per document.

    Parameters
    ----------
    pattern_instance : Pattern database
    text : str or None
    corpus : str or None
    use_with_method: specified method to use with this function
    Returns
    -------
    tuple(dict, dict)
        overalldocuments statistics dict, per document statistics dict

    """
    stats_general = {}
    stats_per_doc = {}
    all_tag_text_one_word = {}
    all_tag_text_two_words = {}
    longest_pat = pattern_instance.get_longest_pattern() + 1  # +1 to offset the double range()
    language = pattern_instance.language

    # language choice to select the stopwords
    if language == "en":
        with open("TermExtraction/term_extraction/resources/stopWords/Stop-words-english.txt", 'r') as f:
            stop_words = f.read().split('\n')
        list_stop_words = list(stop_words)
    elif language == "fr":
        with open("TermExtraction/term_extraction/resources/stopWords/Stop-words-french.txt", 'r') as f:
            stop_words = f.read().split('\n')
        list_stop_words = list(stop_words)

    if not text is None:
        corpus = [text]

    if use_with_method == "tf_idf":
        # loop through each document in the corpus(set of documents), doc is a dataframe
        for ix, doc in enumerate(corpus):
            list_tag_text_two_words = []
            stats_per_doc[ix] = {}
            # list of tokens for each document
            words = doc["word"].values
            tag_text_one_word = words.copy()
            tag_text_two_words = words.copy()
            # Part Of Speech corresponds to text
            partofspeech_vals = doc["pos"].values
            lemma_vals = doc["lemma"].values
            # loop through part of speech for each document
            for iy, pos in enumerate(partofspeech_vals):
                # loop to get pos sequence with the longest pattern
                for i in range(longest_pat):
                    # part of speech sequence
                    pos_seq = [partofspeech_vals[iy + dec] for dec in range(i) if iy + dec < len(partofspeech_vals)]
                    # search if the list pos_seq match with defined patterns
                    flag, pattern, freq = pattern_instance.match(pos_seq)

                    # when matching sequence found
                    if flag:
                        # matching term
                        t = " ".join([words[iy + dec] for dec in range(i) if iy + dec < len(partofspeech_vals)])
                        term = t.lower()
                        term = term.replace(" - ", "-")
                        word_list = term.split(" ")
                        length_word = len(word_list)

                        # test stopwords specifically for french
                        # NOUN
                        if length_word == 1:
                            if term.startswith("-") or term.endswith("-") or term.endswith("”") or \
                                    term.lower().replace("_", "-") in list_stop_words:
                                continue
                            else:
                                tag_text_one_word[iy] = "<phrase>" + t + "</phrase>"
                                # tag_text_one_word[iy] = "<phrase>" + term + "</phrase>"

                        # NOUN NOUN
                        elif length_word == 2:
                            if term.startswith("-") or term.endswith("-") or term.endswith("”") or \
                                    term.lower().replace("_", "-") in list_stop_words:
                                continue
                            else:
                                # term = word_list[0] + "_" + word_list[1]
                                term_string = "<phrase>" + t + "</phrase>"
                                # term_string = "<phrase>" + term + "</phrase>"
                                tag_text_two_words[iy:iy + length_word] = [term_string]
                        if term not in stats_general:
                            stats_general[term] = {
                                "freq": 1,
                                "num_doc": 1,
                                "list_index_doc": [ix],
                                "pattern_used": pattern,
                                "last_saw": ix
                            }
                        elif term in stats_general:
                            # term in the same document, we add the frequency for that term
                            stats_general[term]["freq"] += 1
                            # change index of last saw for recently term in nth doc (same term appear in diff document)
                            if ix > stats_general[term]["last_saw"]:
                                stats_general[term]["last_saw"] = ix
                                stats_general[term]["num_doc"] += 1
                                stats_general[term]["list_index_doc"].append(ix)
                        # term is not in statistics then add that term into dict
                        if term not in stats_per_doc[ix]:
                            stats_per_doc[ix][term] = {
                                "freq": 0
                            }
                        # add frequency for that term
                        stats_per_doc[ix][term]["freq"] += 1
            list_tokens_one_word = list(tag_text_one_word)
            all_tag_text_one_word[ix] = " ".join(token for token in list_tokens_one_word)
            list_two_words_term = list(tag_text_two_words)
            # loop to remove redundancy tag (<phrase></phrase>) from list of token for two words term
            for item in list_two_words_term:
                if item.startswith('<phrase>') and len(list_tag_text_two_words) > 0:
                    if item == list_tag_text_two_words[-1]:
                        continue
                list_tag_text_two_words.append(item)
            all_tag_text_two_words[ix] = " ".join(token for token in list_tag_text_two_words)
        if language == "en":
            dir_loc_one_term = "dataset/EN/beginner/tf_idf_" + language + "_one_term_tag.json"
            with open(dir_loc_one_term, 'w') as convert_file:
                convert_file.write(json.dumps(all_tag_text_one_word, ensure_ascii=False))

            dir_loc_two_terms = "dataset/EN/beginner/tf_idf_" + language + "_two_terms_tag.json"
            with open(dir_loc_two_terms, 'w') as convert_file:
                convert_file.write(json.dumps(all_tag_text_two_words, ensure_ascii=False))
        elif language == "fr":
            dir_loc_one_term = "dataset/FR/beginner/tf_idf_" + language + "_one_term_tag.json"
            with open(dir_loc_one_term, 'w') as convert_file:
                convert_file.write(json.dumps(all_tag_text_one_word, ensure_ascii=False))

            dir_loc_two_terms = "dataset/FR/beginner/tf_idf_" + language + "_two_terms_tag.json"
            with open(dir_loc_two_terms, 'w') as convert_file:
                convert_file.write(json.dumps(all_tag_text_two_words, ensure_ascii=False))
    elif use_with_method == "c_value":
        # loop through each document in the corpus(set of documents)
        for ix, doc in enumerate(corpus):  # doc is a dataframe
            stats_per_doc[ix] = {}
            # list of tokens for each document
            words = doc["word"].values
            # Part Of Speech corresponds to text
            partofspeech_vals = doc["pos"].values
            # loop through part of speech for each document
            for iy, pos in enumerate(partofspeech_vals):
                # loop to get pos sequence with the longest pattern
                for i in range(longest_pat):
                    # part of speech sequence
                    pos_seq = [partofspeech_vals[iy + dec] for dec in range(i) if iy + dec < len(partofspeech_vals)]
                    # search if the list pos_seq match with defined patterns
                    flag, pattern, freq = pattern_instance.match(pos_seq)
                    # when matching sequence found
                    if flag:
                        # matching term
                        t = " ".join([words[iy + dec] for dec in range(i) if iy + dec < len(partofspeech_vals)])
                        term = t.lower()
                        term = term.replace(" - ", "-")
                        term = term.replace(" - ", "-")

                        if term not in stats_general:
                            stats_general[term] = {
                                "freq": 1,
                                "num_doc": 1,
                                "list_index_doc": [ix],
                                "pattern_used": pattern,
                                "num_words": count_words(term),
                                "last_saw": ix
                            }
                        elif term in stats_general:
                            # term in the same document, we add the frequency for that term
                            stats_general[term]["freq"] += 1
                            # change index of last saw for recently term in nth doc (same term appear in diff document)
                            if ix > stats_general[term]["last_saw"]:
                                stats_general[term]["last_saw"] = ix
                                stats_general[term]["num_doc"] += 1
                                stats_general[term]["list_index_doc"].append(ix)
                        # term is not in statistics then add that term into dict
                        if term not in stats_per_doc[ix]:
                            stats_per_doc[ix][term] = {
                                "freq": 0
                            }
                        # add frequency for that term
                        stats_per_doc[ix][term]["freq"] += 1

    return stats_general, stats_per_doc


def term_in_term(term1, term2):
    """
    Return true if a term appears in a second one.

    Parameters
    ----------
    term1 : str
    term2 : str

    Returns
    -------
    bool
        if term1 is in term2
    """
    found = re.findall("({0} | {0} | {0})".format(term1), term2)
    return len(found) > 0


def contained_in_other_keywords(term, general_stats_dict):
    """
    Return the number of time a term appears in the other identified term and the sum of their frequency.

    Parameters
    ----------
    term : str
    general_stats_dict : dict

    Returns
    -------
    tuple(int, int)
        count of other terms in which the selected term appears, sum of these termes "new frequency"
    """
    count_in = 0
    sum_new_freq = 0
    # list of list contain term/word, number of words, and new_freq of each term
    data_keywords = [[k, v["num_words"], v["freq"]] for k, v in general_stats_dict.items()]
    # loop through each list above
    for data in data_keywords:
        # check if a term contains in another terms in general_stats of a unique document
        if term_in_term(term, data[0]):
            # total number of the candidate terms(i.e. longer candidate terms that a term appeared in)
            count_in += 1
            # sigma b∈T(a) of f(b), T(a) is the set of candidate terms, f(.) is its frequency of occurrence in doc
            sum_new_freq += data[-1]

    return count_in, sum_new_freq
