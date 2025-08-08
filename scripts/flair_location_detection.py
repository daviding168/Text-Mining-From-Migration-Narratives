from flair.data import Sentence
from flair.models import SequenceTagger
from flair.splitter import SegtokSentenceSplitter
import re
import json
import numpy as np


def read_recits_from_text_file(file_path):
    recit_list = []
    # Open the file and read its contents
    with open(file_path, 'r') as file:
        # initialize an empty dictionary to store each entry
        entry_dict = {}
        # read each line in the file
        for line in file:
            # strip whitespace from the beginning and end of the line
            line = line.strip()
            # iff the line is not empty
            if line:
                # if the line starts with '****', it indicates the beginning of a new entry
                if line.startswith('****'):
                    # if there is already data in entry_dict, append it to data_list
                    if entry_dict:
                        recit_list.append(entry_dict)
                        entry_dict = {}  # Reset the entry dictionary for the new entry
                    # set the current line as the key for the entry_dict
                    key = line
                else:
                    # if the line does not start with '****', it is the value of the current entry
                    # add it to the entry_dict using the key
                    entry_dict[key] = line
        # append the last entry after the loop ends
        if entry_dict:
            recit_list.append(entry_dict)

    return recit_list


def find_start_offset_given_word_sentence(sentence, word):
    start_offsets = []
    start = 0
    while True:
        index = sentence.find(word, start)
        if index == -1:
            break
        start_offsets.append(index)
        start = index + 1

    return start_offsets


def ner_location_using_flair_for_english_and_french(text, lang, threshold):
    model_name = "../flairNER/ner-" + lang + "/pytorch_model.bin"
    tagger = SequenceTagger.load(model_name)

    # initialize sentence splitter
    splitter = SegtokSentenceSplitter()
    # use splitter to split text into list of sentences
    sentences = splitter.split(text)
    # predict NER tags
    tagger.predict(sentences)
    list_sentences = [re.sub(r'Sentence\[\d+\]:\s', "", str(sentence).split("→")[0].strip()).replace('"', '') for
                      sentence in sentences]

    list_dict_ent = []
    # iterate through sentences and print predicted labels
    for idx, output_flair in enumerate(sentences):
        # print(output_flair)
        curr_sentence = re.sub(r'Sentence\[\d+\]:\s', "", str(output_flair).split("→")[0].strip()).replace('"', '')
        # print(idx, curr_sentence)
        list_start_offsets_curr = []
        list_ent_infos = []
        for ent_idx, entity in enumerate(output_flair.get_spans('ner')):
            # entity
            ent = re.sub(r'Span\[(\d+):(\d+)\]:', "", str(entity).split("→")[0]).strip().replace('"', '')
            # score of entity
            score_ent = float(str(entity).split("→")[1].strip().split(" ")[1].replace("(", "").replace(")", ""))
            # type of entity
            type_ent = str(entity).split("→")[1].strip().split(" ")[0]
            list_ent_infos.append((idx, ent, score_ent, type_ent))
            ent_start_offsets_curr_sent = find_start_offset_given_word_sentence(curr_sentence, ent)
            list_start_offsets_curr += ent_start_offsets_curr_sent
        # list of non-redundant offsets
        list_start_offsets_curr = list(set(list_start_offsets_curr))
        # sort the offsets
        list_start_offsets_curr.sort()

        # make a new mapping dictionary
        dict_ent_infos = dict(zip(list_start_offsets_curr, list_ent_infos))

        for key, value in dict_ent_infos.items():
            dict_ent = {"entity_group": value[3], "ner_score": value[2], "word": value[1],
                        "start_offset": key,
                        "sentence_index": value[0]}
            list_dict_ent.append(dict_ent)

    # store final results of flair
    list_output_flair = []
    for dict_mention in list_dict_ent:
        sent_idx = dict_mention["sentence_index"]
        length_prev_sent = sum(len(list_sentences[i]) for i in range(sent_idx)) + sum(1 for i in range(sent_idx))
        start = dict_mention["start_offset"] + length_prev_sent
        dict_mention["start"] = start
        dict_mention["end"] = start + len(dict_mention["word"])
        dict_mention.pop("start_offset")
        dict_mention.pop("sentence_index")
        # filter only location entities
        if dict_mention["entity_group"] == "LOC" and dict_mention["ner_score"] >= threshold:
            list_output_flair.append(dict_mention)

    return list_output_flair


def precision_recall_f_score_flair(annotated_list, flair_list):
    excluded_keys_annotated_list = {"ner_score", "wikidataId"}
    excluded_keys_flair_list = {"ner_score"}
    # filter out excluded keys and values from dictionaries
    filtered_annotated_list = [{k: v for k, v in d.items() if k not in excluded_keys_annotated_list} for d in
                               annotated_list]
    filtered_flair_list = [{k: v for k, v in d.items() if k not in excluded_keys_flair_list} for d in flair_list]

    set_annotated = {frozenset(d.items()) for d in filtered_annotated_list}
    set_flair = {frozenset(d.items()) for d in filtered_flair_list}

    intersec_set = set_annotated.intersection(set_flair)

    # convert the intersection back to dictionaries
    intersec_dict = [dict(items) for items in intersec_set]

    # Find the remaining elements for each list
    remaining_annotated_list = [dict(items) for items in (set_annotated - intersec_set)]
    remaining_flair_list = [dict(items) for items in (set_flair - intersec_set)]

    # evaluation methods
    true_positive = len(intersec_dict)
    false_positive = len(remaining_flair_list)
    false_negative = len(remaining_annotated_list)
    precision_score = true_positive / (true_positive + false_positive)
    recall_score = true_positive / (true_positive + false_negative)

    f_score = (2 * precision_score * recall_score) / (precision_score + recall_score)

    return precision_score, recall_score, f_score, remaining_annotated_list


def get_output_dict_flair_given_list_dict_infos_recit_text(list_dict_infos_recit, lang, threshold):
    dict_infos_list_output_flair = {}
    for dict_infos_recit in list_dict_infos_recit:
        for recit_info, recit_text in dict_infos_recit.items():
            dict_infos_list_output_flair[recit_info] = ner_location_using_flair_for_english_and_french(recit_text, lang,
                                                                                                       threshold)

    return dict_infos_list_output_flair


def get_annotated_dict_infos_recit_text_from_json_file(file_path):
    with open(file_path, 'r') as file:
        dict_annotated_infos_mention_entity = json.load(file)
    return dict_annotated_infos_mention_entity


def calculate_precision_recall_f1_score(dict_annotated_recit, dict_flair_recit):
    dict_infos_precision = {}
    dict_infos_recall = {}
    dict_infos_f1_score = {}
    list_precision = []
    list_recall = []
    list_f1_score = []
    list_non_intersection = []
    for annotated_recit_infos, list_annotated_mentions in dict_annotated_recit.items():
        for flair_recit_infos, list_flair_mentions in dict_flair_recit.items():
            if annotated_recit_infos == flair_recit_infos:
                prec, rec, f1_score, non_intersection_list = precision_recall_f_score_flair(
                    annotated_list=list_annotated_mentions,
                    flair_list=list_flair_mentions)
                dict_infos_precision[annotated_recit_infos] = prec
                dict_infos_recall[annotated_recit_infos] = rec
                dict_infos_f1_score[annotated_recit_infos] = f1_score
                list_precision.append(prec)
                list_recall.append(rec)
                list_f1_score.append(f1_score)
                list_non_intersection += non_intersection_list

    return np.mean(list_precision), np.mean(list_recall), np.mean(list_f1_score), list_non_intersection


'''
# example usage for english
text_file = "../dataset/EN/recits_EN/english_recits.txt"
annotated_json_file_path = "../annotated_mention_entity/annotated_english_recits.json"
list_infos_narrative_text = read_recits_from_text_file(file_path=text_file)
#output_dict_flair = get_output_dict_flair_given_list_dict_infos_recit_text(
#    list_dict_infos_recit=list_infos_narrative_text, lang="english", threshold=0.6)
#print(output_dict_flair)
output_annotated_dict = get_annotated_dict_infos_recit_text_from_json_file(file_path=annotated_json_file_path)
avg_prec_en, avg_rec_en, avg_f1_score_en = calculate_precision_recall_f1_score(
    dict_annotated_recit=output_annotated_dict, dict_flair_recit=output_dict_flair)

print("English Recits")
print("Average precision: ", avg_prec_en)
print("Average recall: ", avg_rec_en)
print("Average f1-score: ", avg_f1_score_en)


# example usage for french
text_file_fr = "../dataset/FR/recits_FR/french_recits.txt"
annotated_json_file_path_fr = "../annotated_mention_entity/annotated_french_recits.json"
list_infos_narrative_text_fr = read_recits_from_text_file(file_path=text_file_fr)
output_dict_flair_fr = get_output_dict_flair_given_list_dict_infos_recit_text(
    list_dict_infos_recit=list_infos_narrative_text_fr, lang="french", threshold=0.55)
# print(output_dict_flair_fr)
output_dict_flair_fr = get_annotated_dict_infos_recit_text_from_json_file(
    "../annotated_mention_entity/output_flair_french_recits.json")
#print(output_dict_flair_fr)
output_annotated_dict_fr = get_annotated_dict_infos_recit_text_from_json_file(file_path=annotated_json_file_path_fr)
avg_prec_fr, avg_rec_fr, avg_f1_score_fr = calculate_precision_recall_f1_score(
    dict_annotated_recit=output_annotated_dict_fr, dict_flair_recit=new_output_dict_flair_fr)

print("French Recits")
print("Average precision: ", avg_prec_fr)
print("Average recall: ", avg_rec_fr)
print("Average f1-score: ", avg_f1_score_fr)


output_dict_flair_en = get_annotated_dict_infos_recit_text_from_json_file(
    file_path="../annotated_mention_entity/pre-output_flair_english_recits.json")
output_annotated_dict_en = get_annotated_dict_infos_recit_text_from_json_file(
    file_path="../annotated_mention_entity/annotated_english_recits.json")

avg_prec_en, avg_rec_en, avg_f1_score_en, _ = calculate_precision_recall_f1_score(
    dict_annotated_recit=output_annotated_dict_en, dict_flair_recit=output_dict_flair_en)

print("English Recits")
print("Average precision: ", avg_prec_en)
print("Average recall: ", avg_rec_en)
print("Average f1-score: ", avg_f1_score_en)

output_dict_flair_fr = get_annotated_dict_infos_recit_text_from_json_file(
    file_path="../annotated_mention_entity/pre-output_flair_french_recits.json")
output_annotated_dict_fr = get_annotated_dict_infos_recit_text_from_json_file(
    file_path="../annotated_mention_entity/annotated_french_recits.json")

avg_prec_fr, avg_rec_fr, avg_f1_score_fr, _ = calculate_precision_recall_f1_score(
    dict_annotated_recit=output_annotated_dict_fr, dict_flair_recit=output_dict_flair_fr)

print("French Recits")
print("Average precision: ", avg_prec_fr)
print("Average recall: ", avg_rec_fr)
print("Average f1-score: ", avg_f1_score_fr)
'''
