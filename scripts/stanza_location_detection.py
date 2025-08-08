import stanza
import numpy as np
import json

# put your proxies
proxies = {'http': '', 'https': ''}


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


def get_annotated_dict_infos_recit_text_from_json_file(file_path):
    with open(file_path, 'r') as file:
        dict_annotated_infos_mention_entity = json.load(file)
    return dict_annotated_infos_mention_entity


def get_output_dict_stanza_for_english_and_french(path_file, lang):
    if lang == "en":
        stanza.download(lang='en', proxies=proxies)  # This downloads the English models for the neural pipeline
        nlp = stanza.Pipeline(lang='en', processors="tokenize, ner",
                              download_method=None)  # This sets up a default neural pipeline in English
    elif lang == "fr":
        stanza.download(lang='fr', proxies=proxies)  # This downloads the French models for the neural pipeline
        nlp = stanza.Pipeline(lang='fr', processors="tokenize, ner",
                              download_method=None)  # This sets up a default neural pipeline in French

    list_infos_narrative_text = read_recits_from_text_file(file_path=path_file)

    dict_infos_list_output_stanza = {}
    for recite_dict in list_infos_narrative_text:
        for recite_info, recite_text in recite_dict.items():
            doc = nlp(recite_text)
            list_dict_ent_each_text = []
            for ent in doc.ents:
                if lang == "en":
                    if ent.type == "GPE":
                        dict_ent = {"entity_group": "LOC",  #
                                    "ner_score": 1.0,
                                    "word": ent.text,
                                    "start": ent.start_char,
                                    "end": ent.end_char}
                        list_dict_ent_each_text.append(dict_ent)
                elif lang == "fr":
                    if ent.type == "LOC":
                        dict_ent = {"entity_group": ent.type,
                                    "ner_score": 1.0,
                                    "word": ent.text,
                                    "start": ent.start_char,
                                    "end": ent.end_char}
                        list_dict_ent_each_text.append(dict_ent)
            dict_infos_list_output_stanza[recite_info] = list_dict_ent_each_text

    return dict_infos_list_output_stanza


def precision_recall_f_score_stanza(annotated_list, stanza_list):
    excluded_keys_annotated_list = {"ner_score", "wikidataId"}
    excluded_keys_stanza_list = {"ner_score"}
    # filter out excluded keys and values from dictionaries
    filtered_annotated_list = [{k: v for k, v in d.items() if k not in excluded_keys_annotated_list} for d in
                               annotated_list]
    filtered_stanza_list = [{k: v for k, v in d.items() if k not in excluded_keys_stanza_list} for d in stanza_list]

    set_annotated = {frozenset(d.items()) for d in filtered_annotated_list}
    set_stanza = {frozenset(d.items()) for d in filtered_stanza_list}

    intersec_set = set_annotated.intersection(set_stanza)

    # convert the intersection back to dictionaries
    intersec_dict = [dict(items) for items in intersec_set]

    # Find the remaining elements for each list
    remaining_annotated_list = [dict(items) for items in (set_annotated - intersec_set)]
    remaining_stanza_list = [dict(items) for items in (set_stanza - intersec_set)]

    # evaluation methods
    true_positive = len(intersec_dict)
    false_positive = len(remaining_stanza_list)
    false_negative = len(remaining_annotated_list)
    precision_score = true_positive / (true_positive + false_positive)
    recall_score = true_positive / (true_positive + false_negative)
    f_score = (2 * precision_score * recall_score) / (precision_score + recall_score)

    return precision_score, recall_score, f_score


def calculate_avg_precision_recall_f1_score_for_stanza(dict_annotated_recit, dict_stanza_recit):
    dict_infos_precision = {}
    dict_infos_recall = {}
    dict_infos_f1_score = {}
    list_precision = []
    list_recall = []
    list_f1_score = []
    for annotated_recit_infos, list_annotated_mentions in dict_annotated_recit.items():
        for stanza_recit_infos, list_stanza_mentions in dict_stanza_recit.items():
            if annotated_recit_infos == stanza_recit_infos:
                prec, rec, f1_score = precision_recall_f_score_stanza(annotated_list=list_annotated_mentions,
                                                                      stanza_list=list_stanza_mentions)
                dict_infos_precision[annotated_recit_infos] = prec
                dict_infos_recall[annotated_recit_infos] = rec
                dict_infos_f1_score[annotated_recit_infos] = f1_score
                list_precision.append(prec)
                list_recall.append(rec)
                list_f1_score.append(f1_score)

    return np.mean(list_precision), np.mean(list_recall), np.mean(list_f1_score)


"""
# example usage for English
text_file_en = "../dataset/EN/recits_EN/english_recits.txt"
annotated_json_file_path_en = "../annotated_mention_entity/annotated_english_recits.json"
output_annotated_dict_en = get_annotated_dict_infos_recit_text_from_json_file(file_path=annotated_json_file_path_en)
output_dict_stanza_en = get_output_dict_stanza_for_english_and_french(path_file=text_file_en, lang="en")
avg_prec_en, avg_rec_en, avg_f1_score_en = calculate_avg_precision_recall_f1_score_for_stanza(
    dict_annotated_recit=output_annotated_dict_en, dict_stanza_recit=output_dict_stanza_en)
with open("../annotated_mention_entity/pre-output_stanza_english_recits.json", "w") as fout:
    json.dump(output_dict_stanza_en, fout, indent=4)
print("English Recits")
print("Average precision: ", avg_prec_en)
print("Average recall: ", avg_rec_en)
print("Average f1-score: ", avg_f1_score_en)

# example usage for French
text_file_fr = "../dataset/FR/recits_FR/french_recits.txt"
annotated_json_file_path_fr = "../annotated_mention_entity/annotated_french_recits.json"
output_annotated_dict_fr = get_annotated_dict_infos_recit_text_from_json_file(file_path=annotated_json_file_path_fr)
output_dict_stanza_fr = get_output_dict_stanza_for_english_and_french(path_file=text_file_fr, lang="fr")
avg_prec_fr, avg_rec_fr, avg_f1_score_fr = calculate_avg_precision_recall_f1_score_for_stanza(
    dict_annotated_recit=output_annotated_dict_fr, dict_stanza_recit=output_dict_stanza_fr)
with open("../annotated_mention_entity/pre-output_stanza_french_recits.json", "w") as fout:
    json.dump(output_dict_stanza_fr, fout, indent=4)
print("French Recits")
print("Average precision: ", avg_prec_fr)
print("Average recall: ", avg_rec_fr)
print("Average f1-score: ", avg_f1_score_fr)


output_dict_stanza_en = get_annotated_dict_infos_recit_text_from_json_file(
    file_path="../annotated_mention_entity/pre-output_stanza_english_recits.json")
output_annotated_dict_en = get_annotated_dict_infos_recit_text_from_json_file(
    file_path="../annotated_mention_entity/annotated_english_recits.json")

avg_prec_en, avg_rec_en, avg_f1_score_en = calculate_avg_precision_recall_f1_score_for_stanza(
    dict_annotated_recit=output_annotated_dict_en, dict_stanza_recit=output_dict_stanza_en)

print("English Recits")
print("Average precision: ", avg_prec_en)
print("Average recall: ", avg_rec_en)
print("Average f1-score: ", avg_f1_score_en)

output_dict_stanza_fr = get_annotated_dict_infos_recit_text_from_json_file(
    file_path="../annotated_mention_entity/pre-output_stanza_french_recits.json")
output_annotated_dict_fr = get_annotated_dict_infos_recit_text_from_json_file(
    file_path="../annotated_mention_entity/annotated_french_recits.json")

avg_prec_fr, avg_rec_fr, avg_f1_score_fr = calculate_avg_precision_recall_f1_score_for_stanza(
    dict_annotated_recit=output_annotated_dict_fr, dict_stanza_recit=output_dict_stanza_fr)

print("French Recits")
print("Average precision: ", avg_prec_fr)
print("Average recall: ", avg_rec_fr)
print("Average f1-score: ", avg_f1_score_fr)
"""
