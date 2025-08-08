import spacy
import numpy as np
import json

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


def get_output_dict_spacy_for_english_and_french(path_file, lang):
    if lang == "en":
        nlp = spacy.load("en_core_web_trf")  # en_core_web_sm en_core_web_md en_core_web_lg en_core_web_trf
    elif lang == "fr":
        nlp = spacy.load("fr_core_news_md")  # fr_core_news_sm fr_core_news_md fr_core_news_lg
    list_infos_narrative_text = read_recits_from_text_file(file_path=path_file)

    dict_infos_list_output_spacy = {}
    for recite_dict in list_infos_narrative_text:
        for recite_info, recite_text in recite_dict.items():
            doc = nlp(recite_text)
            list_dict_ent_each_text = []
            for ent in doc.ents:
                if lang == "en":
                    if ent.label_ == "GPE":
                        dict_ent = {"entity_group": "LOC",  #
                                    "ner_score": 1.0,
                                    "word": ent.text,
                                    "start": ent.start_char,
                                    "end": ent.end_char}
                        list_dict_ent_each_text.append(dict_ent)
                elif lang == "fr":
                    if ent.label_ == "LOC":
                        dict_ent = {"entity_group": ent.label_,
                                    "ner_score": 1.0,
                                    "word": ent.text,
                                    "start": ent.start_char,
                                    "end": ent.end_char}
                        list_dict_ent_each_text.append(dict_ent)
            dict_infos_list_output_spacy[recite_info] = list_dict_ent_each_text

    return dict_infos_list_output_spacy


def precision_recall_f_score_spacy(annotated_list, spacy_list):
    excluded_keys_annotated_list = {"ner_score", "wikidataId"}
    excluded_keys_spacy_list = {"ner_score"}
    # filter out excluded keys and values from dictionaries
    filtered_annotated_list = [{k: v for k, v in d.items() if k not in excluded_keys_annotated_list} for d in
                               annotated_list]
    filtered_spacy_list = [{k: v for k, v in d.items() if k not in excluded_keys_spacy_list} for d in spacy_list]

    set_annotated = {frozenset(d.items()) for d in filtered_annotated_list}
    set_spacy = {frozenset(d.items()) for d in filtered_spacy_list}

    intersec_set = set_annotated.intersection(set_spacy)

    # convert the intersection back to dictionaries
    intersec_dict = [dict(items) for items in intersec_set]

    # Find the remaining elements for each list
    remaining_annotated_list = [dict(items) for items in (set_annotated - intersec_set)]
    remaining_spacy_list = [dict(items) for items in (set_spacy - intersec_set)]

    # evaluation methods
    true_positive = len(intersec_dict)
    false_positive = len(remaining_spacy_list)
    false_negative = len(remaining_annotated_list)
    precision_score = true_positive / (true_positive + false_positive)
    recall_score = true_positive / (true_positive + false_negative)

    f_score = (2 * precision_score * recall_score) / (precision_score + recall_score)

    return precision_score, recall_score, f_score


def calculate_avg_precision_recall_f1_score_for_spacy(dict_annotated_recit, dict_spacy_recit):
    dict_infos_precision = {}
    dict_infos_recall = {}
    dict_infos_f1_score = {}
    list_precision = []
    list_recall = []
    list_f1_score = []
    for annotated_recit_infos, list_annotated_mentions in dict_annotated_recit.items():
        for spacy_recit_infos, list_spacy_mentions in dict_spacy_recit.items():
            if annotated_recit_infos == spacy_recit_infos:
                prec, rec, f1_score = precision_recall_f_score_spacy(annotated_list=list_annotated_mentions,
                                                                     spacy_list=list_spacy_mentions)
                dict_infos_precision[annotated_recit_infos] = prec
                dict_infos_recall[annotated_recit_infos] = rec
                dict_infos_f1_score[annotated_recit_infos] = f1_score
                list_precision.append(prec)
                list_recall.append(rec)
                list_f1_score.append(f1_score)

    return np.mean(list_precision), np.mean(list_recall), np.mean(list_f1_score)

"""
# example usage for english
text_file = "../dataset/EN/recits_EN/english_recits.txt"
annotated_json_file_path = "../annotated_mention_entity/annotated_english_recits.json"
output_annotated_dict = get_annotated_dict_infos_recit_text_from_json_file(file_path=annotated_json_file_path)
output_dict_spacy = get_output_dict_spacy_for_english_and_french(path_file=text_file, lang="en")
avg_prec_en, avg_rec_en, avg_f1_score_en = calculate_avg_precision_recall_f1_score_for_spacy(
    dict_annotated_recit=output_annotated_dict, dict_spacy_recit=output_dict_spacy)
print(output_dict_spacy)
with open("../annotated_mention_entity/pre-output_spacy_english_recits.json", "w") as fout:
    json.dump(output_dict_spacy, fout, indent=4)
print("English Recits")
print("Average precision: ", avg_prec_en)
print("Average recall: ", avg_rec_en)
print("Average f1-score: ", avg_f1_score_en)

# example usage for french
text_file = "../dataset/FR/recits_FR/french_recits.txt"
annotated_json_file_path = "../annotated_mention_entity/annotated_french_recits.json"
output_annotated_dict = get_annotated_dict_infos_recit_text_from_json_file(file_path=annotated_json_file_path)
output_dict_spacy = get_output_dict_spacy_for_english_and_french(path_file=text_file, lang="fr")
avg_prec_fr, avg_rec_fr, avg_f1_score_fr = calculate_avg_precision_recall_f1_score_for_spacy(
    dict_annotated_recit=output_annotated_dict, dict_spacy_recit=output_dict_spacy)
with open("../annotated_mention_entity/pre-output_spacy_french_recits.json", "w") as fout:
    json.dump(output_dict_spacy, fout, indent=4)
print("French Recits")
print("Average precision: ", avg_prec_fr)
print("Average recall: ", avg_rec_fr)
print("Average f1-score: ", avg_f1_score_fr)

output_dict_spacy_en = get_annotated_dict_infos_recit_text_from_json_file(
    file_path="../annotated_mention_entity/pre-output_spacy_english_recits.json")
output_annotated_dict_en = get_annotated_dict_infos_recit_text_from_json_file(
    file_path="annotated_mention_entity/annotated_english_recits.json")

avg_prec_en, avg_rec_en, avg_f1_score_en = calculate_avg_precision_recall_f1_score_for_spacy(
    dict_annotated_recit=output_annotated_dict_en, dict_spacy_recit=output_dict_spacy_en)

print("English Recits")
print("Average precision: ", avg_prec_en)
print("Average recall: ", avg_rec_en)
print("Average f1-score: ", avg_f1_score_en)

output_dict_spacy_fr = get_annotated_dict_infos_recit_text_from_json_file(
    file_path="../annotated_mention_entity/pre-output_spacy_french_recits.json")
output_annotated_dict_fr = get_annotated_dict_infos_recit_text_from_json_file(
    file_path="../annotated_mention_entity/annotated_french_recits.json")

avg_prec_fr, avg_rec_fr, avg_f1_score_fr = calculate_avg_precision_recall_f1_score_for_spacy(
    dict_annotated_recit=output_annotated_dict_fr, dict_spacy_recit=output_dict_spacy_fr)

print("French Recits")
print("Average precision: ", avg_prec_fr)
print("Average recall: ", avg_rec_fr)
print("Average f1-score: ", avg_f1_score_fr)
"""

