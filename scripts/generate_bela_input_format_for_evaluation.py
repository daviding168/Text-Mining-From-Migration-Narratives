from flair.data import Sentence
from flair.models import SequenceTagger
from flair.splitter import SegtokSentenceSplitter
import re
import json
import numpy as np
from copy import deepcopy


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


def ner_location_using_flair_for_english_and_french(text, lang):
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
        if dict_mention["entity_group"] == "LOC":
            list_output_flair.append(dict_mention)

    return list_output_flair


def get_annotated_dict_infos_recit_text_from_json_file(file_path):
    with open(file_path, 'r') as file:
        dict_annotated_infos_mention_entity = json.load(file)
    return dict_annotated_infos_mention_entity


def get_list_dict_bela_format_given_list_sentences_list_dict_mentions(idx_recit, l_sentences, l_dict_mentions):
    dict_format_bela = {}
    list_dict_format_bela = []
    for idx_sent, sentence in enumerate(l_sentences):
        list_gt_entities = list()
        for dict_mention in l_dict_mentions:
            start_offset = dict_mention["start"]
            end_offset = dict_mention["end"]
            lower_bound = sum(len(l_sentences[idx]) for idx in range(idx_sent)) + sum(1 for _ in range(idx_sent))
            upper_bound = lower_bound + len(l_sentences[idx_sent])

            if lower_bound <= start_offset <= upper_bound and lower_bound <= end_offset <= upper_bound:
                ent = dict_mention["word"]
                start_offset_current_sent = start_offset - lower_bound
                mention_list = [0, 0, dict_mention["wikidataId"], "wiki", start_offset_current_sent, len(ent)]
                list_gt_entities.append(mention_list)

        dict_format_bela["document_id"] = "recit_" + str(idx_recit) + "_sentence_" + str(idx_sent)
        dict_format_bela["original_text"] = sentence
        dict_format_bela["gt_entities"] = list_gt_entities
        dict_format_bela_deep_copy = deepcopy(dict_format_bela)
        list_dict_format_bela.append(dict_format_bela_deep_copy)

    list_dict_format_bela_with_mention = [d for d in list_dict_format_bela if d["gt_entities"] != []]

    list_dict_format_bela_with_mention_dump = [json.dumps(d) for d in list_dict_format_bela_with_mention]

    return list_dict_format_bela_with_mention_dump
    # return list_dict_format_bela_with_mention


def write_annotated_entity_text_by_text_for_input_bela_format(annotated_dict, recit_list, file_name):
    list_dict_format_bela = []
    for recit in recit_list:
        dict_format_bela = {}
        for recit_title, corpus_text in recit.items():
            dict_format_bela["document_id"] = recit_title
            dict_format_bela["original_text"] = corpus_text
            list_gt_entities = []
            for mention_dict in annotated_dict[recit_title]:
                mention_list = [0, 0, mention_dict["wikidataId"], "wiki", mention_dict["start"],
                                len(mention_dict["word"])]
                list_gt_entities.append(mention_list)
            dict_format_bela["gt_entities"] = list_gt_entities
            # print(dict_format_bela)
            # list_dict_format_bela.append(json.dumps(dict_format_bela))
            list_dict_format_bela.append(dict_format_bela)

    with open(file_name, "w") as fout:
        for narrative in list_dict_format_bela:
            fout.write(str(narrative))
            fout.write("\n")


def write_annotated_entity_sentence_by_sentence_for_input_bela_format(annotated_dict, recit_list, file_name):
    list_dict_format_bela = []
    for recit_idx, recit in enumerate(recit_list):
        for recit_title, corpus_text in recit.items():
            # initialize sentence splitter
            splitter = SegtokSentenceSplitter()
            # use splitter to split text into list of sentences
            sentences = splitter.split(corpus_text)
            # predict NER tags
            # tagger.predict(sentences)
            list_all_sentences = [
                re.sub(r'Sentence\[\d+\]:\s', "", str(sentence).split("→")[0].strip()).replace('"', '')
                for sentence in sentences]
            # list of mention dictionaries
            list_all_mentions = annotated_dict[recit_title]
            list_dict_format_bela += get_list_dict_bela_format_given_list_sentences_list_dict_mentions(
                idx_recit=recit_idx, l_sentences=list_all_sentences, l_dict_mentions=list_all_mentions)

    with open(file_name, "w") as fout:
        for narrative in list_dict_format_bela:
            fout.write(str(narrative))
            fout.write("\n")

# annotated_dict = get_annotated_dict_infos_recit_text_from_json_file("../annotated_mention_entity/annotated_english_recits.json")
# recit_list = read_recits_from_text_file("../dataset/FR/recits_FR/french_recits.txt")
# print(annotated_dict)
# list_mention = []
# for k, v in annotated_dict.items():
#    for ment in v:
#        list_mention.append(ment)
# print(len(list_mention))
# write_annotated_entity_sentence_by_sentence_for_input_bela_format(annotated_dict=annotated_dict, recit_list=recit_list,
#                                                                  file_name="sent_french_test.jsonl")

# annotated_dict = get_annotated_dict_infos_recit_text_from_json_file("../annotated_mention_entity/annotated_french_recits.json")
# recit_list = read_recits_from_text_file("../dataset/FR/recits_FR/french_recits.txt")
# write_annotated_entity_text_by_text_for_input_bela_format(annotated_dict=annotated_dict, recit_list=recit_list,
#                                                          file_name="text_french_test.jsonl")
