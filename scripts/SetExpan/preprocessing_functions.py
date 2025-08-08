import time
import json
import re
from tqdm import tqdm
from collections import deque
import spacy
from spacy.symbols import POS, TAG
import mmap
from itertools import groupby
import itertools

from math import log
from collections import defaultdict
from scripts.SetExpan.probase import ProbaseConcept
from transformers import pipeline


DEBUG = True

# INIT SpaCy
nlp_en = spacy.load('en_core_web_sm')
nlp_en.get_pipe("attribute_ruler").add([[{"TEXT": "<phrase>"}]],
                                       {"LEMMA": "", POS: "START_PHRASE", TAG: "START_PHRASE"})
nlp_en.get_pipe("attribute_ruler").add([[{"TEXT": "</phrase>"}]], {"LEMMA": "", POS: "END_PHRASE", TAG: "END_PHRASE"})
p2tok_list_en = {}  # global cache of phrase to token

# nlp_fr = spacy.load('fr_core_news_sm')
nlp_fr = spacy.load("fr_dep_news_trf")

nlp_fr.get_pipe("attribute_ruler").add([[{"TEXT": "<phrase>"}]],
                                       {"LEMMA": "", POS: "START_PHRASE", TAG: "START_PHRASE"})
nlp_fr.get_pipe("attribute_ruler").add([[{"TEXT": "</phrase>"}]], {"LEMMA": "", POS: "END_PHRASE", TAG: "END_PHRASE"})
p2tok_list_fr = {}  # global cache of phrase to token


def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


def clean_text(text):
    text = re.sub(r"[^\x00-\x7FÀ-ÖØ-öø-ÿa-z\u0161\u0111\u010D\u0107\u017E]+$/gi", '', text)
    # add space before and after <phrase> tags
    text = re.sub(r"<phrase>", " <phrase> ", text)
    text = re.sub(r"</phrase>", " </phrase> ", text)
    # text = re.sub(r"<phrase>", " ", text)
    # text = re.sub(r"</phrase>", " ", text)
    # add space before and after special characters
    # text = re.sub(r"([.,!:?()])", r" \1 ", text)
    # replace multiple continuous whitespace with a single whitespace
    text = re.sub(r"\s{2,}", " ", text)
    # text = text.replace("-", " ")

    return text


def find(haystack, needle):
    """Return the index at which the sequence needle appears in the
    sequence haystack, or -1 if it is not found, using the Boyer-
    Moore-Horspool algorithm. The elements of needle and haystack must
    be hashable.

    >>> find([1, 1, 2], [1, 2])
    1

    """
    h = len(haystack)
    n = len(needle)
    skip = {needle[i]: n - i - 1 for i in range(n - 1)}
    i = n - 1
    while i < h:
        for j in range(n):
            if haystack[i - j].lower() != needle[-j - 1]:
                i += skip.get(haystack[i].lower(), n)
                break
        else:
            return i - n + 1
    return -1


def obtain_p_tokens(p, lang):
    '''
    :param p: a phrase string
    :return: a list of token text
    '''

    if lang == "en":

        if p in p2tok_list_en:
            return p2tok_list_en[p]
        else:
            p_tokens = [tok.text for tok in nlp_en(p)]
            p2tok_list_en[p] = p_tokens
            return p_tokens

    elif lang == "fr":
        if p in p2tok_list_fr:
            return p2tok_list_fr[p]
        else:
            p_tokens = [tok.text for tok in nlp_fr(p)]
            p2tok_list_fr[p] = p_tokens
            return p_tokens


def fix_apostrophe(text):
    return re.sub(r"\b(\w)\s+'", r"\1'", text)


def deduplicate(ems):
    """Remove duplicates in entity mention list

    ems: a dictionary
    return: a dictionary without duplicates
    """
    uniques = set([(em['entityId'], em['start'], em['end'], em['text'], em['type']) for em in ems])
    res = []
    for ele in uniques:
        res.append({'entityId': ele[0], 'start': ele[1], 'end': ele[2], 'text': ele[3], 'type': ele[4]})
    return res


def process_one_doc(article, articleId, entity2id, lang):
    result = []
    # list of all the word in between <phrase></phrase>
    phrases = []
    output_token_list = []

    # go over once
    article = clean_text(article)
    q = deque()
    IN_PHRASE_FLAG = False
    for token in article.split(" "):
        if token == "":
            continue
        elif token == "<phrase>":
            IN_PHRASE_FLAG = True
        elif token == "</phrase>":
            current_phrase_list = []
            while len(q) != 0:
                current_phrase_list.append(q.popleft())
            # phrases.append(" ".join(current_phrase_list).lower())
            phrases.append(" ".join(current_phrase_list))
            IN_PHRASE_FLAG = False
        else:
            if IN_PHRASE_FLAG:  # in the middle of a phrase, push the token into queue
                q.append(token)

            # put all the token information into the output fields
            output_token_list.append(token)

    # remove redundant successive elements from list
    new_output_token_list = [key for key, _group in groupby(output_token_list)]

    text = " ".join(new_output_token_list)

    if lang == "en":
        doc = nlp_en(text)
    elif lang == "fr":
        doc = nlp_fr(text)

    sentId = 0
    for sent in doc.sents:  # doc.sents is just to separate a text into several sentences
        # print(sent)

        NPs = []
        pos = []
        lemmas = []
        deps = []
        tokens = []

        # for each sentence get its noun phrases or noun chunks
        for s in sent.noun_chunks:
            NPs.append(s)
        # get pos tag and dependencies for each sentence
        for token in sent:
            tokens.append(token.text)
            pos.append(token.tag_)
            lemmas.append(token.lemma_)
            deps.append(token.dep_)

        entityMentions = []
        # loop through all the quality or tag word, e.g <phrase>car</phrase>, check if it's a part of NP
        for p in phrases:  # all the tag words
            for np in NPs:
                # find if p is a substring of np [in fact, some tag entities disappeared here because they do not belong to noun phrases (NP)]
                if np.text.lower().find(p) != -1:
                    # start offset of a current sentence
                    sent_offset = sent.start
                    # check language
                    if lang == "en":
                        p_tokens = obtain_p_tokens(p, "en")  # Just to partition p into several tokens.
                    elif lang == "fr":
                        p_tokens = obtain_p_tokens(p, "fr")  # Just to partition p into several tokens.

                    # find token of the p_tokens
                    offset = find(tokens[np.start - sent_offset:np.end - sent_offset], p_tokens)
                    if offset == -1:
                        # SKIP THIS AS THIS IS NOT EXACTLY MATCH
                        continue

                    start_offset = np.start + offset - sent_offset

                    if lang == "en":
                        text_phrase = " ".join(p_tokens)

                    elif lang == "fr":
                        if "'" in p_tokens:
                            text_phrase = fix_apostrophe(" ".join(p_tokens))
                        else:
                            text_phrase = " ".join(p_tokens)
                    entity_id = entity2id[text_phrase]

                    ent = {"entityId": entity_id, "text": text_phrase, "start": start_offset,
                           "end": start_offset + len(p_tokens) - 1, "type": "phrase"}

                    # sanity check
                    if lang == "en":
                        if ent["text"] != " ".join(x.lower() for x in tokens[ent["start"]:ent["end"] + 1]):
                            print("NOT MATCH", p, " ".join(tokens[ent["start"]:ent["end"] + 1]))
                            print("SENT", " ".join(tokens))
                            print("SENT2", sent.text)
                    elif lang == "fr":
                        if ent["text"] != fix_apostrophe(
                                " ".join(x.lower() for x in tokens[ent["start"]:ent["end"] + 1])):
                            print("NOT MATCH", p, " ".join(tokens[ent["start"]:ent["end"] + 1]))
                            print("SENT", " ".join(tokens))
                            print("SENT2", sent.text)

                    entityMentions.append(ent)

        # remove duplicate entity mentions (## to do: check why there are duplicates in entityMentions)
        new_entityMentions = deduplicate(entityMentions)

        res = {"articleId": articleId, "sentId": sentId, "tokens": tokens, "pos": pos, "lemma": lemmas, "dep": deps,
               "entityMentions": new_entityMentions,
               "np_chunks": [{"text": t.text, "start": t.start - sent.start, "end": t.end - sent.start - 1} for t in
                             NPs]}
        result.append(res)
        sentId += 1

    return result


def process_corpus(input_path_1, input_path_2, output_path_1, output_path_2, entity2id_path, lang, real_suffix):
    entity2id = {}
    with open(entity2id_path, "r") as f:
        for entity_id in f.readlines():
            list_entity_id = entity_id.split("\t")
            entity2id[list_entity_id[0]] = int(list_entity_id[1])

    start = time.time()

    if lang == "en":
        with open(input_path_1, "r") as fin_1, open(input_path_2, "r") as fin_2, open(output_path_1, "w") as fout_1, \
                open(output_path_2, "w") as fout_2:
            # one term tag
            for cnt, line in tqdm(enumerate(fin_1), total=get_num_lines(input_path_1)):
                line = line.strip()
                # try:
                article_result = process_one_doc(line, "{}-{}".format(real_suffix, cnt), entity2id, lang="en")
                for sent in article_result:
                    json.dump(sent, fout_1)
                    fout_1.write("\n")
            # two terms tag
            for cnt, line in tqdm(enumerate(fin_2), total=get_num_lines(input_path_2)):
                line = line.strip()
                # try:
                article_result = process_one_doc(line, "{}-{}".format(real_suffix, cnt), entity2id, lang="en")
                for sent in article_result:
                    json.dump(sent, fout_2)
                    fout_2.write("\n")
    elif lang == "fr":
        with open(input_path_1, "r") as fin_1, open(input_path_2, "r") as fin_2, open(output_path_1, "w") as fout_1, \
                open(output_path_2, "w") as fout_2:
            # one term tag
            for cnt, line in tqdm(enumerate(fin_1), total=get_num_lines(input_path_1)):
                line = line.strip()
                # try:
                article_result = process_one_doc(line, "{}-{}".format(real_suffix, cnt), entity2id, lang="fr")
                for sent in article_result:
                    json.dump(sent, fout_1)
                    fout_1.write("\n")
            # two terms tag
            for cnt, line in tqdm(enumerate(fin_2), total=get_num_lines(input_path_2)):
                line = line.strip()
                # try:
                article_result = process_one_doc(line, "{}-{}".format(real_suffix, cnt), entity2id, lang="fr")
                for sent in article_result:
                    json.dump(sent, fout_2)
                    fout_2.write("\n")
    end = time.time()

    print("Finish NLP processing, using time %s (second)" % (end - start))


def equal_dicts(d1, d2, ignore_keys):
    d1_filtered = {k: v for k, v in d1.items() if k not in ignore_keys}
    d2_filtered = {k: v for k, v in d2.items() if k not in ignore_keys}
    return d1_filtered == d2_filtered


def merge_dictionary(dic1, dic2):
    dic3 = dic1.copy()
    dic3["entityMentions"] = dic1["entityMentions"] + dic2["entityMentions"]
    return dic3


def merge_term(one_term_path, two_terms_path, output_path):
    with open(one_term_path, "r") as fin_1, open(two_terms_path, "r") as fin_2, open(output_path, "w") as fout:
        content_1 = list(fin_1.readlines())
        content_2 = list(fin_2.readlines())
        for i, j in tqdm(enumerate(content_1), total=len(content_1)):
            dic_1 = json.loads(content_1[i])
            dic_2 = json.loads(content_2[i])
            # verify before merging those dictionaries whether each line is the same
            if equal_dicts(dic_1, dic_2, "entityMentions"):
                dic_merge = merge_dictionary(dic_1, dic_2)
            else:
                print("Attention!! Line {} does not have the same dictionary.".format(i))
            json.dump(dic_merge, fout)
            fout.write("\n")


# features extraction functions
def getSkipgrams(tokens, start, end):
    cleaned_tokens = []
    for tok in tokens:
        if tok == "\t":
            cleaned_tokens.append("TAB")
        else:
            cleaned_tokens.append(tok)
    # 6 skip-pattern features
    positions = [(-1, 1), (-2, 1), (-3, 1), (-1, 3), (-2, 2), (-1, 2)]
    skipgrams = []
    for pos in positions:
        sg = ' '.join(cleaned_tokens[start + pos[0]:start]) + ' __ ' + ' '.join(
            cleaned_tokens[end + 1:end + 1 + pos[1]]).lower()

        skipgrams.append(sg)

    # return list(set(skipgrams))
    return skipgrams


def processSentence(sent):
    sentInfo = json.loads(sent)
    eidSkipgrams = {}
    eidPairs = []
    tokens = sentInfo['tokens']
    eids = set()
    for em in sentInfo['entityMentions']:
        eid = em['entityId']

        start = em['start']
        end = em['end']
        eids.add(eid)

        for skipgram in getSkipgrams(tokens, start, end):
            key = (eid, skipgram)
            if key in eidSkipgrams:
                eidSkipgrams[key] += 1
            else:
                eidSkipgrams[key] = 1

    for pair in itertools.combinations(eids, 2):
        eidPairs.append(frozenset(pair))
    return eidSkipgrams, eidPairs


def writeMapToFile(map, outFilename):
    with open(outFilename, 'w') as fout:
        for key in map:
            lkey = list(key)
            fout.write(str(lkey[0]) + '\t' + str(lkey[1]) + '\t' + str(map[key]) + '\n')


def updateMapFromMap(fromMap, toMap):
    for key in fromMap:
        if key in toMap:
            toMap[key] += fromMap[key]
        else:
            toMap[key] = fromMap[key]
    return toMap


def updateMapFromList(fromList, toMap):
    for ele in fromList:
        if ele in toMap:
            toMap[ele] += 1
        else:
            toMap[ele] = 1
    return toMap


def extractFeatures(infilename, outputFolder):
    eidSkipgramCounts = {}
    eidPairCounts = {}  # entity sentence-level co-occurrence features
    with open(infilename, 'r') as fin:
        for line in tqdm(fin, total=get_num_lines(infilename),
                         desc="Generating skipgram and sentence-level co-occurrence features"):
            eidSkipgrams, eidPairs = processSentence(line)
            updateMapFromMap(eidSkipgrams, eidSkipgramCounts)
            updateMapFromList(eidPairs, eidPairCounts)

    writeMapToFile(eidSkipgramCounts, outputFolder + 'eidSkipgramCounts.txt')
    writeMapToFile(eidPairCounts, outputFolder + 'eidSentPairCount.txt')


def calculate_TFIDF_strength_new(inputFileName, outputFileName):
    eid_w_feature2count = defaultdict()  # mapping between (eid, feature) -> count
    feature2eidcount = defaultdict(int)  # number of distinct eids that match a given feature
    feature2eidcountsum = defaultdict(int)  # total occurrence of different eids that match a given feature

    eid_set = set()
    with open(inputFileName, "r") as fin:
        for line in tqdm(fin, total=get_num_lines(inputFileName),
                         desc="Transform Features in {}".format(inputFileName)):
            seg = line.strip().split("\t")
            try:
                eid = seg[0]
                feature = seg[1]
                count = float(seg[2])
            except:
                print(seg)
                print(eid, feature, count)
                return

            eid_set.add(eid)
            eid_w_feature2count[(eid, feature)] = count
            feature2eidcount[feature] += 1
            feature2eidcountsum[feature] += count

    print("\n[INFO] Start calculating TF-IDF strength")
    #print("All vocabularies: ", eid_set)
    E = len(eid_set)  # vocabulary size
    print("Vocabulary size: ", E)

    with open(outputFileName, "w") as fout:
        for key in tqdm(eid_w_feature2count.keys(), desc="Process (eid, feature) pairs"):
            # raw co-occurrence count between entity e and context feature c
            X_e_c = eid_w_feature2count[key]
            # context feature
            feature = key[1]
            # (for entity e and context feature)
            # weight between each pair of entity e and context feature c using TF-IDF with number of distinct eids that match this feature
            # (for entity e and its concepts)
            # weight between each pair of entity e and its concepts using TF-IDF with number of distinct eids that match a given concept
            f_e_c_count = log(1 + X_e_c) * (log(E) - log(feature2eidcount[feature]))
            # (for entity e and context feature)
            # weight between each pair of entity e and context feature c using TF-IDF with total occurrence of eids matched this feature
            # (for entity e and its concepts)
            # weight between each pair of entity e and its concepts using TF-IDF with total occurrence of eids that match a given concept
            f_e_c_strength = log(1 + X_e_c) * (log(E) - log(feature2eidcountsum[feature]))

            fout.write(key[0] + "\t" + key[1] + "\t" + str(f_e_c_count) + "\t" + str(f_e_c_strength) + "\n")


def get_probase(phrase_file, save_file, pb_database: ProbaseConcept, topK):
    with open(phrase_file, "r") as fin, open(save_file, "w") as fout:
        list_entityIdpair = list(fin.readlines())
        for i in range(0, len(list_entityIdpair)):
            entity = list_entityIdpair[i].split("\t")[0]
            # alter word with "_" to " "
            entity = entity.replace("_", " ")
            fout.write("{}\t{}\n".format(entity, pb_database.conceptualize(entity, topK=topK)))


def get_list_tuple_type_score(camembert_fill_mask: pipeline, entity, topK):
    """

    Parameters
    ----------
    camembert_fill_mask
    entity: input entity for generating types
    topK: top k type to return

    Returns
    -------
    it will generate a French knowledge base
    """
    list_dict_type = camembert_fill_mask(entity + " est le type de <mask>.", top_k=topK)
    list_tuple_type_score = []
    for dict_type in list_dict_type:
        list_tuple_type_score.append((dict_type["token_str"], dict_type["score"]))

    return list_tuple_type_score


def get_MLM_type(phrase_file, save_file, camembert_fill_mask, topK):
    """

    Parameters
    ----------
    phrase_file
    save_file
    camembert_fill_mask
    topK

    Returns
    -------
    Write topK types of each input entity into the text file
    """
    with open(phrase_file, "r") as fin, open(save_file, "w") as fout:
        list_entityIdpair = list(fin.readlines())
        for i in range(0, len(list_entityIdpair)):
            entity = list_entityIdpair[i].split("\t")[0]
            # alter word with "_" to " "
            entity = entity.replace("_", " ")
            fout.write("{}\t{}\n".format(entity, get_list_tuple_type_score(camembert_fill_mask, entity, topK)))
