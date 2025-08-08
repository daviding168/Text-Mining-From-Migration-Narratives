import pandas as pd
import json
from scripts.SetExpan.preprocessing_functions import process_corpus, merge_term, extractFeatures, calculate_TFIDF_strength_new, \
    get_MLM_type
from scripts.SetExpan.dataLoader import loadMap, writeMapToFile
import fasttext
import fasttext.util
from transformers import pipeline
from collections import defaultdict



def run_all_the_preprocessing_for_french():
    beginner_folder = "dataset/FR/beginner/"
    intermediate_folder = "dataset/FR/intermediate/"

    file_terms_fr = pd.read_excel(beginner_folder + "tf_idf_fr.xlsx")
    terms_fr_df = file_terms_fr[['term', 'rank']]
    list_terms_fr_term = list(file_terms_fr['term'])

    entity2id = {}
    with open(intermediate_folder + "entity2id.txt", "w") as f:
        for i in range(0, len(list_terms_fr_term)):
            f.write("{}\t{}\n".format(list_terms_fr_term[i], i))

    data_1 = json.load(open(beginner_folder + "tf_idf_fr_one_term_tag.json"))
    with open(beginner_folder + "tf_idf_fr_one_term_tag.txt", "w") as f:
        for i in range(0, len(data_1)):
            f.write(data_1[str(i)])
            f.write("\n")

    data_2 = json.load(open(beginner_folder + "tf_idf_fr_two_terms_tag.json"))
    with open(beginner_folder + "tf_idf_fr_two_terms_tag.txt", "w") as f:
        for i in range(0, len(data_2)):
            f.write(data_2[str(i)])
            f.write("\n")

    process_corpus(beginner_folder + "tf_idf_fr_one_term_tag.txt",
                   beginner_folder + "tf_idf_fr_two_terms_tag.txt",
                   beginner_folder + "sentences_fr_one_term.json", beginner_folder + "sentences_fr_two_terms.json",
                   intermediate_folder + "entity2id.txt", lang="fr", real_suffix="recit")

    merge_term(beginner_folder + "sentences_fr_one_term.json", beginner_folder + "sentences_fr_two_terms.json",
               intermediate_folder + "sentences_french_terms.json")

    # features extraction
    inputfilename = 'sentences_french_terms.json'
    extractFeatures(intermediate_folder + inputfilename, intermediate_folder)

    # Calculate TF-IDF strength using SkipgramCounts.txt
    featureName = 'Skipgram'
    inputFileName = intermediate_folder + 'eid' + featureName + 'Counts.txt'
    outputFileName = intermediate_folder + 'eid' + featureName + '2TFIDFStrength.txt'
    calculate_TFIDF_strength_new(inputFileName, outputFileName)

    ft = fasttext.load_model('dataset/FR/embeddings/cc.fr.300.bin')
    fasttext.util.reduce_model(ft, 100)
    ft.save_model('dataset/FR/embeddings/cc.fr.100.bin')
    ft_wiki_cbow = fasttext.load_model('dataset/FR/embeddings/cc.fr.100.bin')

    with open(intermediate_folder + "entity2id.txt", "r") as fin, open(intermediate_folder + "eid2embed.txt", "w") as fout:
        all_entIdPair = list(fin.readlines())
        for entId in all_entIdPair:
            entIdPair = entId.split("\t")
            entId = list(map(lambda s: s.strip(), entIdPair))
            ent = entId[0]
            entity = ent.replace("_", " ")
            Id = entId[1]
            # multiple-words term
            if len(entity.split(" ")) > 1:
                multi_words = entity.split(" ")[0] + "_" + entity.split(" ")[1]
                # vector_string = " ".join([str(ele) for ele in list(ft_wiki_cbow.get_sentence_vector(entity))])
                vector_string = " ".join([str(ele) for ele in list(ft_wiki_cbow.get_word_vector(multi_words))])
            # one word term
            else:
                vector_string = " ".join([str(ele) for ele in list(ft_wiki_cbow.get_word_vector(entity))])
            fout.write("{} {}\n".format(Id, vector_string))

    camembert_fill_mask = pipeline("fill-mask", model="camembert-base", tokenizer="camembert-base")
    # camembert_fill_mask = pipeline("fill-mask", model="cmarkea/distilcamembert-base", tokenizer="cmarkea/distilcamembert-base")

    phrase_file = intermediate_folder + "entity2id.txt"  # entity2id.txt
    save_file = intermediate_folder + "linked_results.txt"  # linked results

    get_MLM_type(phrase_file, save_file, camembert_fill_mask, topK=30)

    eidMapFilename = intermediate_folder + 'entity2id.txt'
    CamembertLinkedFile = intermediate_folder + 'linked_results.txt'
    outFile = intermediate_folder + 'eidTypeCounts.txt'

    ent2eidMap = loadMap(eidMapFilename)
    eidType2count = defaultdict(float)

    with open(CamembertLinkedFile) as fin:
        for line in fin:
            seg = line.strip('\r\n').split('\t')
            if len(seg) <= 1:
                continue
            entity = seg[0]
            if entity not in ent2eidMap:
                print("[WARNING] No id entity: {}".format(entity))
                continue
            else:
                eid = ent2eidMap[entity]
            types = eval(seg[1])
            for tup in types:
                t = tup[0]
                strength = tup[1]
                eidType2count[(eid, t)] = strength

    writeMapToFile(eidType2count, outFile)

    featureName = 'Type'
    inputFileName = intermediate_folder + 'eid' + featureName + 'Counts.txt'
    outputFileName = intermediate_folder + 'eid' + featureName + '2TFIDFStrength.txt'
    calculate_TFIDF_strength_new(inputFileName, outputFileName)
