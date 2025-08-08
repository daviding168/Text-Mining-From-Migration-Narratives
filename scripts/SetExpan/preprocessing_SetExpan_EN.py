import pandas as pd
from scripts.SetExpan.preprocessing_functions import *
from scripts.SetExpan.probase import ProbaseConcept
from scripts.SetExpan.dataLoader import loadMap, writeMapToFile
from collections import defaultdict
import fasttext
import fasttext.util

def run_all_the_preprocessing_for_english():
    beginner_folder = "dataset/EN/beginner/"
    intermediate_folder = "dataset/EN/intermediate/"
    file_terms_eng = pd.read_excel(beginner_folder + "tf_idf_en.xlsx")
    list_terms_eng_term = list(file_terms_eng['term'])
    with open(intermediate_folder + "entity2id.txt", "w") as f:
        for i in range(0, len(list_terms_eng_term)):
            f.write("{}\t{}\n".format(list_terms_eng_term[i], i))

    data_1 = json.load(open(beginner_folder + "tf_idf_en_one_term_tag.json"))
    with open(beginner_folder + "tf_idf_en_one_term_tag.txt", "w") as f:
        for i in range(0, len(data_1)):
            f.write(data_1[str(i)])
            f.write("\n")

    data_2 = json.load(open(beginner_folder + "tf_idf_en_two_terms_tag.json"))
    with open(beginner_folder + "tf_idf_en_two_terms_tag.txt", "w") as f:
        for i in range(0, len(data_2)):
            f.write(data_2[str(i)])
            f.write("\n")

    process_corpus(beginner_folder + "tf_idf_en_one_term_tag.txt",
                   beginner_folder + "tf_idf_en_two_terms_tag.txt",
                   beginner_folder + "sentences_en_one_term.json",
                   beginner_folder + "sentences_en_two_terms.json", intermediate_folder + "entity2id.txt", lang="en",
                   real_suffix="recit")

    merge_term(beginner_folder + "sentences_en_one_term.json", beginner_folder + "sentences_en_two_terms.json",
               intermediate_folder + "sentences_english_terms.json")

    # features extraction
    inputfilename = "sentences_english_terms.json"
    extractFeatures(intermediate_folder + inputfilename, intermediate_folder)
    featureName = "Skipgram"
    inputFileName = intermediate_folder + 'eid' + featureName + 'Counts.txt'
    outputFileName = intermediate_folder + 'eid' + featureName + '2TFIDFStrength.txt'
    calculate_TFIDF_strength_new(inputFileName, outputFileName)

    # Probase type extraction
    phrase_file = intermediate_folder + "entity2id.txt"  # entity2id.txt
    save_file = intermediate_folder + "linked_results.txt"  # linked results
    pb_database = ProbaseConcept("dataset/EN/KB/data-concept-instance-relations.txt")
    get_probase(phrase_file, save_file, pb_database, topK=15)

    eidMapFilename = intermediate_folder + 'entity2id.txt'
    probaseLinkedFile = intermediate_folder + 'linked_results.txt'
    outFile = intermediate_folder + 'eidTypeCounts.txt'

    ent2eidMap = loadMap(eidMapFilename)
    eidType2count = defaultdict(float)

    with open(probaseLinkedFile) as fin:
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

    ft = fasttext.load_model('dataset/EN/embeddings/wiki.en.bin')
    fasttext.util.reduce_model(ft, 100)
    ft.save_model('dataset/EN/embeddings/wiki.en.100.bin')

    # pre-trained skip-gram model
    ft_wiki_skipgram = fasttext.load_model('dataset/EN/embeddings/wiki.en.100.bin')

    with open(intermediate_folder + "entity2id.txt", "r") as fin, open(intermediate_folder + "eid2embed.txt",
                                                                       "w") as fout:
        all_entIdPair = list(fin.readlines())
        for entId in all_entIdPair:
            entIdPair = entId.split("\t")
            entId = list(map(lambda s: s.strip(), entIdPair))
            ent = entId[0]
            # entity = entId[0]
            entity = ent.replace("_", " ")
            Id = entId[1]
            # multiple-words term
            if len(entity.split(" ")) > 1:
                multi_words = entity.split(" ")[0] + "_" + entity.split(" ")[1]
                vector_string = " ".join([str(ele) for ele in list(ft_wiki_skipgram.get_word_vector(multi_words))])
            # one word term
            else:
                vector_string = " ".join([str(ele) for ele in list(ft_wiki_skipgram.get_word_vector(entity))])
            fout.write("{} {}\n".format(Id, vector_string))

    featureName = 'Type'
    inputFileName = intermediate_folder + 'eid' + featureName + 'Counts.txt'
    outputFileName = intermediate_folder + 'eid' + featureName + '2TFIDFStrength.txt'
    calculate_TFIDF_strength_new(inputFileName, outputFileName)
