import nltk
from TermExtraction.term_extraction.term_extraction import TermExtraction
import re

# set your own proxy
nltk.set_proxy('')


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


def get_list_preprocess_recit(recit_list, lang):
    """
    :param recit_dict: dictionary contains all the text of recit
    :param lang: input language of the corpus
    :return: list of cleaned recit's text
    """

    list_all_corpus = []
    for corpus_dict in recit_list:
        for corpus_info, corpus_text in corpus_dict.items():
            list_all_corpus.append(corpus_text)
    # remove redundant recites
    list_all_text = list(set(list_all_corpus))
    new_list_all_text = []

    for doc in list_all_text:
        doc = doc.strip()
        doc = doc.replace("’", "'")
        doc = re.sub('[“”]', '"', doc)

        if lang == "en":
            # two words term
            doc = doc.replace("-", "_")
            # doc = doc.replace("…", " ")
        # remove non-ascii character except accent letters and š đ č ć ž characters both lower and upper cases
        doc = re.sub(r"[^\x00-\x7FÀ-ÖØ-öø-ÿa-z\u0161\u0111\u010D\u0107\u017E]+$/gi", "", doc)
        # replace multiple continuous punctuations
        doc = re.sub(r"\!+", "! ", doc)
        doc = re.sub(r"\,+", ", ", doc)
        doc = re.sub(r"\?+", "? ", doc)

        # remove "\t" character
        doc = " ".join(doc.split("\t"))

        # replace multiple continuous whitespace with a single whitespace
        doc = re.sub(r"\s{2,}", " ", doc)
        new_list_all_text.append(doc)

    return new_list_all_text


def f_tfidf_c_value_term_extraction(list_text, language):
    '''

    :param list_text: list of recits
    :param language: language of the extracted corpus
    :param threshold_score: minimum score of retained terms
    :return: dataframe of the extracted term
    '''

    if language == "en":
        auto_term_extraction = TermExtraction(language="en")

    elif language == "fr":
        auto_term_extraction = TermExtraction(language="fr")

    res = auto_term_extraction.extract_term_corpus(list_text, "tf_idf")
    res.index.names = ['term']
    res.reset_index(inplace=True)
    res = res[['term', 'freq', 'rank', 'pattern_used']]
    res_order = res.sort_values(by='rank', ascending=False)

    return res_order


'''
# English récits
text_file_EN = "../dataset/EN/recits_EN/english_recits.txt"
list_text_file_EN = read_recits_from_text_file(text_file_EN)
# print(list_text_file_EN)
list_preprocess_text_file_EN = get_list_preprocess_recit(list_text_file_EN, lang="en")
# print(list_preprocess_text_file_EN)
# term extraction process
result_EN = f_tfidf_c_value_term_extraction(list_preprocess_text_file_EN, "en")
result_EN.to_excel('../dataset/EN/beginner/tf_idf_en.xlsx', index=False)


# French récits
text_file_FR = "../dataset/FR/recits_FR/french_recits.txt"
list_text_file_FR = read_recits_from_text_file(text_file_FR)
list_preprocess_text_file_FR = get_list_preprocess_recit(list_text_file_FR, lang="fr")
# Term extraction process
result_FR = f_tfidf_c_value_term_extraction(list_preprocess_text_file_FR, "fr")
result_FR.to_excel('../dataset/FR/beginner/tf_idf_fr.xlsx', index=False)
'''
