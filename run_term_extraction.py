from scripts.automatic_term_extraction import read_recits_from_text_file, get_list_preprocess_recit, \
    f_tfidf_c_value_term_extraction

# English récits
text_file_EN = "dataset/EN/recits_EN/english_recits.txt"
list_text_file_EN = read_recits_from_text_file(text_file_EN)
list_preprocess_text_file_EN = get_list_preprocess_recit(list_text_file_EN, lang="en")
# Term extraction process
result_EN = f_tfidf_c_value_term_extraction(list_preprocess_text_file_EN, "en")
result_EN.to_excel('dataset/EN/beginner/tf_idf_en.xlsx', index=False)

# French récits
text_file_FR = "dataset/FR/recits_FR/french_recits.txt"
list_text_file_FR = read_recits_from_text_file(text_file_FR)
list_preprocess_text_file_FR = get_list_preprocess_recit(list_text_file_FR, lang="fr")
# Term extraction process
result_FR = f_tfidf_c_value_term_extraction(list_preprocess_text_file_FR, "fr")
result_FR.to_excel('dataset/FR/beginner/tf_idf_fr.xlsx', index=False)
