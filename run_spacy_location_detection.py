from scripts.spacy_location_detection import *

output_dict_spacy_en = get_annotated_dict_infos_recit_text_from_json_file(
    file_path="annotated_mention_entity/pre-output_spacy_english_recits.json")
output_annotated_dict_en = get_annotated_dict_infos_recit_text_from_json_file(
    file_path="annotated_mention_entity/annotated_english_recits.json")

avg_prec_en, avg_rec_en, avg_f1_score_en = calculate_avg_precision_recall_f1_score_for_spacy(
    dict_annotated_recit=output_annotated_dict_en, dict_spacy_recit=output_dict_spacy_en)

print("English narratives")
print("Average precision: ", avg_prec_en)
print("Average recall: ", avg_rec_en)
print("Average f1-score: ", avg_f1_score_en)

output_dict_spacy_fr = get_annotated_dict_infos_recit_text_from_json_file(
    file_path="annotated_mention_entity/pre-output_spacy_french_recits.json")
output_annotated_dict_fr = get_annotated_dict_infos_recit_text_from_json_file(
    file_path="annotated_mention_entity/annotated_french_recits.json")

avg_prec_fr, avg_rec_fr, avg_f1_score_fr = calculate_avg_precision_recall_f1_score_for_spacy(
    dict_annotated_recit=output_annotated_dict_fr, dict_spacy_recit=output_dict_spacy_fr)

print("French narratives")
print("Average precision: ", avg_prec_fr)
print("Average recall: ", avg_rec_fr)
print("Average f1-score: ", avg_f1_score_fr)
