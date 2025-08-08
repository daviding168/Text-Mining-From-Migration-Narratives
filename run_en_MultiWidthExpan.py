from scripts.SetExpan.main_en import *

iter_num = 10  # can initialize until 15, 20, ... no more quality terms will be included after 10 iteration
expanded_taxonomy, seed_taxonomy = runMultiSetExpan(iteration_number=iter_num, lang="en", threshold_boolean=True,
                                                    debug=False)
# get annotated taxonomy
annotated_taxonomy = annotated_taxonomy(lang="en")

subtract_expanded_taxonomy = taxonomy_subtract(expanded_taxonomy, seed_taxonomy)
subtract_annotated_taxonomy = taxonomy_subtract(annotated_taxonomy, seed_taxonomy)
print("Annotated Sets: ")
print(annotated_taxonomy)
print("\nSet of Seed entities: ")
print(seed_taxonomy)
print("\nThreshold: ")
print(level2threshold)
print("\nExpanded results: ")
print(subtract_expanded_taxonomy)
list_all_expanded_terms = get_all_terms(subtract_expanded_taxonomy)
# list of all terms across different concepts
list_all_annotated_terms = get_all_terms(subtract_annotated_taxonomy)
print("\nAfter ", iter_num, " iterations")
results_taxo_local = evaluation_local_for_each_concept_terms(subtract_annotated_taxonomy,
                                                             subtract_expanded_taxonomy)
print(results_taxo_local)
