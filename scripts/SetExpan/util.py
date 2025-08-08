from operator import attrgetter

def getMostProbableNodeIdx(nodes):
    """
    get the index of the most probable child with its parents
    :param nodes: a list of TreeNode objects
    :param eid2patterns:
    :param eidAndPattern2strength:
    :return:
    """
    # Sanity checking
    assert len(nodes) > 0
    # verify to make sure that all entities in list have the same eid
    eid = nodes[0].eid
    for i in range(1, len(nodes)):
        assert nodes[i].eid == eid

    # Rule 1: User is the big boss
    for i in range(len(nodes)):
        if nodes[i].isUserProvided:
            return [i]
    max_confidence_score = max(nodes, key=attrgetter('confidence_score')).confidence_score
    return [i for i, j in enumerate(nodes) if j.confidence_score == max_confidence_score]


def get_all_terms(taxonomy):
    """
    function to get a single list contains all the terms of different concepts
    """
    list_all_terms = []

    if isinstance(taxonomy, tuple) and len(taxonomy) == 2: return [taxonomy]

    if isinstance(taxonomy, list):
        if all(isinstance(ele, tuple) for ele in taxonomy):
            return [ele[0] for ele in taxonomy]
        else:
            return taxonomy

    if isinstance(taxonomy, dict):
        taxonomy = taxonomy.values()
    if hasattr(taxonomy, '__iter__') and not isinstance(taxonomy, str):
        for i in taxonomy:
            list_all_terms += get_all_terms(i)

    return list_all_terms


def precision_recall_f1_score(relevant_list, retrieve_list):
    """
    Evaluation of Information Retrieval in terms of precision, recall and f1
    """
    relevant_list = [x.lower() for x in relevant_list]
    retrieve_list = [x.lower() for x in retrieve_list]
    if len(retrieve_list) == 0:
        precision = 0
    else:
        precision = len(list(set(relevant_list) & set(retrieve_list))) / len(retrieve_list)

    recall = len(list(set(relevant_list) & set(retrieve_list))) / len(relevant_list)

    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = (2 * precision * recall) / (precision + recall)

    return precision, recall, f1_score


def evaluation_local_for_each_concept_terms(relevant_taxo, retrieve_taxo):
    """
    function to evaluate each concept separately between annotated taxo and retrieve taxo

    """
    if isinstance(relevant_taxo, list) and isinstance(retrieve_taxo, list):
        precision, recall, f1 = precision_recall_f1_score(relevant_taxo, retrieve_taxo)
        # print(precision, recall, f1)

        final_result = {"precision": precision, "recall": recall, "f1_score": f1}

        return final_result

    if isinstance(relevant_taxo, dict) and isinstance(retrieve_taxo, dict):
        return {k1: evaluation_local_for_each_concept_terms(v1, v2)
                for (k1, v1), (k2, v2) in
                zip(relevant_taxo.items(), retrieve_taxo.items())}


def taxonomy_subtract(expanded_taxonomy, seed_taxonomy):
    subtract_taxonomy = {}
    for (k1, v1), (k2, v2) in zip(expanded_taxonomy.items(), seed_taxonomy.items()):
        subtract_taxonomy[k1] = list(set(v1) - set(v2))

    return subtract_taxonomy
