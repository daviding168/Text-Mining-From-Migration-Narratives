from collections import defaultdict
from scripts.SetExpan.util import getMostProbableNodeIdx, get_all_terms, precision_recall_f1_score, \
    evaluation_local_for_each_concept_terms, taxonomy_subtract
from scripts.SetExpan.dataLoader import loadEidToEntityMap, loadFeaturesAndEidMap, loadWeightByEidAndFeatureMap, \
    loadEntityEmbedding
from scripts.SetExpan.seedLoader import load_seeds, annotated_taxonomy
from scripts.SetExpan.treeNode import TreeNode
from scripts.SetExpan.set_expan_standalone_fr import setExpan

# the maximum size of output taxonomy tree
level2max_children = {-1: 4, 0: 50}
# the feature relative weights used for expanding a level x node's children
level2source_weights = {
    0: {"sg": 1.0, "tp": 20.0, "eb": 1.0},
    1: {"sg": 1.0, "tp": 20.0, "eb": 1.0}
}
# the maximum expanded entity number in each iteration under each node
level2max_expand_eids = {-1: 0, 0: 10}
# the global level-wise reference_edges between each two levels
level2reference_edges = defaultdict(list)
level2threshold = {"fr": {"Hébergement": 0.72, "Moyens de transport": 0.76, "Environnement": 0.8, "Membres de famille": 0.75}}


def runMultiSetExpan(iteration_number=10, lang="en", threshold_boolean=False, debug=False):
    print("=== Start loading data ...... ===")
    folder = 'dataset/' + lang.upper() + '/intermediate/'
    eid2ename, ename2eid = loadEidToEntityMap(folder + 'entity2id.txt')
    eid2patterns, pattern2eids = loadFeaturesAndEidMap(folder + 'eidSkipgramCounts.txt')
    eidAndPattern2strength = loadWeightByEidAndFeatureMap(folder + 'eidSkipgram2TFIDFStrength.txt', idx=-1)
    eid2types, type2eids = loadFeaturesAndEidMap(folder + 'eidTypeCounts.txt')
    eidAndType2strength = loadWeightByEidAndFeatureMap(folder + 'eidType2TFIDFStrength.txt', idx=-1)
    eid2embed, embed_matrix, eid2rank, rank2eid, embed_matrix_array = loadEntityEmbedding(folder + 'eid2embed.txt')
    print("=== Start loading seed supervision ...... ===")
    userInput = load_seeds(lang=lang)
    if len(userInput) == 0:
        print("Terminated due to no user seed. Please specify seed taxonomy in seedLoader.py")
        exit(-1)
    stopLevel = max([ele[1] for ele in userInput]) + 1

    rootNode = None
    ename2treeNode = {}
    for i, node in enumerate(userInput):
        if i == 0:  # ROOT
            rootNode = TreeNode(parent=None, level=-1, eid=-1, ename="ROOT", isUserProvided=True, confidence_score=0.0,
                                max_children=level2max_children[-1])
            ename2treeNode["ROOT"] = rootNode

            label_idx = -2
            # loop through label terms
            for children in node[2]:
                newNode = TreeNode(parent=rootNode, level=0, eid=label_idx, ename=children,
                                   isUserProvided=True, confidence_score=0.0, max_children=level2max_children[0])
                ename2treeNode[children] = newNode
                rootNode.addChildren([newNode])
                label_idx -= 1
        # Surface label names
        else:
            ename = node[0]
            childrens = node[2]

            if ename in ename2treeNode:  # existing node
                parent_treeNode = ename2treeNode[ename]
                # real terms to get expanded
                for children in childrens:
                    newNode = TreeNode(parent=parent_treeNode, level=parent_treeNode.level + 1, eid=ename2eid[children],
                                       ename=children, isUserProvided=True, confidence_score=1.0,
                                       max_children=0)

                    ename2treeNode[children] = newNode
                    parent_treeNode.addChildren([newNode])
                    level2reference_edges[parent_treeNode.level].append((parent_treeNode.eid, newNode.eid))

            else:  # not existing node
                print("[ERROR] disconnected tree node: %s" % node)

    seed_taxonomy_dict = {}
    for node in rootNode.children:
        seed_taxonomy_dict[node.ename] = []
        for child in node.children:
            seed_taxonomy_dict[node.ename].append(child.ename)

    print("=== Storing seed taxonomy for evaluation... ===")
    print("=== Finish loading seed supervision ...... ===")
    print("=== Start MultiSetExpan for " + lang.upper() + " ...... ===")
    update = True
    num_iters = iteration_number
    iters = 0
    # loop of each iteration
    while update:
        eid2nodes = {}
        eidsWithConflicts = set()
        # includes all children nodes of rootNode to expand
        targetNodes = [Node for Node in rootNode.children]
        # loop to update or expand the children under the surface label name
        while len(targetNodes) > 0:
            # expand children under current targetNode for the position 0
            targetNode = targetNodes[0]
            # list of the other targetNode from the position 1
            targetNodes = targetNodes[1:]

            if targetNode.eid >= 0:
                # get eid of the targetNode
                eid = targetNode.eid
                # detect conflicts by eid if it exists in dict eid2nodes
                if eid in eid2nodes:
                    eid2nodes[eid].append(targetNode)
                    # add conflict eid into eidsWithConflicts set
                    eidsWithConflicts.add(eid)
                else:
                    eid2nodes[eid] = [targetNode]

            # targetNode is already leaf node, stop expanding
            if targetNode.level >= stopLevel:
                continue

            # targetNode has enough children, just add children to consider, stop expanding
            if len(targetNode.children) > targetNode.max_children:
                targetNodes += targetNode.children
                # print("[INFO: Reach maximum children at node]:", targetNode)
                continue
            # Width expansion: obtain ordered new childrenEids
            seedEidsWithConfidence = [(child.eid, child.confidence_score) for child in targetNode.children]

            negativeSeedEids = targetNode.restrictions
            negativeSeedEids.add(targetNode.eid)  # add parent eid as negative example into SetExpan
            if debug:
                print("[Width Expansion] Expand: {}, restrictions: {}".format(targetNode, negativeSeedEids))
            # at least grow one node
            max_expand_eids = max(len(negativeSeedEids) + 1, level2max_expand_eids[targetNode.level])
            newOrderedChildrenEidsWithConfidence = setExpan(seedEidsWithConfidence, negativeSeedEids, eid2patterns,
                                                            pattern2eids, eidAndPattern2strength, eid2types, type2eids,
                                                            eidAndType2strength, eid2ename, eid2embed,
                                                            source_weights=level2source_weights[targetNode.level],
                                                            max_expand_eids=max_expand_eids, use_embed=True,
                                                            use_type=True)
            newOrderedChildren = []
            for ele in newOrderedChildrenEidsWithConfidence:
                # eid
                newChildEid = ele[0]
                # confidence_score
                confidence_score = ele[1]
                # new expanded child (the leaf node), the instances of each surface label name
                newChild = TreeNode(parent=targetNode, level=targetNode.level + 1, eid=newChildEid,
                                    ename=eid2ename[newChildEid], isUserProvided=False,
                                    confidence_score=confidence_score,
                                    max_children=targetNode.level)

                # max_children=level2max_children[targetNode.level + 1])
                newOrderedChildren.append(newChild)
            # update current targetNode with new children into the list of all children
            targetNode.addChildren(newOrderedChildren)
            # Add its children as in the queue
            targetNodes += targetNode.children
        # tree is expanded in this iter
        iters += 1
        print("Checking conflict...")
        # check conflicts after each iteration
        nodesToDelete = []
        # loop through eidsWithConflicts set
        for eid in eidsWithConflicts:
            # retrieve list of conflict nodes with the same eid but each with different parents and confidence_score
            conflictNodes = eid2nodes[eid]
            # most probable node index to keep and the rest will delete from their parents
            mostProbableNodeIdx = getMostProbableNodeIdx(conflictNodes)
            for i in range(len(conflictNodes)):
                if i in mostProbableNodeIdx:
                    continue
                nodesToDelete.append(conflictNodes[i])
        # delete conflict nodes from their parents
        for node in nodesToDelete:
            node.parent.cutFromChild(node)
            node.delete()

        if threshold_boolean:
            print("Truncating taxonomy based on predefined threshold...")
            nodesToTruncate = []
            for node in rootNode.children:
                for child in node.children:
                    if child.confidence_score < level2threshold[lang][node.ename]:
                        nodesToTruncate.append(child)
            for node in nodesToTruncate:
                node.parent.cutFromChild(node)
                node.delete()
        print("=== Taxonomy Tree at iteration %s ===" % iters)

        if iters >= num_iters:
            break

    expanded_taxonomy_dict = {}
    for node in rootNode.children:
        expanded_taxonomy_dict[node.ename] = []
        for child in node.children:
            expanded_taxonomy_dict[node.ename].append(child.ename)

    return expanded_taxonomy_dict, seed_taxonomy_dict
