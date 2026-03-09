import json
import pandas as pd
import os

# This is to parse IAT annotated graphs
# The output is two .tsv files per json
# One contains the full nodes
# The other contains the relation, the node IDs, and the text

corpus_path = "/home/oenni/Dokumente/ArgumentationCorpora/QT50" #/nodeset28362.json" #/nodeset17918.json"

inf = 0
conf = 0
reph = 0

for filename in os.listdir(corpus_path):
    if filename.endswith(".json"):
        #try:
        print(filename)
        with open(f"{corpus_path}/{filename}", "r") as f:
            data = json.load(f)

        # This is the ist that will capture all the dictionaries of node pairs for one json
        node_pairs = []

        # find all nodes that designate an argumentative relation
        for node in data["nodes"]:
            if node["type"] in ["CA", "RA", "MA"]:
                xa_ID = node["nodeID"]
                xa_type = node["text"]

                # create some temporary helper lists
                ancestors = []
                successors = []
                i1s = []
                i2s = []

                # find all successor and ancestor nodes
                for edge in data["edges"]:
                    if edge["fromID"] == xa_ID:
                        successors.append(edge["toID"])
                    elif edge["toID"] == xa_ID:
                        ancestors.append(edge["fromID"])
                
                # find I-nodes in successors
                for successor_id in successors:
                    for node in data["nodes"]:
                        if node["nodeID"] == successor_id and node["type"] == "I":
                            i1s.append(node)
                
                # find I-nodes in ancestors
                for ancestor_id in ancestors:
                    for node in data["nodes"]:
                        if node["nodeID"] == ancestor_id and node["type"] == "I":
                            i2s.append(node)
                
                # pair all the I1 and I2 nodes that were found connected to one
                # argumentative relation
                pairs = [[i1, i2] for i1 in i1s for i2 in i2s]
                for pair in pairs:
                    node_pairs.append({"Relation":xa_type,
                                    "I1":pair[0],
                                    "I2":pair[1]})

        # based on the I nodes found in relation to the argumentative relation,
        # this bit searches for the corresponding L-nodes

        # first the L1 for the I1
        for pair in node_pairs:
            i1_id = pair["I1"]["nodeID"]
            ancestors = []
            for edge in data["edges"]:
                if edge["toID"] == i1_id:
                    ancestors.append(edge["fromID"])
            for ancestor_id in ancestors:
                for node in data["nodes"]:
                    if node["nodeID"] == ancestor_id and node["type"] == "YA":
                        #print(node)
                        for edge in data["edges"]:
                            if edge["toID"] == ancestor_id:
                                l_candidate_id = edge["fromID"]
                                for node in data["nodes"]:
                                    if node["nodeID"] == l_candidate_id and node["type"] == "L":
                                        pair["L1"] = node
            # In case there is no corresponding L-node, an empty placeholder is created
            if "L1" not in pair.keys():
                pair["L1"] = {"nodeID":None,
                              "text":None}

        # same as for L1 nodes for L2 nodes
        for pair in node_pairs:
            i2_id = pair["I2"]["nodeID"]
            ancestors = []
            for edge in data["edges"]:
                if edge["toID"] == i2_id:
                    ancestors.append(edge["fromID"])
            for ancestor_id in ancestors:
                for node in data["nodes"]:
                    if node["nodeID"] == ancestor_id and node["type"] == "YA":
                        #print(node)
                        for edge in data["edges"]:
                            if edge["toID"] == ancestor_id:
                                l_candidate_id = edge["fromID"]
                                for node in data["nodes"]:
                                    if node["nodeID"] == l_candidate_id and node["type"] == "L":
                                        pair["L2"] = node
            if "L2" not in pair.keys():
                pair["L2"] = {"nodeID":None,
                              "text":None}

        #for pair in node_pairs:
        #    print(pair)

        # create a second dictionary separating node ids an text in nodes
        # also introducing a unique ID for later reference
        node_pairs_text = []
        i = 0
        for pair in node_pairs:
            if int(i) < 10:
                i = f"00{i}"
            elif int(i) > 9 and i < 100:
                i = f"0{i}"
            node_pairs_text.append({"ID":f"QT50_{filename.split(".")[0]}_{i}",
                                    "relation":pair["Relation"],
                                    "I1_id":pair["I1"]["nodeID"],
                                    "I2_id":pair["I2"]["nodeID"],
                                    "I1_text":pair["I1"]["text"],
                                    "I2_text":pair["I2"]["text"],
                                    "L1_id":pair["L1"]["nodeID"],
                                    "L2_id":pair["L2"]["nodeID"],
                                    "L1_text":pair["L1"]["text"],
                                    "L2_text":pair["L2"]["text"]
                                    })
            i = int(i)
            i += 1

        # turn the list of dictionaries to data frames
        df_full = pd.DataFrame(node_pairs)
        df_text = pd.DataFrame(node_pairs_text)

        try:
            inf += df_text.value_counts("relation")["Default Inference"]
            conf += df_text.value_counts("relation")["Default Conflict"]
            reph += df_text.value_counts("relation")["Default Rephrase"]
        except:
            print("x")

        # save the data frames to .tsv
        df_full.to_csv(f"/home/oenni/Dokumente/NLI-Argumentation-project/corpus/QT50_parsed/full/QT50_{filename.split(".")[0]}_full_parsed.tsv", sep="\t", index=False)
        df_text.to_csv(f"/home/oenni/Dokumente/NLI-Argumentation-project/corpus/QT50_parsed/text/QT50_{filename.split(".")[0]}_parsed.tsv", sep="\t", index=False)

print("Inference count:", inf)
print("Conflict count:", conf)
print("Rephrase count:", reph)