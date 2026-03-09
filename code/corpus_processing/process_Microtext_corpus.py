import os
import pandas as pd
import xmltodict

 #parsing xml files from corpora with beautifulsoup

# passing the path as os.path.dirname of the directory 
# containing the xml files in the variable
# corpus_path
corpus_path = "/home/oenni/Dokumente/ArgumentationCorpora/arg-microtexts-master/corpus/en" #/micro_b001.xml" #/debate-AssistedSuicide-S3.xml"

collected_df = []

for filename in os.listdir(corpus_path):
    #print(filename)
    if filename.endswith(".xml"):
        print(filename)
        with open(f"{corpus_path}/{filename}", "r", encoding="utf-8") as f:
            data = f.read()

        # passing the stored data inside
        # the beautifulsoup parser, storing
        # the returned object
        dicti = xmltodict.parse(data)

        pairing = []

        # make a mapping dict for edu to adu id based on the "seg"-type edges
        adu_mapping_dict = {}
        edges = dicti["arggraph"]["edge"]
        edus = dicti["arggraph"]["edu"]
        for edge in edges:
            if edge["@type"] == "seg":
                for edu in edus:
                    if edu["@id"] == edge["@src"]:
                        text = edu["#text"]
                adu_mapping_dict[edge["@trg"]] = text

        i = 0
        for edge in edges:
            if edge["@type"] in ["sup", "exa" ,"reb"]:
                pairing.append({"ID":f"{dicti["arggraph"]["@id"]}_0{i}" if i < 10 else f"x_{i}",
                                "relation":"attack" if edge["@type"]=="reb" else "support", 
                                "seg1_text":adu_mapping_dict[edge["@src"]],
                                "seg2_text":adu_mapping_dict[edge["@trg"]],
                                "topic":None if "@topic_id" not in dicti["arggraph"].keys() else dicti["arggraph"]["@topic_id"]})
                i += 1



        #for edge in b_edge:
        #    if edge["src"].startswith("e") and edge["trg"].startswith("a"):


        df = pd.DataFrame(pairing)

        collected_df.append(df)
            #except:
                #print(filename, "not processed")

full_df = pd.concat(collected_df, axis=0)

print(full_df.value_counts("relation"))
print(full_df.value_counts("topic"))

full_df.to_csv("/home/oenni/Dokumente/NLI-Argumentation-project/corpus/Microtext_parsed/Microtext_parsed.tsv", sep="\t", index=False)
