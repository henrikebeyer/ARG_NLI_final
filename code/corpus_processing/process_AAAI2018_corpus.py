import os
import pandas as pd
from bs4 import BeautifulSoup #parsing xml files from corpora with beautifulsoup

# passing the path as os.path.dirname of the directory 
# containing the xml files in the variable
# corpus_path
corpus_path = "/home/oenni/Dokumente/ArgumentationCorpora/DatasetAAAI2018/xml-relations" #/debate-AssistedSuicide-S3.xml"

for filename in os.listdir(corpus_path):
    print(filename)
    if filename.endswith(".xml"):
        with open(f"{corpus_path}/{filename}", "r", encoding="utf-8") as f:
            data = f.read()

        # passing the stored data inside
        # the beautifulsoup parser, storing
        # the returned object

        Bs_data = BeautifulSoup(data, "xml")

        # extracting the debate_id for later identification
        b_debate_id = Bs_data.find("argument")["debate_id"]
        
        # extracting attack and support pairs
        b_pairs = Bs_data.find_all("pair")

        # extracting the raw text of attacks and supports
        texts = [t.text.split("\n")[1:-1] for t in b_pairs]
        #supports = [support.text.split("\n")[1:-1] for support in b_support]

        # generate the right amount of labels
        labels = [pair["relation"] for pair in b_pairs]

        # separate the first and second part of the argument pairs
        p1 = [text[0] for text in texts] #+ [attack[0] for attack in attacks]
        p2 = [text[1] for text in texts] #+ [attack[1] for attack in attacks]

        # create IDs for later identifiability of the data
        ID = []
        for i in range(len(labels)):
            if i < 10:
                ID.append(f"AAAI2018_0{b_debate_id}_0{i}")
            else:
                ID.append(f"AAAI2018_0{b_debate_id}_{i}")

        # create a pd DataFrame from the obtained data and write it out
        df = pd.DataFrame(columns=["ID", "relation", "seg1_text", "seg2_text"])
        df["ID"] = ID
        df["seg1_text"] = p1
        df["seg2_text"] = p2
        df["relation"] = labels

        print(df.value_counts("relation"))
        df.to_csv(f"/home/oenni/Dokumente/NLI-Argumentation-project/corpus/AAAI2018_parsed/AAAI2018_debate_0{b_debate_id}.tsv", sep="\t", index=False)