import os
import pandas as pd
from bs4 import BeautifulSoup #parsing xml files from corpora with beautifulsoup

# passing the path as os.path.dirname of the directory 
# containing the xml files in the variable
# corpus_path
corpus_path = "/home/oenni/Dokumente/ArgumentationCorpora/Dataset-ArgumentationEmotions-IJCAI2015/xml-relations" #/debate-Abortion-25Nov.xml"

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
        b_attack = Bs_data.find_all("pair",{"relation":"attack"})
        b_support = Bs_data.find_all("pair", {"relation":"support"})

        # extracting the raw text of attacks and supports
        attacks = [attack.text.split("\n")[1:-1] for attack in b_attack]
        supports = [support.text.split("\n")[1:-1] for support in b_support]

        #print(attacks[1])
        #print(supports[1])


        # generate the right amount of labels
        labels = ["support" for i in range(len(supports))] + ["attack" for i in range(len(attacks))]

        # separate the first and second part of the argument pairs
        p1 = [support[0] for support in supports] + [attack[0] for attack in attacks]
        p2 = [support[1] for support in supports] + [attack[1] for attack in attacks]

        # create IDs for later identifiability of the data
        ID = []
        for i in range(len(labels)):
            if i < 10:
                ID.append(f"IJCAI2015_0{b_debate_id}_0{i}")
            else:
                ID.append(f"AAAI2018_0{b_debate_id}_{i}")


        # create a pd DataFrame from the obtained data and write it out
        df = pd.DataFrame({"ID":ID, "relation":labels, "seg1_text":p1, "seg2_text":p2})

        df.to_csv(f"/home/oenni/Dokumente/NLI-Argumentation-project/corpus/IJCAI2015_parsed/IJCAI2015_debate_0{b_debate_id}.tsv", sep="\t", index=False)
        
