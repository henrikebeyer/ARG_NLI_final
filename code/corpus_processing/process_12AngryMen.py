import os
import pandas as pd
from bs4 import BeautifulSoup #parsing xml files from corpora with beautifulsoup

# NOTE: It is necessary to replace all accitential uppercase "ENTAILMENT" as XML labels
# with "entailment" so that the script can run

# passing the path as os.path.dirname of the directory 
# containing the xml files in the variable
# corpus_path
corpus_path = "/home/oenni/Dokumente/ArgumentationCorpora/12AngryMen/12AngryMen_final_dataset.xml"

with open(corpus_path, "r", encoding="utf-8") as f:
    data = f.read()

# passing the stored data inside
# the beautifulsoup parser, storing
# the returned object

Bs_data = BeautifulSoup(data, "xml")

pairs = Bs_data.find_all("pair")

pairs = Bs_data.find_all("pair")


labels = [pair["BAF"] for pair in pairs]
nli_labels = [pair["entailment"] for pair in pairs]
topics = [pair["topic"] for pair in pairs]
IDs = [f"12Ang_00{idx["id"]}" if int(idx["id"]) < 10 else f"12Ang_0{idx["id"]}" if int(idx["id"]) < 100 else f"12Ang_{idx["id"]}" for idx in pairs]
a1 = [pair.find("t").text for pair in pairs]
a2 = [pair.find("h").text for pair in pairs]


df = pd.DataFrame(columns=["ID", "relation", "seg1_text", "seg2_text", "topic", "nli"])

df["ID"] = IDs
df["topic"] = topics
df["seg1_text"] = a1
df["seg2_text"] = a2
df["relation"] = labels
df["nli"] = nli_labels

print(df.value_counts("topic"))
df.to_csv("/home/oenni/Dokumente/NLI-Argumentation-project/corpus/12AngryMen_parsed/12AngryMen_parsed.tsv", sep="\t", index=False)