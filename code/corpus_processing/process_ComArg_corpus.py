import os
import pandas as pd
from bs4 import BeautifulSoup


def parse_corpus(subcorpus):
    print(subcorpus)
    corpus_path = f"/home/oenni/Dokumente/ArgumentationCorpora/comarg/{subcorpus}.xml"

    with open(f"{corpus_path}", "r", encoding="utf-8") as f:
        data = f.read()

    # passing the stored data inside
    # the beautifulsoup parser, storing
    # the returned object

    Bs_data = BeautifulSoup(data, "xml")

    units = Bs_data.find_all("unit")

    arg1 = [unit.find("comment").find("text").text for unit in units]
    arg2 = [unit.find("argument").find("text").text for unit in units]
    labels = ["support" if int(unit.find("label").text) > 3 else "attack" if int(unit.find("label").text) < 3 else "unrelated" for unit in units]
    #for unit in units:
    #    arg1 = unit.find("comment").find("text").text
    #    arg2 = unit.find("argument").find("text").text
    #    lab = int(unit.find("label").text)
    #    label = "support" if lab > 3 else "attack" if lab < 3 else "unrelated"

    #    print(arg1, arg2, label)

    ID = [f"ComArg_{subcorpus}_000{i}" if i < 10 else f"ComArg_{subcorpus}_00{i}" if i < 100 else f"ComArg_{subcorpus}_0{i}" if i < 1000 else f"ComArg_{subcorpus}_{i}" for i in range(len(arg1))]
    topic = "gay marriage" if subcorpus == "GM" else "Under God in pledge"

    df = pd.DataFrame()
    df["ID"] = ID
    df["relation"] = labels
    df["seg1_text"] = arg1
    df["seg2_text"] = arg2
    df["topic"] = ["gay marriage" for i in range(len(arg1))]

    print(df.value_counts("relation"))

    df.to_csv(f"/home/oenni/Dokumente/NLI-Argumentation-project/corpus/ComArg_parsed/{subcorpus}_parsed.tsv", sep="\t", index=False)

parse_corpus("UGIP")
parse_corpus("GM")