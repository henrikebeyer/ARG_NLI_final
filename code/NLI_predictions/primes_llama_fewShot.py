import pandas as pd

angry_men_path = "/home/oenni/Dokumente/NLI-Argumentation-project/corpus/12AngryMen_parsed/12AngryMen_parsed.tsv"
debatepedia_ext_path = "/home/oenni/Dokumente/NLI-Argumentation-project/corpus/DebatepediaExtended_parsed/Debatepedia_parsed.tsv"
file_path = "/home/oenni/Dokumente/NLI-Argumentation-project/annotation/nli_gold_df.tsv"

#angry_men = pd.read_csv(angry_men_path, sep="\t")
#debatepedia = pd.read_csv(debatepedia_ext_path, sep="\t")
#annot = pd.read_csv(file_path, sep="\t")

#print(angry_men)
#print(debatepedia)
#print(annot)

data = pd.read_csv(file_path, sep="\t")

print(data)

nums = [2, 4, 6, 8, 16, 32, 64]
rand = [42]

for randint in rand:
    for num in nums:
        primes = data.sample(n = num, random_state=randint)
        primes["nli"] = ["neutral" if str(label).lower() not in ["entailment", "contradiction"] else label.lower() for label in primes["nli"]]
        #print(primes["nli"])
        samples = data.drop(primes.index)

        primes.to_csv(f"/home/oenni/Dokumente/NLI-Argumentation-project/NLI_labeling/priming/LLM_primes_{num}_{randint}.tsv", sep="\t", index=False)
        samples.to_csv(f"/home/oenni/Dokumente/NLI-Argumentation-project/NLI_labeling/priming/LLM_samples_{num}_{randint}.tsv", sep="\t", index=False)