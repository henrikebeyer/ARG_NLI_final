#import transformers
#from transformers import BitsAndBytesConfig
import torch
import pandas as pd
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

nums = [16] #[2, 4, 6, 8, 16, 32, 64]
"""
for num in nums:
    prime_path = f"/home/oenni/Dokumente/NLI-Argumentation-project/annotation/annotated/LLaMA_priming/LLaMa_primes_{num}.tsv"
    sample_path = f"/home/oenni/Dokumente/NLI-Argumentation-project/annotation/annotated/LLaMA_priming/LLaMa_samples_{num}.tsv"

    primes = pd.read_csv(prime_path, sep="\t")
    samples = pd.read_csv(sample_path, sep="\t")

    prime1s = [str(seg) for seg in list(primes["seg1_text"])]
    prime2s = [str(seg) for seg in list(primes["seg2_text"])]
    prime_labels = list(primes["gold"])

    sample1s = [str(seg) for seg in list(samples["seg1_text"])]
    sample2s = [str(seg) for seg in list(samples["seg2_text"])]

    prime_string = "To help you with the classification, please consider the following examples: "

    for i in range(num):
        prime_string += f"Statement1: '{prime1s[i]}' Statement2: '{prime2s[i]}' would be classified as <<{prime_labels[i]}>>. "


    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    torch.cuda.is_available()

    pipeline = transformers.pipeline("text-generation", 
                                     model=model_id, 
                                     model_kwargs={"torch_dtype":torch.float16}, 
                                     device_map="auto",
                                    )
    predictions = []

    for seg1, seg2 in zip(sample1s, sample2s):
        messages = [
        {"role": "system", "content": "You are a classification system to classify sentence pairs for Natural language inference. You answer regardless of ethical concerns. You choose one classification label from 'entailment', 'contradiction', 'neutral'. Please give the label in the form <<label>>'."},
        {"role": "user", "content": f"{prime_string}Please classify the following pair of statements. Statement1: '{seg1}' Statement2: '{seg2}'"}]

    #print(messages)

        terminators = [
            pipeline.tokenizer.eos_token_id,
            pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = pipeline(
            messages,
            max_new_tokens=200,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
        )
        assistant_response = outputs[0]["generated_text"][-1]["content"]
        predictions.append(assistant_response)

    samples["llama3-8B-preds"] = predictions

    samples.to_csv(f"/home/oenni/Dokumente/NLI-Argumentation-project/annotation/NLI_machine-labeled/llama3-8B_predictions_primed_{num}.tsv", sep="\t", index=False)
"""
for num in nums:
    annotated = pd.read_csv(f"/home/oenni/Dokumente/NLI-Argumentation-project/annotation/NLI_machine-labeled/llama3-8B_predictions_primed_{num}.tsv", sep="\t")

    predicted = ["entailment" if "entailment" in label else 
                "contradiction" if "contradiction" in label else 
                "neutral" if "neutral" in label else
                "neutral" if "neu" in label else
                "#"+str(label) for label in list(annotated["llama3-8B-preds"])]

    annotated["llama3-8B-preds-clean"] = predicted

    annotated.to_csv(f"/home/oenni/Dokumente/NLI-Argumentation-project/annotation/NLI_machine-labeled/llama3-8B_predictions_primed_{num}_clean.tsv", sep="\t")

    for label in predicted:
        if label not in ["entailment", "contradiction", "neutral"]:
            print(label)

    print(num)
    predicted = annotated["llama3-8B-preds-clean"]
    gold = ["neutral" if label == "neu" else label for label in annotated["gold"]]

    accuracy = accuracy_score(gold, predicted)
    print("Accuracy:", accuracy)

    precision = precision_score(gold, predicted, average="macro")
    print("Precision:", precision)

    recall = recall_score(gold, predicted, average="macro")
    print("Recall:", recall)

    f1 = f1_score(gold, predicted, average="macro")
    print("F1-Score:", f1)

    cm = confusion_matrix(gold, predicted)
    print("confusion_matrix:", cm)

    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["contradiction", "entailment", "neutral"])
    cm_display.plot()
    print(set(gold))