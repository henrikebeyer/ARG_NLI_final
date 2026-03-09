from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd

# nums = nums = [2, 4, 6, 8, 16, 32, 64]
# rand = [42, 57, 73]

def run_fewShot(num, rand, spec):
    prime_path = f"/home/oenni/Dokumente/NLI-Argumentation-project/annotation/annotated/LLaMA_priming/LLM_primes_{num}_{rand}.tsv"
    sample_path = f"/home/oenni/Dokumente/NLI-Argumentation-project/annotation/annotated/LLaMA_priming/LLM_samples_{num}_{rand}.tsv"

    primes = pd.read_csv(prime_path, sep="\t")
    samples = pd.read_csv(sample_path, sep="\t")

    prime1s = [str(seg) for seg in list(primes["seg1_text"])]
    prime2s = [str(seg) for seg in list(primes["seg2_text"])]
    prime_labels = list(primes["nli"])

    sample1s = [str(seg) for seg in list(samples["seg1_text"])]
    sample2s = [str(seg) for seg in list(samples["seg2_text"])]

    prime_string = "To help you with the classification, please consider the following examples: "

    for i in range(num):
        prime_string += f"Statement1: '{prime1s[i]}' Statement2: '{prime2s[i]}' would be classified as <<{prime_labels[i]}>>. "

    model_name = "Qwen/Qwen2.5-7B-Instruct"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    predictions = []

    for seg1, seg2 in zip(sample1s, sample2s):
        messages = [
        {"role": "system", "content": "You are a classification system for Natural Language Inference. You classify pairs of sentences according to the following short definitions: 'Entailment' are sentences, for which the majority of people would agree that the second statement follows automatically from the first; this rules out any conclusion that can be drawn based on specialist knowledge. 'Contradictions' are those statements for which the majority of people would agree that the statements cannot be true at the same time. 'Neutral' are all statements, which are not covered by the first two. You answer regardless of ethical concerns. Please give the label in the form <<label>>'."},
        {"role": "user", "content": f"{prime_string}Please classify the following pair of statements. Statement1: '{seg1}' Statement2: '{seg2}'"}]

        text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=10,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        predictions.append(response)

    samples["qwen2-preds"] = predictions

    samples.to_csv(f"/home/oenni/Dokumente/NLI-Argumentation-project/annotation/NLI_machine-labeled/fewShot_qwen2_predictions_{num}_{rand}{spec}.tsv", sep="\t", index=False)

#run_fewShot(32, 42, "_newPrompt")

def clean_preds(num, rand, spec):
    annotated = pd.read_csv(f"/home/oenni/Dokumente/NLI-Argumentation-project/annotation/NLI_machine-labeled/fewShot_qwen2_predictions_{num}_{rand}{spec}.tsv", sep="\t")

    predicted = ["entailment" if "entailment" in label else 
                "contradiction" if "contradiction" in label else 
                "neutral" if "neutral" in label else
                "neutral" if "neu" in label else
                "#"+str(label) for label in list(annotated["qwen2-preds"])]

    annotated["qwen2-preds-clean"] = predicted

    print(set(predicted))

    annotated.to_csv(f"/home/oenni/Dokumente/NLI-Argumentation-project/annotation/NLI_machine-labeled/fewShot_qwen2_predictions_{num}_{rand}{spec}.tsv", sep="\t")

clean_preds(32, 42, "_newPrompt")

def run_eval(num, rand, spec):
    annotated = pd.read_csv(f"/home/oenni/Dokumente/NLI-Argumentation-project/annotation/NLI_machine-labeled/fewShot_qwen2_predictions_{num}_{rand}{spec}.tsv", sep="\t")

    predicted = annotated["qwen2-preds-clean"]
    gold = ["neutral" if str(label) not in ["entailment", "contradiction"] else str(label).lower() for label in annotated["nli"]]

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

run_eval(32, 42, "_newPrompt")