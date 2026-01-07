import datetime
import os

import argparse
import random

import yaml
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from data.utils import load_data
from experiments.mbert_fine_tuning import label2id
from experiments.utils import save_json, load_json

datetime = datetime.datetime


def get_parser():
    """Get the argument parser for the script."""
    parser = argparse.ArgumentParser(
        description=(
            "Run experiments to generate multilingual counterfactual examples'"
        )
    )

    parser.add_argument(
        "--hf_token",
        type=str,
        default=os.environ.get("HF_TOKEN_PATH", None),
        help=(
            "HuggingFace token for accessing models. Defaults to environment "
            "variable HF_TOKEN_PATH."
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/counterfactuals",
        help="Directory to save experiment results.",
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        help="Directory to save models.",
    )


    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
        help="Model to run the experiments",
    )

    parser.add_argument(
        "--language",
        type=str,
        default="de",
        help="Language for the experiments",
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="sib200",
        help="Dataset for the experiments",
    )

    return parser


def load_prediction(dataset_name, language):
    content = load_json(f"./results/predictions/{language}_{dataset_name}_predictions.json")
    return [i["prediction"] for i in content]


def get_prompt_template(first, second, prediction, language, dataset_name):
    prompt_template = ""
    language_dict = {
        "en": "English",
        "de": "German",
        "es": "Spanish",
        "ar": "Arabic",
        "sw": "swahili",
        "hi": "Hindi",
    }

    if dataset_name == "XNLI":
        # modify premise/first
        prediction = int(prediction)
        int2label = {
            0: "entailment",
            1: "neutral",
            2: "contradiction"
        }

        labels = ["neutral", "entailment", "contradiction"]
        rand = random.randint(0, 1)
        labels.remove(int2label[prediction])
        prediction = int2label[prediction]
        target_label = labels[rand]

        prompt_template = f"""
Given two sentences (premise and hypothesis) in {language_dict[language]} and their original relationship, determine whether they
entail, contradict, or are neutral to each other. Change the premise with minimal edits to achieve
the target relation from the original one and output the edited premise surrounding by <edit>[premise]</edit> in {language}. 
Do not make any unnecessary changes.

#####Begin Example####

Original relation: **entailment**
premise: A woman is talking to a man. 
hypothesis: Brown-haired woman talking to man with backpack.
Target relation: **neutral**

Step 1: Identify phrases, words in the premise leading to the entailment relation:
'man',
Step 2: Change these phrases, words to get neutral relation with minimal changes:
'man' to 'student'.
Step 3: replace the phrases, words from step 1 in the original text by the phrases, words, sentences
in step 2:

Edited premise: <edit>A woman is talking to a student.</edit>

#####End Example####

Request: Given two sentences (premise and hypothesis) in {language_dict[language]} and their original relationship, determine
whether they entail, contradict, or are neutral to each other. Change the premise with minimal edits
to achieve the neutral relation from the original one  and output the edited premise surrounding by 
<edit>[premise]</edit> in {language_dict[language]}. Do not make any unnecessary changes. Do not add anything else.

Original relation: **{prediction}**
premise: {first}
hypothesis: {second}
Target relation: **{target_label}**
Edited premise:
        """
    else:
        labels = ["science/technology", "travel", "politics", "sports", "health", "entertainment", "geography"]
        rand = random.randint(0, 5)
        prediction = {value: key for key, value in label2id.items()}[int(prediction)]

        labels.remove(prediction)
        target_label = labels[rand]

        prompt_template = f"""
Given a sentence in {language_dict[language]} classified as belonging to one of the topics: 
"science/technology", "travel", "politics", "sports", "health", "entertainment", "geography". Modify the 
sentence to change its topic to the specified target topic and output the edited sentence surrounding by 
<edit>[sentence]</edit> in {language_dict[language]}.
Do not make any unnecessary changes.

#####Begin Example####

Original topic: **sports**
Sentence: The athlete set a new record in the marathon.
Target topic: **health**

Step 1: Identify key phrases or words determining the original topic:
'athlete', 'record', 'marathon'.
Step 2: Modify these key phrases or words minimally to reflect the target topic (health):
'athlete' to 'patient', 'set a new record' to 'showed improvement', 'marathon' to 'rehabilitation'.
Step 3: Replace the identified words or phrases in the original sentence:

Edited sentence: <edit>The patient showed improvement in the rehabilitation.</edit>

#####End Example####

Request: Given a sentence in {language_dict[language]} classified as belonging to one of the topics: 
"science/technology", "travel", "politics", "sports", "health", "entertainment", "geography". Modify the 
sentence to change its topic to the specified target topic and output the edited sentence surrounding by 
<edit>[sentence]</edit> in {language_dict[language]}.
Do not make any unnecessary changes.

Original topic: **{prediction}**
Sentence: {first}
Target topic: **{target_label}**
Edited sentence:
        """

    return prompt_template, prediction, target_label


def counterfactual_generation(model, tokenizer, language, dataset_name, experiment_dir):
    first_list, second_list, label_list = load_data(dataset_name, language)
    predictions = load_prediction(dataset_name, language)

    results = []

    for idx in tqdm(range(len(label_list))):
        if dataset_name == "XNLI":
            prompt_template, prediction, target_label = get_prompt_template(first_list[idx][language], second_list[idx], predictions[idx], language, dataset_name)
        else:
            prompt_template, prediction, target_label = get_prompt_template(first_list[idx], None, predictions[idx], language, dataset_name)

        messages = [
            {"role": "system", "content": "You are an excellent assistant for text editing."},
            {"role": "user", "content": prompt_template}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=2048
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        print(response)

        if dataset_name == "XNLI":
            results.append({
                "idx": idx,
                "premise": first_list[idx][language],
                "hypothesis": second_list[idx],
                "counterfactual": response,
                "prediction": prediction,
                "target": target_label
            })
        else:
            results.append({
                "idx": idx,
                "text": first_list[idx],
                "counterfactual": response,
                "prediction": prediction,
                "target": target_label
            })

        save_json(
            results,
            os.path.join(
                experiment_dir,
                f"{language}_{dataset_name}_{model.name_or_path.split('/')[1]}.json"
            )
        )


def main(args) -> None:
    print(vars(args))

    output_dir = args.output_dir
    model_name_or_path = args.model_name
    hf_token = args.hf_token
    language = args.language
    dataset_name = args.dataset_name
    cache_dir = args.cache_dir

    now = datetime.now()
    dt_string = now.strftime("%y%m%d_%H%M%S")
    safe_model_name_or_path = model_name_or_path.split("/")[1]
    experiment_dir = os.path.join(output_dir, dt_string + "_" + safe_model_name_or_path)
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"Saving results to {experiment_dir}")

    # Save args to a yaml file
    with open(os.path.join(experiment_dir, "args.yaml"), "w") as f:
        yaml.dump(vars(args), f)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="balanced",
        trust_remote_code=True,
        revision="main",
        token=hf_token,
        cache_dir=cache_dir
    )

    counterfactual_generation(model, tokenizer, language, dataset_name, experiment_dir)


if __name__ == "__main__":
    main(get_parser().parse_args())