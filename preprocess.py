import json
from datasets import load_dataset
from transformers import AutoTokenizer

def config_load():
    with open("config.json", "r") as f:
        config = json.load(f)
    return config

def dataset_load(config):
    return load_dataset(
        config["data_type"],
        data_files=config["data_files"]
    )

def clean_questions(examples):
    examples["question"] = [entry.strip() for entry in examples["question"]]
    return examples

def clean_context_answer(examples):
    cleaned_context = [entry.strip() for entry in examples["context"]]
    cleaned_answer = [entry.strip() for entry in examples["answer"]]

    for i in range(len(examples["answer"])):
        cleaned_answer_start = cleaned_context.find(cleaned_answer["text"][i])
        cleaned_answer_end = cleaned_answer_start + len(examples["answer"][i])

        if cleaned_answer_start == -1:
            return examples
        
        if cleaned_answer["text"][i] != cleaned_context[cleaned_answer_start : cleaned_answer_end]:
            return examples
        
        cleaned_answer["answer_start"][i] = cleaned_answer_start
        cleaned_answer["answer_end"][i] = cleaned_answer_end
    
    examples["context"] = cleaned_context
    examples["answer"] = cleaned_answer

    return examples

def clean_examples(examples):
    examples = clean_questions(examples)
    examples = clean_context_answer(examples)
    return examples

def tokenize_examples(examples, config):
    tokenizer_path = config["path"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    tokenized_examples = tokenizer(
        examples["question" if config["pad_on_right"] else "context"],
        examples["context" if config["pad_on_right"] else "question"],
        truncation="only_second" if config["pad_on_right"] else "only_first",
        max_length=config["max_length"],
        stride=config["doc_stride"],
        return_overflowing_tokens=config["overflowing_tokens"],
        return_offsets_mapping=config["offsets_mapping"],
        padding="max_length"
    )

    return tokenizer, tokenized_examples

def label_examples(examples, tokenized_examples, tokenizer, config):
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        sequence_ids = tokenized_examples.sequence_ids(i)

        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]

        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if config["pad_on_right"] else 0):
                token_start_index += 1
            
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if config["pad_on_right"] else 0):
                token_end_index -= 1
            
            if not(offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples

def prepare_train_features(examples, config):

    clean_questions(examples)
    tokenizer, tokenized_examples = tokenize_examples(examples, config["tokenizer"])
    tokenized_examples = label_examples(examples, tokenized_examples, tokenizer, config["tokenizer"])
    return tokenized_examples

def preprocess():
    config = config_load()
    datasets = dataset_load(config["load_dataset"])

    tokenized_datasets = datasets.map(prepare_train_features, batched=True,
        remove_columns=datasets["train"].column_names, fn_kwargs={"config" : config})
    
    return tokenized_datasets

preprocess()