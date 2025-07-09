# Overview

This is a pipeline to be used in producing **domain specific extractive QA SLMs** trained off datasets that follow the **SQuAD dataset format**. 
It is abstracted into a **main** funciton that uses three distinct classes: **preprocess, process, and postprocess**.

Besides the relevant pipeline code, the repository contains files relevant for demonstration. It contains a base bert model along with a
trained version of the model that was trained using the pipeline and datasets generated from mock_dataset.py. The file QA_predict.py compares the predictions
of the two models to demonstrate the effectievness of training. 

There is also a requirements.txt that can be used to easily download relevant imports to python virtual enviroment.

## Main
The main function is named run_QA.py and is the master function that use the aforementioned classes. It begins by using HuggingFace's API to 
parse through relevant arguments that can either be passed on the command line or through a json file. Model Arugments and Data Aruguments are classes 
abstracted from the licensed version of run_QA.py for readability purposes, Training Arugments parameters can be found here https://huggingface.co/docs/transformers/en/main_classes/trainer.
LoRA Arguments is a inserted class of arguments that contain relevant paramters for LoRA use in training.

It then checks for checkpoints within the output directory if the script was paused during training, if checkpoint exists the script will resume where it left up later.

Afterward the script loads up relevant datasets depending on Data Arguments (train, validation, and test). It then uses Model Arguments to load model itself, it's config, and it's tokenizer.
Depending on whether LoRA is requested to be used in training, the script will decide whether to load a Peft Model from the base model. There is also a check to see whether or not the tokenizer is classified as "fast",
which most relevant models will contain. 

There is also logic for logging defined in the main function.

### Preprocess
Preprocessing is the stage that focuses on preparing the dataset for training. Here data is briefly cleaned before then being ran through the tokenizer to produce examples that are then 
labeled in relation to where the given answer is in within the given context. The dataset must be tokenzied as the model understands tokens rather than raw data when training.

Datasets must follow SQuAD format where an entry may look like:
{
  "id": 0,
  "context": "placeholder"
  "question": "What is the first word in the context?"
  "answers": {
    "text": [
      "placeholder",
    ],
    "answer_start": [
      0
    ]
  },
  "is_impossible": false
}

Data Arugments determine whether or not train, validation, and/or test examples are generated.

Basic getter methods are also defined.

### Process
Process is the stage where the pipeline trains/finetunes the model on the generated exampls from Preprocess, using Training Arguments to initialize the trainer. It uses HuggingFace's API
to train from scratch or checkpoint if it was previously detected. It also, while training, will output log information onto the console depending Training Arguments that can be used to
evaluate how well training is going. The class uses methods defined in Post Process to achieve this evaulation/metric logging.

If such logging/metrics are enabled, pay attention to trends on 'loss' and 'eval_f1', where the former should ideally be decreasing and the latter should ideally be
increasing as training occurs.

Training trains the model on getting used to identifying specific patterns/language from the dataset so that it can make predictions when given similar data inputs in the future. 

### Post Process
Post Process is the stage where evaluation and prediciton of the now trained/finetuned model occurs. The information outputted from this stage can be uesd by the user to determine
whether or not to rerun the pipeline for a better trained/fintuned model. 'eval_f1' should be relatively high if the model was suffiecntly trained, if not, user should consider modifying
TrainingArguments such as 'epoch' and 'learning_rate' to increase the training time. The metric could also be highlighting an issue with the dataset used as well.

## Limitations/Improvements
As previously mentioned the pipeline is limited to extractive QA training and datasets that follow SQuAD formatting. The latter can be resolved in future iteration through script that
recieves input from the user on what fields from the incompatible dataset correspond to the expected SQuAD fields (though may still require some manual data preperation from user). The former
is not so much an issue as the pipeline is focused on SLMs whose main advantage over LLMs are in QA tasks on specific domains.

## Third-Party Code Attribution

This project includes code adapted from the Hugging Face Transformers library
(https://github.com/huggingface/transformers), licensed under the Apache License 2.0.

Major changes were made to example Question Answering scritps:
- abstracted classes and functions to respective files
- implemented LoRA in training logic

See LICENSE for full license text.
