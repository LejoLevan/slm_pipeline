# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Changes made by Jonathan Le (2025):
# - abstracted from run_qa.py into a respective class for the step in model training
# - maintained original comments as relevant

from trainer_qa import QuestionAnsweringTrainer

import transformers
from transformers import (
    DataCollatorWithPadding,
    default_data_collator,
)

class process:
    def __init__(
        self,

        model,
        tokenizer,

        training_args,
        data_args,

        preprocess_object,
        post_process_object,

        last_checkpoint,
    ):
        self.model = model
        self.tokenizer = tokenizer

        self.training_args = training_args
        self.data_args = data_args

        # [JL] preprocess created variables
        self.train_dataset = preprocess_object.get_preprocessed_train()

        self.eval_dataset = preprocess_object.get_preprocessed_eval()
        self.eval_examples = preprocess_object.get_eval_examples()
        #

        # [JL] postprocess created variables
        self.post_processing_function = post_process_object.post_processing_function
        self.compute_metrics = post_process_object.compute_metrics
        #

        self.last_checkpoint = last_checkpoint

        # Data collator
        # We have already padded to max length if the corresponding flag is True, otherwise we need to pad in the data
        # collator.
        self.data_collator = (
            default_data_collator
            if data_args.pad_to_max_length
            else DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)
        )

        self.trainer = None
    
    def init_trainer(self):
        # Initialize our Trainer
        self.trainer = QuestionAnsweringTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset if self.training_args.do_train else None,
            eval_dataset=self.eval_dataset if self.training_args.do_eval else None,
            eval_examples=self.eval_examples if self.training_args.do_eval else None,
            processing_class=self.tokenizer,
            data_collator=self.data_collator,
            post_process_function=self.post_processing_function,
            compute_metrics=self.compute_metrics,
        )
    
    def training(self):
        # Training
        checkpoint = None
        if self.training_args.resume_from_checkpoint is not None:
            checkpoint = self.training_args.resume_from_checkpoint
        elif self.last_checkpoint is not None:
            checkpoint = self.last_checkpoint
        train_result = self.trainer.train(resume_from_checkpoint=checkpoint)
        self.trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            self.data_args.max_train_samples if self.data_args.max_train_samples is not None else len(self.train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(self.train_dataset))

        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        self.trainer.save_state()
    
    def get_trainer(self):
        return self.trainer
