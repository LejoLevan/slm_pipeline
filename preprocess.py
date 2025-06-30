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

class preprocess:
    def __init__(
        self,

        logger,

        training_args,
        data_args,

        raw_datasets,
        tokenizer,     
    ):
        self.logger = logger


        """inputs"""
        self.training_args = training_args
        self.data_args = data_args

        self.raw_datasets = raw_datasets
        self.tokenizer = tokenizer

        """outputs"""
        # Preprocessing the datasets.
        # Preprocessing is slightly different for training and evaluation.
        if training_args.do_train:
            self.column_names = raw_datasets["train"].column_names
        elif training_args.do_eval:
            self.column_names = raw_datasets["validation"].column_names
        else:
            self.column_names = raw_datasets["test"].column_names
        self.question_column_name = "question" if "question" in self.column_names else self.column_names[0]
        self.context_column_name = "context" if "context" in self.column_names else self.column_names[1]
        self.answer_column_name = "answers" if "answers" in self.column_names else self.column_names[2]

        # Padding side determines if we do (question|context) or (context|question).
        self.pad_on_right = self.tokenizer.padding_side == "right"

        if data_args.max_seq_length > self.tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the "
                f"model ({self.tokenizer.model_max_length}). Using max_seq_length={self.tokenizer.model_max_length}."
            )
        self.max_seq_length = min(data_args.max_seq_length, self.tokenizer.model_max_length)

        self.train_dataset = None

        self.eval_dataset = None
        self.eval_examples = None

        self.predict_dataset = None
        self.predict_examples = None

        self.training_preprocess()
        self.validation_test_preprocess()
    
    def _prepare_train_features(self, examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[self.question_column_name] = [q.lstrip() for q in examples[self.question_column_name]]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = self.tokenizer(
            examples[self.question_column_name if self.pad_on_right else self.context_column_name],
            examples[self.context_column_name if self.pad_on_right else self.question_column_name],
            truncation="only_second" if self.pad_on_right else "only_first",
            max_length=self.max_seq_length,
            stride=self.data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if self.data_args.pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            if self.tokenizer.cls_token_id in input_ids:
                cls_index = input_ids.index(self.tokenizer.cls_token_id)
            elif self.tokenizer.bos_token_id in input_ids:
                cls_index = input_ids.index(self.tokenizer.bos_token_id)
            else:
                cls_index = 0

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples[self.answer_column_name][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if self.pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if self.pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    def training_preprocess(self):
        if "train" not in self.raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        self.train_dataset = self.raw_datasets["train"]
        if self.data_args.max_train_samples is not None:
            # We will select sample from whole data if argument is specified
            max_train_samples = min(len(self.train_dataset), self.data_args.max_train_samples)
            self.train_dataset = self.train_dataset.select(range(max_train_samples))
        # Create train feature from dataset
        with self.training_args.main_process_first(desc="train dataset map pre-processing"):
            self.train_dataset = self.train_dataset.map(
                self.prepare_train_features,
                batched=True,
                num_proc=self.data_args.preprocessing_num_workers,
                remove_columns=self.column_names,
                load_from_cache_file=not self.data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
        if self.data_args.max_train_samples is not None:
            # Number of samples might increase during Feature Creation, We select only specified max samples
            max_train_samples = min(len(self.train_dataset), self.data_args.max_train_samples)
            self.train_dataset = self.train_dataset.select(range(max_train_samples))

    def _prepare_validation_predict_features(self, examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[self.question_column_name] = [q.lstrip() for q in examples[self.question_column_name]]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = self.tokenizer(
            examples[self.question_column_name if self.pad_on_right else self.context_column_name],
            examples[self.context_column_name if self.pad_on_right else self.question_column_name],
            truncation="only_second" if self.pad_on_right else "only_first",
            max_length=self.max_seq_length,
            stride=self.data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if self.data_args.pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if self.pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    def validation_preprocess(self):
        if "validation" not in self.raw_datasets:
                raise ValueError("--do_eval requires a validation dataset")
        self.eval_examples = self.raw_datasets["validation"]
        if self.data_args.max_eval_samples is not None:
            # We will select sample from whole data
            max_eval_samples = min(len(self.eval_examples), self.data_args.max_eval_samples)
            self.eval_examples = self.eval_examples.select(range(max_eval_samples))
        # Validation Feature Creation
        with self.training_args.main_process_first(desc="validation dataset map pre-processing"):
            self.eval_dataset = self.eval_examples.map(
                self.prepare_validation_features,
                batched=True,
                num_proc=self.data_args.preprocessing_num_workers,
                remove_columns=self.column_names,
                load_from_cache_file=not self.data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )
        if self.data_args.max_eval_samples is not None:
            # During Feature creation dataset samples might increase, we will select required samples again
            max_eval_samples = min(len(self.eval_dataset), self.data_args.max_eval_samples)
            self.eval_dataset = self.eval_dataset.select(range(max_eval_samples))

    def predict_preprocess(self):
        if "test" not in self.raw_datasets:
                raise ValueError("--do_predict requires a test dataset")
        self.predict_examples = self.raw_datasets["test"]
        if self.data_args.max_predict_samples is not None:
            # We will select sample from whole data
            self.predict_examples = self.predict_examples.select(range(self.data_args.max_predict_samples))
        # Predict Feature Creation
        with self.training_args.main_process_first(desc="prediction dataset map pre-processing"):
            self.predict_dataset = self.predict_examples.map(
                self.prepare_validation_features,
                batched=True,
                num_proc=self.data_args.preprocessing_num_workers,
                remove_columns=self.column_names,
                load_from_cache_file=not self.data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )
        if self.data_args.max_predict_samples is not None:
            # During Feature creation dataset samples might increase, we will select required samples again
            max_predict_samples = min(len(self.predict_dataset), self.data_args.max_predict_samples)
            self.predict_dataset = self.predict_dataset.select(range(max_predict_samples))

    def get_preprocessed_train(self):
        return self.train_dataset
    def get_preprocessed_eval(self):
        return self.eval_dataset
    def get_preprocessed_predict(self):
        return self.predict_dataset
    
    def get_predict_examples(self):
        return self.predict_examples
    def get_eval_examples(self):
        return self.eval_examples()
    
    def get_answer_column_name(self):
        return self.answer_column_name
    
    #define rest of getter methods later