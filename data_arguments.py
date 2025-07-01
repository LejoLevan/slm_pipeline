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
# - abstracted into it's own file
# - reinterpreted metadata information
# - modified extension list (added parquet as valid extension)
"""data training arguments class
"""
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class DataTrainingArguments:
    """class contains all relevant parameters related to what dataset is to be acted upon
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "name of dataset if using datasets library"}
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "config name of the dataset if using datasets library"}
    )

    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "path of training data file"}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "path of validation data file"}
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "path of test data file"}
    )

    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "whether or not to overwite cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "num processes"}
    )
    max_seq_length: int = field(
        default=384,
        metadata= {"help": "max total input seq length after tokenization. "
        "any longer will be truncated, shorter will be padded"}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={"help": "pads all samples to 'max_seq_length if true, otherwise pad dynamically"
        "during batching (faster on gpu)"}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "truncates num of training examples to value (debugging or quicker training)"}
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "truncates num of evaluation examples to value (debugging or quicker training)"}
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={"help": "truncates num of predication examples to value (debugging or quicker training)"}
    )
    version_2_with_negative: bool = field(
        default=False,
        metadata={"help": "there exists examples that do not have answers if true"}
    )
    null_score_diff_threshold: float = field(
        default=0.0,
        metadata={"help": "used in conjunction with 'version_2_with_negative=True'"
        "if best answer has score lower than the score fo null answer minus threeshold, "
        "null answer is selected"}
    )
    doc_stride: int = field(
        default=128,
        metadata={"help": "stride between chunks when chunking a long document"}
    )
    n_best_size: int = field(
        default=20,
        metadata={"help": "num of n-best answer predications"}
    )
    max_answer_length: int = field(
        default=30,
        metadata={"help": "max length of an generated answer"}
    )

    def __post_init__(self):
        if(
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
            and self.test_file is None
        ):
            raise ValueError("need either dataset name or dataset file paths")
        else:
            extension_list = ["csv", "json", "parquet"]
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in extension_list
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in extension_list
            if self.test_file is not None:
                extension = self.test_file.split(".")[-1]
                assert extension in extension_list