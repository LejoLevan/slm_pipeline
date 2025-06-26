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
# - abstracted out classes: ModelArguments, DataTrainingArguments
# - rewrote/deleted comments for preferred readability and understanding

import os
import sys
import logging
import warnings

from DataTrainingArguments import DataTrainingArguments
from ModelArguments import ModelArguments

import transformers
from transformers import (
    TrainingArguments,
    HfArgumentParser
)

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

if __name__ == "__main__":
    main()