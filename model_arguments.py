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
# - reinterpretated metadeta information
"""model arguments class
"""
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelArguments:
    """class contains all relevant parameters related to what model is to be acted upon
    """

    model_name_or_path: str = field(
        metadata={"help": "pretrained model system path or model identifer from huggingface"}
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "pretrained config path if differs from model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "pretrained tokenizer path if differs from model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "where pretrained models from huggingface are stored"}
    )