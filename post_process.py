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

import evaluate
from utils_qa import postprocess_qa_predictions

import transformers
from transformers import (
    EvalPrediction
)

class post_process:
    def __init__(
        self,

        logger,
        log_level,

        training_args,
        data_args,
        model_args,

        preprocess_object,
    ):
        self.logger = logger

        self.training_args = training_args
        self.data_args = data_args
        self.model_args = model_args

        self.eval_dataset = preprocess_object.get_preprocessed_eval()

        self.predict_dataset = preprocess_object.get_preprocessed_predict()
        self.predict_examples = preprocess_object.get_predict_examples()

        self.answer_column_name = preprocess_object.get_answer_column_name()

        self.log_level = log_level

        self.metric = evaluate.load(
            "squad_v2" if self.data_args.version_2_with_negative else "squad", cache_dir=self.model_args.cache_dir
        )

        self.trainer = None
    
    def post_processing_function(self, examples, features, predictions, stage="eval"):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=self.data_args.version_2_with_negative,
            n_best_size=self.data_args.n_best_size,
            max_answer_length=self.data_args.max_answer_length,
            null_score_diff_threshold=self.data_args.null_score_diff_threshold,
            output_dir=self.training_args.output_dir,
            log_level=self.log_level,
            prefix=stage,
        )
        # Format the result to the format the metric expects.
        if self.data_args.version_2_with_negative:
            formatted_predictions = [
                {"id": str(k), "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
            ]
        else:
            formatted_predictions = [{"id": str(k), "prediction_text": v} for k, v in predictions.items()]

        references = [{"id": str(ex["id"]), "answers": ex[self.answer_column_name]} for ex in examples]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    def evaluation(self):
        self.logger.info("*** Evaluate ***")
        metrics = self.trainer.evaluate()

        max_eval_samples = self.data_args.max_eval_samples if self.data_args.max_eval_samples is not None else len(self.eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(self.eval_dataset))

        self.trainer.log_metrics("eval", metrics)
        self.trainer.save_metrics("eval", metrics)

    def prediction(self):
        self.logger.info("*** Predict ***")
        results = self.trainer.predict(self.predict_dataset, self.predict_examples)
        metrics = results.metrics

        max_predict_samples = (
            self.data_args.max_predict_samples if self.data_args.max_predict_samples is not None else len(self.predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(self.predict_dataset))

        self.trainer.log_metrics("predict", metrics)
        self.trainer.save_metrics("predict", metrics)
    
    def compute_metrics(self, p: EvalPrediction):
        return self.metric.compute(predictions=p.predictions, references=p.label_ids)

    def set_trainer(self, trainer):
        self.trainer = trainer
