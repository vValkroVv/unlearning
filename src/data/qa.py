import torch
from torch.utils.data import Dataset
from typing import Any

from data.utils import load_hf_dataset, preprocess_chat_instance, add_dataset_index


class QADataset(Dataset):
    def __init__(
        self,
        hf_args,
        template_args,
        tokenizer,
        question_key="question",
        answer_key="answer",
        few_shot_dataset_hf_args=None,
        max_length=512,
        predict_with_generate=False,
    ):
        super(QADataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = load_hf_dataset(**hf_args)
        self.data = add_dataset_index(self.data)
        self.fs_data = None
        if few_shot_dataset_hf_args is not None:
            raw_data = load_hf_dataset(**few_shot_dataset_hf_args)
            self.fs_data = {}
            self.fs_data[question_key] = raw_data[question_key]
            self.fs_data[answer_key] = raw_data[answer_key]
        self.template_args = template_args
        self.question_key = question_key
        self.answer_key = answer_key
        self.predict_with_generate = predict_with_generate

    def __len__(self):
        return len(self.data)

    def _process_sample(self, question, answer, index=-1):
        if self.fs_data is None:
            prompt_msgs, response_msgs = [question], [answer]
        else:
            prompt_msgs = self.fs_data[self.question_key] + [question]
            response_msgs = self.fs_data[self.answer_key] + [answer]
        tokenized_data = preprocess_chat_instance(
            self.tokenizer,
            self.template_args,
            prompt_msgs,
            response_msgs,
            self.max_length,
            self.predict_with_generate,
        )
        item_dct = {
            "input_ids": tokenized_data["input_ids"],
            "labels": tokenized_data["labels"],
            "attention_mask": tokenized_data["attention_mask"],
            "index": index,
        }
        return item_dct

    def __getitem__(self, idx):
        row = self.data[idx]
        question = row[self.question_key]
        answer = row[self.answer_key]
        index = row["index"]
        pop_sum = row.get("pop_sum", None)
        if isinstance(answer, str):
            item = self._process_sample(question=question, answer=answer, index=index)
            if pop_sum is not None:
                item["pop_sum"] = pop_sum
        elif isinstance(answer, list):
            item = {}
            for i, ans in enumerate(answer):
                sample_item = self._process_sample(
                    question=question, answer=ans, index=index
                )
                if pop_sum is not None:
                    sample_item["pop_sum"] = pop_sum
                item[i] = sample_item
        else:
            raise NotImplementedError("answer format not found")
        return item


class QAwithIdkDataset(QADataset):
    def __init__(self, idk_path, return_original=True, *args, **kwargs):
        self.idk_path = idk_path
        self.return_original = return_original
        self.idk_responses = open(self.idk_path, "r").readlines()
        super().__init__(*args, **kwargs)

    def item_with_idk(self, question):
        rand_pos = torch.randint(0, len(self.idk_responses), (1,)).item()
        idk_response = self.idk_responses[rand_pos].strip()
        idk_item = self._process_sample(question=question, answer=idk_response)
        return idk_item

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        question = self.data[idx][self.question_key]
        if isinstance(item, dict):
            return_item = {"original": item}
            idk_item = self.item_with_idk(question)
            return_item["alternate"] = idk_item
            # return_item = [item, idk_item]
        elif isinstance(item, list) or isinstance(item, tuple):
            return_item = []
            for sample_item in item:
                return_item = {"original": sample_item}
                idk_item = self.item_with_idk(question)
                return_item["alternate"] = idk_item
                # return_item.append([sample_item, idk_item])
        return return_item if self.return_original else return_item["alternate"]


class QAwithAlternateDataset(QADataset):
    def __init__(self, alternate_key, return_original=True, *args, **kwargs):
        self.alternate_key = alternate_key
        self.return_original = return_original
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        question = self.data[idx][self.question_key]
        if isinstance(item, dict):
            return_item = {"original": item}
            alt_item = self._process_sample(
                question=question, answer=self.data[idx][self.alternate_key]
            )
            return_item["alternate"] = alt_item
            # return_item = [item, idk_item]
        elif isinstance(item, list) or isinstance(item, tuple):
            return_item = []
            for sample_item in item:
                return_item = {"original": sample_item}
                alt_item = self._process_sample(
                    question=question, answer=self.data[idx][self.alternate_key]
                )
                return_item["alternate"] = alt_item
                # return_item.append([sample_item, idk_item])
        return return_item if self.return_original else return_item["alternate"]


class QAwithAlternateMetadataDataset(QAwithAlternateDataset):
    def __init__(self, metadata_keys=None, *args, **kwargs):
        self.metadata_keys = list(metadata_keys or [])
        super().__init__(*args, **kwargs)
        if self.metadata_keys and not self.return_original:
            raise ValueError(
                "QAwithAlternateMetadataDataset requires return_original=True when "
                "metadata_keys are requested."
            )

    def _coerce_metadata_value(self, key, value):
        if hasattr(value, "item"):
            value = value.item()
        if key == "index":
            return int(value)
        return float(value)

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        if not self.metadata_keys:
            return item

        row = self.data[idx]
        for key in self.metadata_keys:
            if key not in row:
                raise KeyError(
                    f"Metadata key `{key}` is missing from dataset row {idx}."
                )
            item[key] = self._coerce_metadata_value(key=key, value=row[key])
        return item


class QAAnswerIndexDataset(QADataset):
    def __init__(self, answer_index=0, *args, **kwargs):
        self.answer_index = int(answer_index)
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        row = self.data[idx]
        question = row[self.question_key]
        answer = row[self.answer_key]
        index = row["index"]
        pop_sum = row.get("pop_sum", None)

        if isinstance(answer, list):
            if not answer:
                raise ValueError(f"Empty answer list at index {idx}")
            if self.answer_index >= len(answer) or self.answer_index < -len(answer):
                raise IndexError(
                    f"answer_index {self.answer_index} out of range for index {idx}"
                )
            answer = answer[self.answer_index]

        if not isinstance(answer, str):
            raise NotImplementedError("answer format not found")

        item = self._process_sample(question=question, answer=answer, index=index)
        if pop_sum is not None:
            item["pop_sum"] = pop_sum
        return item


class QAMultiCFDataset(QADataset):
    def __init__(
        self,
        alternate_key="alternate",
        alternate_set_key="alternate_set",
        alternate_weights_key="alternate_set_weights",
        metadata_keys=None,
        return_original=True,
        max_alternates=0,
        normalize_alternate_weights=True,
        *args,
        **kwargs,
    ):
        self.alternate_key = alternate_key
        self.alternate_set_key = alternate_set_key
        self.alternate_weights_key = alternate_weights_key
        self.metadata_keys = list(metadata_keys or [])
        self.return_original = bool(return_original)
        self.max_alternates = int(max_alternates)
        self.normalize_alternate_weights = bool(normalize_alternate_weights)
        super().__init__(*args, **kwargs)
        if not self.return_original:
            raise ValueError("QAMultiCFDataset requires return_original=True.")

    def _coerce_metadata_value(self, key: str, value: Any):
        if hasattr(value, "item"):
            value = value.item()
        if key == "index":
            return int(value)
        return float(value)

    def _gather_alternate_texts(self, row):
        candidates = []
        for key in (self.alternate_key, self.alternate_set_key):
            value = row.get(key)
            if isinstance(value, str):
                candidates.append(value.strip())
            elif isinstance(value, (list, tuple)):
                for alternate in value:
                    if isinstance(alternate, str):
                        candidates.append(alternate.strip())

        deduped = []
        seen = set()
        for candidate in candidates:
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            deduped.append(candidate)

        if self.max_alternates > 0:
            deduped = deduped[: self.max_alternates]
        if not deduped:
            raise ValueError(
                f"QAMultiCFDataset expected at least one alternate for index {row['index']}."
            )
        return deduped

    def _alternate_weights(self, row, alternate_count: int):
        raw_weights = row.get(self.alternate_weights_key)
        if not isinstance(raw_weights, (list, tuple)):
            weights = [1.0 for _ in range(alternate_count)]
        else:
            weights = [float(weight) for weight in raw_weights[:alternate_count]]
            if len(weights) < alternate_count:
                weights.extend([0.0] * (alternate_count - len(weights)))

        if self.normalize_alternate_weights:
            weight_sum = sum(weights)
            if weight_sum > 0.0:
                weights = [weight / weight_sum for weight in weights]
            else:
                weights = [1.0 / float(alternate_count) for _ in range(alternate_count)]
        return weights

    def __getitem__(self, idx):
        row = self.data[idx]
        question = row[self.question_key]
        answer = row[self.answer_key]
        index = int(row["index"])

        if not isinstance(answer, str):
            raise NotImplementedError("QAMultiCFDataset expects string answers.")

        alternates = self._gather_alternate_texts(row)
        item = {
            "original": self._process_sample(question=question, answer=answer, index=index),
            "alternates": [
                self._process_sample(question=question, answer=alternate, index=index)
                for alternate in alternates
            ],
            "alternate_mask": [1 for _ in alternates],
            "alternate_weights": self._alternate_weights(
                row=row,
                alternate_count=len(alternates),
            ),
        }

        for key in self.metadata_keys:
            if key not in row:
                raise KeyError(
                    f"Metadata key `{key}` is missing from dataset row {idx}."
                )
            item[key] = self._coerce_metadata_value(key=key, value=row[key])
        return item


class QABoundaryCFDataset(QAwithAlternateMetadataDataset):
    def __init__(
        self,
        local_retain_question_key="local_retain_question",
        local_retain_answer_key="local_retain_answer",
        local_retain_index_key="local_retain_index",
        *args,
        **kwargs,
    ):
        self.local_retain_question_key = local_retain_question_key
        self.local_retain_answer_key = local_retain_answer_key
        self.local_retain_index_key = local_retain_index_key
        super().__init__(*args, **kwargs)
        if not self.return_original:
            raise ValueError("QABoundaryCFDataset requires return_original=True.")

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        row = self.data[idx]
        if self.local_retain_question_key not in row:
            raise KeyError(
                f"BoundaryCF row {idx} is missing `{self.local_retain_question_key}`."
            )
        if self.local_retain_answer_key not in row:
            raise KeyError(
                f"BoundaryCF row {idx} is missing `{self.local_retain_answer_key}`."
            )

        retain_index = int(row.get(self.local_retain_index_key, -1))
        item["local_retain"] = self._process_sample(
            question=row[self.local_retain_question_key],
            answer=row[self.local_retain_answer_key],
            index=retain_index,
        )
        item["local_retain_index"] = retain_index
        return item
