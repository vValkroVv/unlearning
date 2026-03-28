import torch
import transformers
import numbers
from typing import Dict, Sequence
from data.utils import IGNORE_INDEX


class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        padding_side: str = "right",
        index: str = None,
    ):
        self.tokenizer = tokenizer
        self.padding_side = padding_side
        self.index = index

    def get_instances_from_key(self, instances: Sequence[Dict], key: str):
        ret_instances = [instance[key] for instance in instances]
        return ret_instances

    def _pad_tokens(self, input_ids, padding_value):
        if self.padding_side == "right":
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=padding_value
            )
        else:
            input_ids = torch.nn.utils.rnn.pad_sequence(
                [torch.flip(i, dims=[0]) for i in input_ids],
                batch_first=True,
                padding_value=padding_value,
            ).flip(dims=[1])
        return input_ids

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        first_instance = instances[0]
        if torch.is_tensor(first_instance):
            return torch.stack(instances)
        if isinstance(first_instance, bool):
            return torch.tensor(instances, dtype=torch.bool)
        if isinstance(first_instance, numbers.Integral):
            return torch.tensor(instances, dtype=torch.long)
        if isinstance(first_instance, numbers.Real):
            return torch.tensor(instances, dtype=torch.float32)

        assert isinstance(first_instance, dict)
        return_dct = {}
        if "input_ids" not in first_instance:
            for key in first_instance.keys():
                key_instances = self.get_instances_from_key(
                    instances=instances, key=key
                )
                return_dct[key] = self(key_instances)
        else:
            input_ids = [instance["input_ids"] for instance in instances]
            input_ids = self._pad_tokens(input_ids, self.tokenizer.pad_token_id)
            attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
            return_dct.update({"input_ids": input_ids})
            return_dct.update({"attention_mask": attention_mask})
            if "labels" in first_instance:
                labels = [instance["labels"] for instance in instances]
                labels = self._pad_tokens(labels, IGNORE_INDEX)
                return_dct.update({"labels": labels})
            # Optional auxiliary numeric fields (e.g., pop_sum) passed through if present
            if "pop_sum" in first_instance:
                return_dct.update(
                    {
                        "pop_sum": torch.tensor(
                            [float(example["pop_sum"]) for example in instances],
                            dtype=torch.float32,
                        )
                    }
                )
            if self.index:
                if self.index in first_instance:
                    return_dct.update(
                        {
                            self.index: torch.tensor(
                                [example[self.index] for example in instances]
                            )
                        }
                    )
                else:
                    raise Warning(f"{self.index} not found in dataset")
        return return_dct


class DataCollatorForMultiCF(DataCollatorForSupervisedDataset):
    """Collate MultiCF batches with a variable number of alternates per sample."""

    def __init__(self, max_alternates=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_alternates = (
            None if max_alternates in (None, "", "null", "None") else int(max_alternates)
        )

    def _empty_alternate_sample(self) -> Dict[str, torch.Tensor]:
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id or 0
        return {
            "input_ids": torch.tensor([pad_token_id, pad_token_id], dtype=torch.long),
            "labels": torch.tensor([IGNORE_INDEX, IGNORE_INDEX], dtype=torch.long),
        }

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        first_instance = instances[0]
        if isinstance(first_instance, dict) and "alternates" in first_instance:
            return_dct = {}
            alternate_counts = [len(instance["alternates"]) for instance in instances]
            max_alternates = max(alternate_counts) if alternate_counts else 0
            if self.max_alternates is not None:
                max_alternates = min(max_alternates, self.max_alternates)
            if max_alternates <= 0:
                raise ValueError("DataCollatorForMultiCF expected at least one alternate.")

            for key in first_instance.keys():
                if key in {"alternates", "alternate_mask", "alternate_weights"}:
                    continue
                key_instances = self.get_instances_from_key(instances=instances, key=key)
                return_dct[key] = super().__call__(key_instances)

            alternate_batches = []
            empty_sample = self._empty_alternate_sample()
            for alt_idx in range(max_alternates):
                alt_instances = []
                for instance in instances:
                    alternates = instance["alternates"][:max_alternates]
                    if alt_idx < len(alternates):
                        alt_instances.append(alternates[alt_idx])
                    else:
                        alt_instances.append(empty_sample)
                alternate_batches.append(super().__call__(alt_instances))

            batch_size = len(instances)
            alternate_mask = torch.zeros((batch_size, max_alternates), dtype=torch.bool)
            alternate_weights = torch.zeros(
                (batch_size, max_alternates), dtype=torch.float32
            )
            for row_idx, instance in enumerate(instances):
                raw_mask = list(instance.get("alternate_mask", []))[:max_alternates]
                raw_weights = list(instance.get("alternate_weights", []))[:max_alternates]
                for alt_idx, value in enumerate(raw_mask):
                    alternate_mask[row_idx, alt_idx] = bool(value)
                for alt_idx, value in enumerate(raw_weights):
                    alternate_weights[row_idx, alt_idx] = float(value)

            return_dct["alternates"] = alternate_batches
            return_dct["alternate_mask"] = alternate_mask
            return_dct["alternate_weights"] = alternate_weights
            return return_dct

        return super().__call__(instances)
