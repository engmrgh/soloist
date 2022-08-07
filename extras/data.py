import json
import os
import torch
import random
import logging
from tqdm import tqdm
from itertools import chain
from torch.utils.data import Dataset
from .utils import (
    segments_encoder,
    create_lm_labels,
    create_input_ids,
    create_token_type_ids,
)

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


MODEL_VECTOR_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]
ALL_MODEL_INPUTS = MODEL_VECTOR_INPUTS + ["mc_label", "mc_token_ids"]


class SoloistDataset(Dataset):
    def __init__(self, dataset_path, dataset_cache, tokenizer, max_seq_length, max_turns, n_fake_instances):
        logger.info("Loading dataset from {}...".format(dataset_path if dataset_path else dataset_cache))
        self._max_seq_length = max_seq_length
        self._max_turns = max_turns
        self._n_fake_instances = n_fake_instances
        self._dataset_cache = dataset_cache
        if not os.path.exists(dataset_cache):
            self.process_and_store(dataset_path, tokenizer, dataset_cache)

    def process_and_store(self, dataset_path, tokenizer, dataseet_cache):
        logger.info("Getting reply and belief pool...")
        reply_pool, belief_pool = self.get_all_replies_and_beliefs(
            dataset_path)

        logger.info("Building examples...")
        self.build_and_save_examples(
            tokenizer, dataset_path, dataseet_cache, reply_pool, belief_pool)

    def get_all_replies_and_beliefs(self, dataset_path):
        with open(dataset_path) as f:
            data = json.load(f)

            reply_pool, belief_pool = list(), list()
            for record in tqdm(data):
                belief, reply = record["belief"], record["reply"]
                reply_pool.append(reply)
                belief_pool.append(belief)

            random.shuffle(reply_pool)
            random.shuffle(belief_pool)

        return reply_pool, belief_pool

    def build_and_save_examples(self, tokenizer, dataset_path, dataset_cache, reply_pool, belief_pool):
        with open(dataset_path) as f, open(dataset_cache, "w") as cf:
            data = json.load(f)

            for record in tqdm(data):
                sample = list()

                # Real instance
                s = self._build_input_from_segments(
                        tokenizer,
                        record["history"][-self._max_turns:],
                        record["belief"],
                        record.get("kb", ""),
                        record["reply"],
                        fake=False,
                    )
                if len(s["input_ids"]) > self._max_seq_length:
                    continue
                sample.append(s)

                # Generating fake examples for contradiction loss
                fake_counter = 0
                while fake_counter < self._n_fake_instances:
                    false_belief = belief_pool[random.randrange(len(belief_pool))]
                    false_reply = reply_pool[random.randrange(len(reply_pool))]
                    s = self._build_input_from_segments(
                        tokenizer,
                        record["history"][-self._max_turns:],
                        false_belief,
                        record.get("kb", ""),
                        false_reply,
                        fake=True,
                    )
                    if len(s["input_ids"]) <= self._max_seq_length:
                        sample.append(s)
                        fake_counter += 1
                cf.write(json.dumps(sample) + "\n")

    def _build_input_from_segments(
        self, tokenizer, history, belief, kb, reply, fake=False
    ):
        history, belief, kb, reply = segments_encoder(
            tokenizer, history, belief, kb, reply
        )

        # NOTE: I didn't add bos token based on article
        ind, eob, eokb, eos = tokenizer.convert_tokens_to_ids(
            ["=>", "<eob>", "<eokb>", "<eos>"]
        )
        sequence = (
            [list(chain(*history)) + [ind]]  +
            [belief + [eob]] +
            [kb + [eokb]] +
            [reply + [eos]]
        )

        example = {}
        example["input_ids"] = create_input_ids(sequence)
        example["token_type_ids"] = create_token_type_ids(tokenizer, sequence)
        example["lm_labels"] = create_lm_labels(sequence, fake)
        example["mc_token_ids"] = len(example["input_ids"]) - 1

        return example

    def _extract_input(self, input_name, examples):
        return [example[input_name] for example in examples]

    def __len__(self):
        with open(self._dataset_cache, encoding='utf-8') as f:
            num_examples = sum(1 for line in f)
        return num_examples

    def __getitem__(self, index):
        REAL_INSTANCE_POSITION = 0

        with open(self._dataset_cache) as f:
            for _ in range(index):
                next(f)

            _examples = json.loads(next(f))
            for ex in _examples:
                ex['input_ids'] = torch.tensor(ex['input_ids'])
                ex['token_type_ids'] = torch.tensor(ex['token_type_ids'])
                ex['lm_labels'] = torch.tensor(ex['lm_labels'])

            example = {input_name: list()
                    for input_name in MODEL_VECTOR_INPUTS + ["mc_token_ids"]}
            random_index = list(range(len(_examples)))
            random.shuffle(random_index)

            for input_name in MODEL_VECTOR_INPUTS + ["mc_token_ids"]:
                inp = self._extract_input(input_name, _examples)
                for ri in random_index:
                    example[input_name].append(inp[ri])
            example["mc_label"] = random_index.index(REAL_INSTANCE_POSITION)

        return example


def collate_fn(batch, pad_token_id, max_seq_len, n_fake_instances):

    proc_batch = {input_name: list() for input_name in ALL_MODEL_INPUTS}

    for input_name in MODEL_VECTOR_INPUTS:
        for example in batch:
            proc_batch[input_name].extend(example[input_name])
    proc_batch["mc_token_ids"] = torch.as_tensor(
        [example["mc_token_ids"] for example in batch])
    proc_batch["mc_label"] = torch.as_tensor(
        [example["mc_label"] for example in batch])

    attention_mask = list()
    for idx, vector in enumerate(proc_batch["input_ids"]):
        attention = [0] * max_seq_len
        if len(vector) < max_seq_len:
            attention[-(max_seq_len - len(vector)):] = [1] * (max_seq_len - len(vector))
        attention_mask.append(attention)
    attention_mask = torch.as_tensor(attention_mask)
    attention_mask = attention_mask.view((-1, (n_fake_instances + 1), max_seq_len))

    for input_name in MODEL_VECTOR_INPUTS:
        for idx, vector in enumerate(proc_batch[input_name]):
            padding = torch.tensor(
                [pad_token_id if input_name != "lm_labels" else -100]
                * (max_seq_len - len(vector))
            )
            proc_batch[input_name][idx] = torch.cat((vector, padding))

        proc_batch[input_name] = torch.stack(proc_batch[input_name])
        proc_batch[input_name] = proc_batch[input_name].view(
            (-1, (n_fake_instances + 1), max_seq_len))

    proc_batch["mc_token_ids"] = proc_batch["mc_token_ids"].view(
        (-1, (n_fake_instances + 1)))

    return (
        proc_batch["input_ids"].to(torch.int64),
        proc_batch["mc_token_ids"].to(torch.int64),
        proc_batch["lm_labels"].to(torch.int64),
        proc_batch["mc_label"].to(torch.int64),
        proc_batch["token_type_ids"].to(torch.int64),
        attention_mask.to(torch.int64)
    )
