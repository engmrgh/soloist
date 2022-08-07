import json
import sqlite3
from argparse import ArgumentParser
import logging
from itertools import chain
import warnings

import torch
from tqdm import tqdm
import torch.nn.functional as F
from ignite.metrics import Bleu
from transformers import (
    GPT2DoubleHeadsModel,
    GPT2Tokenizer,
)

from train import SPECIAL_TOKENS
from extras.metrics import Inform, Success, Combined
from extras.utils import create_input_ids, create_token_type_ids, segments_encoder


MAX_BELIEF_SIZE = 64
MAX_SEQUENCE_LENGTH = 512

logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


def index(arr, val):
    indices = (arr == val).nonzero().squeeze().numpy()
    try:
        return indices.item()
    except ValueError:
        return indices[0].item()


def top_filtering(logits, top_k=0., top_p=0.9, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[
            0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[...,
                                 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


def get_sequence(tokenizer, history, belief, kb, current_output):
    ind, eob, eokb = tokenizer.convert_tokens_to_ids(["=>", "<eob>", "<eokb>"])

    if belief and kb:
        history, belief, kb, _ = segments_encoder(
            tokenizer, history, belief, kb, reply=None)
        return (
            [list(chain(*history)) + [ind]] +
            [belief + [eob]] +
            [kb + [eokb]] +
            [current_output]
        )
    else:
        history, *_ = segments_encoder(tokenizer, history, belief=None, kb=None, reply=None)
        return (
            [list(chain(*history)) + [ind]] +
            [current_output]
        )


def sample_sequence(model, tokenizer, history, belief, kb, args):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)

    current_output = []
    for i in range(args.max_length):
        sequence = get_sequence(tokenizer, history, belief, kb, current_output)
        input_ids = create_input_ids(sequence)
        token_type_ids = create_token_type_ids(tokenizer, sequence)

        input_ids = input_ids.to(args.device)
        token_type_ids= token_type_ids.to(args.device)

        logits = model(input_ids, token_type_ids=token_type_ids)
        if isinstance(logits, tuple):  # for gpt2 and maybe others
            logits = logits[0]
        logits = logits[-1, :] / args.temperature
        logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
        probs = F.softmax(logits, dim=-1)

        prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
        if i < args.min_length and prev.item() in special_tokens_ids:
            while prev.item() in special_tokens_ids:
                if probs.max().item() == 1:
                    warnings.warn(
                        "Warning: model generating special token with probability 1.")
                    break  # avoid infinitely looping over special token
                prev = torch.multinomial(probs, num_samples=1)

        if prev.item() in special_tokens_ids:
            break
        current_output.append(prev.item())

    return current_output


def main():
    parser = ArgumentParser()
    parser.add_argument("--testset", type=str,
                        default="data/test.soloist.json", help="Dataset for evaulating model performance")
    parser.add_argument("--checkpoint", type=str,
                        default="", help="Path, url or short name of the model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available()
                        else "cuda", help="Device (cuda or cpu)")
    parser.add_argument("--max_length", type=int, default=50,
                        help="Maximum length of the output utterances")
    parser.add_argument("--min_length", type=int, default=1,
                        help="Minimum length of the output utterances")
    parser.add_argument("--no_sample", action='store_true',
                        help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    parser.add_argument("--temperature", type=float,
                        default=0.7, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0,
                        help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    parser.add_argument("--max_turn", type=int, default=5,
                        help="Maximum number of history turns fed to model")

    args = parser.parse_args()

    tokenizer = GPT2Tokenizer.from_pretrained(args.checkpoint)
    model = GPT2DoubleHeadsModel.from_pretrained(args.checkpoint)

    dbs, columns = connect_to_db()

    with open(args.testset, "r") as f:
        testset = json.load(f)

    model.to(args.device)
    model.eval()
    metrics = {"bleu": Bleu(), "inform": Inform(), "success": Success()}
    metrics.update({"combined": Combined(metrics)})
    with torch.no_grad():
        for ent in tqdm(testset, desc="Evaluating model"):
            # Round 1: generating belief
            belief = sample_sequence(model=model,
                                     tokenizer=tokenizer,
                                     history=ent["history"][-args.max_turn:],
                                     belief=None,
                                     kb=None,
                                     args=args)
            belief = tokenizer.decode(belief)
            kb = query_knowledge_base(dbs, columns, belief)

            # Round 2: generating response from belief
            reply = sample_sequence(model=model,
                                    tokenizer=tokenizer,
                                    history=ent["history"][-args.max_turn:],
                                    belief=belief,
                                    kb=kb,
                                    args=args)
            reply = tokenizer.decode(reply)

            for name, metric in metrics.items():
                metric.update(([reply.split()], [[ent["reply"].split()]]))

    for name, metric in metrics.items():
        print("{}: {:.4f}".format(name, metric.compute()))


if __name__ == "__main__":
    main()
