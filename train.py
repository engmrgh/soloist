import os
import math
import logging
import socket
from pprint import pformat
from datetime import datetime
from pathlib import Path
from argparse import ArgumentParser
from functools import partial
from itertools import chain

import torch
from torch.utils.data import DataLoader
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, global_step_from_engine
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler
from transformers import (AdamW, OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer,
                          GPT2DoubleHeadsModel, GPT2Tokenizer, WEIGHTS_NAME, CONFIG_NAME)

from extras.data import SoloistDataset, collate_fn
from extras.utils import history_encoder, belief_encoder, kb_encoder


SPECIAL_TOKENS = ["<bos>", "<eos>", "=>",
                  "<eob>", "<db>", "<eokb>",
                  "<user>", "<system>", "<pad>"]
ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>',
                         'additional_special_tokens': ['=>', '<belief>', '<eob>', '<db>',
                                                       '<eokb>', "<user>", "<system>",
                                                       "[attraction_address]",
                                                       "[attraction_area]",
                                                       "[attraction_name]",
                                                       "[attraction_phone]",
                                                       "[attraction_postcode]",
                                                       "[attraction_pricerange]",
                                                       "[attraction_reference]",
                                                       "[hospital_address]",
                                                       "[hospital_department]",
                                                       "[hospital_name]",
                                                       "[hospital_phone]",
                                                       "[hospital_postcode]",
                                                       "[hospital_reference]",
                                                       "[hotel_address]",
                                                       "[hotel_area]",
                                                       "[hotel_name]",
                                                       "[hotel_phone]",
                                                       "[hotel_postcode]",
                                                       "[hotel_pricerange]",
                                                       "[hotel_reference]",
                                                       "[police_address]",
                                                       "[police_name]",
                                                       "[police_phone]",
                                                       "[police_postcode]",
                                                       "[restaurant_address]",
                                                       "[restaurant_area]",
                                                       "[restaurant_food]",
                                                       "[restaurant_name]",
                                                       "[restaurant_phone]",
                                                       "[restaurant_postcode]",
                                                       "[restaurant_pricerange]",
                                                       "[restaurant_reference]",
                                                       "[taxi_phone]",
                                                       "[taxi_type]",
                                                       "[train_price]",
                                                       "[train_reference]",
                                                       "[train_trainid]",
                                                       "[value_count]",
                                                       "[value_day]",
                                                       "[value_place]",
                                                       "[value_time]",
                                                       "[value_pricerange]"]}


logger = logging.getLogger(__file__)


def average_distributed_scalar(scalar, args):
    """ Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. """
    return scalar


def add_special_tokens(model, tokenizer):
    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    orig_num_tokens = len(tokenizer.encoder)
    num_added_tokens = tokenizer.add_special_tokens(
        ATTR_TO_SPECIAL_TOKEN)  # doesn't add if they are already there
    if num_added_tokens > 0:
        model.resize_token_embeddings(
            new_num_tokens=orig_num_tokens + num_added_tokens)


def make_logdir(model_name: str):
    """Create unique path to save results and checkpoints, e.g. runs/Sep22_19-45-59_gpu-7_gpt2"""
    # Code copied from ignite repo
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    logdir = os.path.join(
        'runs', current_time + '_' + socket.gethostname() + '_' + model_name)
    return logdir


def get_data_loaders(args, tokenizer):
    # Load train dataset
    if args.train_cache and os.path.exists(args.train_cache):
        logger.info("Loading train dataset from cache {}".format(args.train_cache))
        train_dataset = torch.load(args.train_cache)
    else:
        logger.info("Loading train dataset from file {}".format(args.train_dataset))
        train_dataset = \
            SoloistDataset(args.train_dataset,
                            tokenizer,
                            max_seq_length=args.max_seq_length,
                            max_turns=args.max_turns,
                            n_fake_instances=args.n_fake_instances)

        if args.train_cache:
            p = Path(args.train_cache)
            p.parent.mkdir(exist_ok=True, parents=True)
            torch.save(train_dataset, args.train_cache)

    # Load validation dataset
    valid_dataset = None
    if args.val_cache and os.path.exists(args.val_cache):
        logger.info("Loading validation dataset from cache {}".format(args.val_cache))
        valid_dataset = torch.load(args.val_cache)
    elif args.val_dataset:
        logger.info("Loading validation dataset from file {}".format(args.val_dataset))
        valid_dataset = \
            SoloistDataset(args.val_dataset,
                            tokenizer,
                            max_seq_length=args.max_seq_length,
                            max_turns=args.max_turns,
                            n_fake_instances=args.n_fake_instances)

        if args.val_cache:
            p = Path(args.val_cache)
            p.parent.mkdir(exist_ok=True, parents=True)
            torch.save(valid_dataset, args.val_cache)
    else:
        train_dataset_size = int(0.9 * len(train_dataset))
        val_dataset_size = len(train_dataset) - train_dataset_size
        train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_dataset_size, val_dataset_size])

    train_loader = \
        DataLoader(train_dataset,
                batch_size=args.train_batch_size,
                collate_fn=partial(collate_fn,
                                    pad_token_id=tokenizer.pad_token_id,
                                    max_seq_len=args.max_seq_length,
                                    n_fake_instances=args.n_fake_instances),
                shuffle=True)

    valid_loader = \
        DataLoader(valid_dataset,
                batch_size=args.valid_batch_size,
                collate_fn=partial(collate_fn,
                                    pad_token_id=tokenizer.pad_token_id,
                                    max_seq_len=args.max_seq_length,
                                    n_fake_instances=args.n_fake_instances),
                shuffle=False)

    return train_loader, valid_loader


def test_model(model, tokenizer, args):
    model.eval()
    with torch.no_grad():
        ind, eob, eokb = tokenizer.convert_tokens_to_ids(
            ["=>", "<eob>", "<eokb>"]
        )
        history = \
            history_encoder(
                tokenizer,
                ["user : hi i'm looking for lodging in cambridge that includes free wifi and is upscale and expensive",
                "system : i found [hotel_name] matching your requirement -s . would you like to stay there ?",
                "user : i actually am looking for a guesthouse, not a hotel.",
                "system : unfortunately , there is nothing that meets your criteria . is there anything else i can help you with ?",
                "user : is there a guesthouse that might be in the cheaper price range in the same area?"])
        sequence = (
            [list(chain(*history)) + [ind]]
        )
        sequence_tensor = torch.as_tensor(list(chain(*sequence))).unsqueeze(dim=0).to(args.device)
        model = model.to(args.device)

        (generated,) = model.generate(
            sequence_tensor,
            max_length=sequence_tensor.shape[1] + 30,
        )

        print("belief:", tokenizer.decode(generated[sequence_tensor.shape[1]:]))

        history = history_encoder(
            tokenizer,
            ["user : hi i'm looking for lodging in cambridge that includes free wifi and is upscale and expensive",
            "system : i found [hotel_name] matching your requirement -s . would you like to stay there ?",
            "user : i actually am looking for a guesthouse, not a hotel.",
            "system : unfortunately , there is nothing that meets your criteria . is there anything else i can help you with ?",
            "user : is there a guesthouse that might be in the cheaper price range in the same area?"])
        belief = belief_encoder(tokenizer, "belief : hotel internet = yes ; type = guesthouse")
        kb = kb_encoder(tokenizer, "kb : hotel more than five")

        sequence = [list(chain(*history))] + [[ind] + belief + [eob]] + [kb + [eokb]]
        sequence_tensor = torch.as_tensor(list(chain(*sequence))).unsqueeze(dim=0).to(args.device)

        (generated,) = model.generate(
            sequence_tensor,
            max_length=args.max_seq_length
        )
        print("response:", tokenizer.decode(generated[sequence_tensor.shape[1]:]))


def train():
    parser = ArgumentParser()
    parser.add_argument("--train_dataset", type=str, default="",
                        help="Path of the training dataset")
    parser.add_argument('--val_dataset', type=str, default="",
                        help="Path of the validation dataset.")
    parser.add_argument("--train_cache", type=str, default="",
                        help="Path of the train dataset cache")
    parser.add_argument("--val_cache", type=str, default="",
                        help="Path of the validation dataset cache")
    parser.add_argument("--pretrained_model_path", type=str,
                        default="gpt2", help="Pretrained model name or ")
    parser.add_argument("--model_checkpoint", type=str,
                        default="", help="Path, url or short name of the model")
    parser.add_argument("--train_batch_size", type=int,
                        default=2, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int,
                        default=2, help="Batch size for validation")
    parser.add_argument("--gradient_accumulation_steps", type=int,
                        default=8, help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float,
                        default=6.25e-5, help="Learning rate")
    parser.add_argument("--max_norm", type=float,
                        default=1.0, help="Clipping gradient norm")
    parser.add_argument("--n_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--eval_before_start", action='store_true',
                        help="If true start with a first evaluation before training")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="Max sequence which all sequences will be padded")
    parser.add_argument("--max_turns", type=int, default=5,
                        help="Max history turns fed to model")
    parser.add_argument("--n_fake_instances", type=int, default=10,
                        help="Number of generated fake instances to train reply, belief contradiction")
    parser.add_argument("--fp16", type=str, default="",
                        help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)")
    args = parser.parse_args()

    # logging is set to INFO (resp. WARN) for main (resp. auxiliary) process. logger.info => log main process only,
    # logger.warning => log all processes
    logging.basicConfig(level=logging.INFO)
    logger.info("Arguments: %s", pformat(args))

    logger.info("Prepare tokenizer, pretrained model and optimizer.")
    # cant use Autotokenizer because checkpoint could be a Path
    tokenizer_class = GPT2Tokenizer if "gpt2" in args.pretrained_model_path else OpenAIGPTTokenizer
    tokenizer = tokenizer_class.from_pretrained(args.pretrained_model_path)

    model_class = GPT2DoubleHeadsModel if "gpt2" in args.pretrained_model_path else OpenAIGPTDoubleHeadsModel
    model = model_class.from_pretrained(args.pretrained_model_path)
    model.to(args.device)
    # Add special tokens if they are not already added
    add_special_tokens(model, tokenizer)
    optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=True)

     # Prepare model for FP16 and distributed training if needed (order is important, distributed should be the last)
    if args.fp16:
        from apex import amp  # Apex is only required if we use fp16 training
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16)

    logger.info("Prepare datasets")
    train_loader, val_loader = get_data_loaders(args, tokenizer)

    # Training function and trainer
    def update(engine, batch):
        model.train()
        batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
        input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids, attention_mask = batch
        (lm_loss), (mc_loss), *_ = model(
            input_ids=input_ids, token_type_ids=token_type_ids, mc_token_ids=mc_token_ids,
            mc_labels=mc_labels, lm_labels=lm_labels
        )
        loss = (lm_loss + mc_loss) / args.gradient_accumulation_steps
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        if engine.state.iteration % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        return loss.item()
    trainer = Engine(update)

    # Evaluation function and evaluator (evaluator output is the input of the metrics)
    def inference(engine, batch):
        model.eval()
        with torch.no_grad():
            batch = tuple(input_tensor.to(args.device)
                          for input_tensor in batch)
            input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids, attention_mask = batch
            logger.info(tokenizer.decode(input_ids[0, -1, :].tolist()))
            # if we dont send labels to model, it doesnt return losses
            lm_logits, mc_logits, *_ = model(
                input_ids, token_type_ids=token_type_ids, mc_token_ids=mc_token_ids,
            )
            lm_logits_flat_shifted = lm_logits[..., :-1,
                                               :].contiguous().view(-1, lm_logits.size(-1))
            lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)
            return (lm_logits_flat_shifted, mc_logits), (lm_labels_flat_shifted, mc_labels)

    evaluator = Engine(inference)

    # Attach evaluation to trainer: we evaluate when we start the training and at the end of each epoch
    trainer.add_event_handler(Events.EPOCH_COMPLETED,
                                lambda _: evaluator.run(val_loader))
    trainer.add_event_handler(Events.ITERATION_STARTED(every=50),
                                lambda _: test_model(model, tokenizer, args))
    if args.n_epochs < 1:
        trainer.add_event_handler(
            Events.COMPLETED, lambda _: evaluator.run(val_loader))
    if args.eval_before_start:
        trainer.add_event_handler(
            Events.STARTED, lambda _: evaluator.run(val_loader))

    # Linearly decrease the learning rate from lr to zero
    scheduler = PiecewiseLinear(
        optimizer, "lr", [(0, args.lr), (args.n_epochs * len(train_loader), 0.0)])
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    # Prepare metrics
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
    metrics = {"nll": Loss(torch.nn.CrossEntropyLoss(ignore_index=-100), output_transform=lambda x: (x[0][0], x[1][0])),
            "accuracy": Accuracy(output_transform=lambda x: (x[0][1], x[1][1]))}
    metrics.update({"average_nll": MetricsLambda(average_distributed_scalar, metrics["nll"], args),
                    "average_accuracy": MetricsLambda(average_distributed_scalar, metrics["accuracy"], args)})
    metrics["average_ppl"] = MetricsLambda(math.exp, metrics["average_nll"])
    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    # On the main process: add progress bar, tensorboard, checkpoints and save model, configuration and tokenizer
    # before we start to train
    pbar = ProgressBar(persist=True)
    pbar.attach(trainer, metric_names=["loss"])

    evaluator.add_event_handler(Events.COMPLETED, lambda _: pbar.log_message(
        "Validation: %s" % pformat(evaluator.state.metrics)))

    log_dir = make_logdir(args.pretrained_model_path)
    tb_logger = TensorboardLogger(log_dir)

    tb_logger.attach(trainer, log_handler=OutputHandler(
        tag="training", metric_names=["loss"]), event_name=Events.ITERATION_COMPLETED)
    tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(
        optimizer), event_name=Events.ITERATION_STARTED)
    tb_logger.attach(evaluator, log_handler=OutputHandler(tag="validation", metric_names=list(
        metrics.keys()), global_step_transform=global_step_from_engine(trainer)), event_name=Events.EPOCH_COMPLETED)

    checkpoint_handler = ModelCheckpoint(
        log_dir, 'checkpoint', save_interval=1, n_saved=3)
    to_load = to_save = {'mymodel': getattr(model, 'module', model)}
    trainer.add_event_handler(Events.ITERATION_STARTED(every=100), checkpoint_handler, to_save)  # "getattr" takes care of distributed encapsulation

    if args.model_checkpoint:
        logger.info('Loading checkpoint from %s', args.model_checkpoint)
        checkpoint_handler.load_objects(checkpoint=Path(args.model_checkpoint), to_load=to_load)

    torch.save(args, log_dir + '/model_training_args.bin')
    getattr(model, 'module', model).config.to_json_file(
        os.path.join(log_dir, CONFIG_NAME))
    tokenizer.save_pretrained(log_dir)

    # Run the training
    trainer.run(train_loader, max_epochs=args.n_epochs)

    # On the main process: close tensorboard logger and rename the last checkpoint
    # (for easy re-loading with OpenAIGPTModel.from_pretrained method)
    if args.n_epochs > 0:
        # TODO: PR in ignite to have better access to saved file paths (cleaner)
        os.rename(os.path.join(
            log_dir, checkpoint_handler._saved[-1][1]), os.path.join(log_dir, WEIGHTS_NAME))
        tb_logger.close()


if __name__ == "__main__":
    train()
