import logging
from pprint import pformat
from argparse import ArgumentParser
from functools import partial

from tqdm.auto import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset, load_metric
from transformers import  GPT2DoubleHeadsModel, GPT2Tokenizer, DataCollatorWithPadding, get_scheduler
from transformers.modeling_outputs import TokenClassifierOutput


logger = logging.getLogger(__file__)


class CustomGPTModel(nn.Module):
    def __init__(self, gpt2_model, num_labels, args):
        super(CustomGPTModel, self).__init__()
        self.gpt2 = gpt2_model
        self.num_labels = num_labels
        self.max_seq_length = args.max_seq_length
        self.classifier = nn.Linear(768 * self.max_seq_length, num_labels)

    def forward(self, input_ids, attention_mask, labels):
        output = self.gpt2(input_ids, attention_mask=attention_mask)

        last_hidden_state = output.hidden_states[0]
        last_hidden_state = last_hidden_state.view(-1, 768 * self.max_seq_length)
        logits = self.classifier(last_hidden_state)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(loss=loss,
                                     logits=logits,
                                     hidden_states=output.hidden_states,
                                     attentions=output.attentions)


def tokenize(batch, tokenizer, args):
  return tokenizer(batch["text"], truncation=True, max_length=args.max_seq_length)


def train():
    parser = ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="gpt2",
                        help="Pretrained model name or path")
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="Max sequence which all sequences will be padded")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Train & eval batch size")
    parser.add_argument("--n_epochs", type=int, default=3,
                        help="Number of training epochs")
    args = parser.parse_args()


    # logging is set to INFO (resp. WARN) for main (resp. auxiliary) process. logger.info => log main process only,
    # logger.warning => log all processes
    logging.basicConfig(level=logging.INFO)
    logger.info("Arguments: %s", pformat(args))

    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    gpt2_model = GPT2DoubleHeadsModel.from_pretrained(args.model_name_or_path,
                                                      output_hidden_states=True,
                                                      output_attentions=True,
                                                      return_dict=True)
    gpt2_model.resize_token_embeddings(new_num_tokens=len(tokenizer.encoder) + 1)
    model = CustomGPTModel(gpt2_model, 77, args)
    model.to(args.device)


    criterion = nn.CrossEntropyLoss() ## If required define your own criterion
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

    dataset = load_dataset('banking77')
    tokenized_dataset = dataset.map(partial(tokenize, tokenizer=tokenizer, args=args))
    tokenized_dataset.set_format("torch",columns=["input_ids", "attention_mask", "label"])
    train_dataset, test_dataset = tokenized_dataset['train'], tokenized_dataset['test']
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer,
                                            padding='max_length',
                                            max_length=args.max_seq_length)
    lr_scheduler = get_scheduler("linear",
                                 optimizer=optimizer,
                                 num_warmup_steps=0,
                                 num_training_steps=args.n_epochs)

    train_dataset_size = int(0.9 * len(train_dataset))
    val_dataset_size = len(train_dataset) - train_dataset_size
    train_dataset, valid_dataset = torch.utils.data.random_split(tokenized_dataset["train"],
                                                                  [train_dataset_size, val_dataset_size])
    train_dataloader = DataLoader(train_dataset,
                                  shuffle=True,
                                  batch_size=args.batch_size,
                                  collate_fn=data_collator)
    val_dataloader = DataLoader(valid_dataset,
                                  shuffle=True,
                                  batch_size=args.batch_size,
                                  collate_fn=data_collator)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=args.batch_size,
                                 collate_fn=data_collator)

    progress_bar_train = tqdm(range(args.n_epochs))
    progress_bar_eval = tqdm(range(args.n_epochs * len(val_dataloader)))

    metric = load_metric("accuracy")
    for epoch in range(args.n_epochs):
        model.train()
        for batch in train_dataloader:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar_train.update(1)

        model.eval()
        for batch in val_dataloader:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])
            progress_bar_eval.update(1)

    print(metric.compute())

    model.eval()
    for batch in test_dataloader:
        batch = {k: v.to(args.device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
        progress_bar_eval.update(1)


if __name__ == "__main__":
    train()