import argparse
import os
import random
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForMultipleChoice
from datasets import load_dataset, load_metric
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm import tqdm

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

SEED = 595
set_seed(SEED)
device = torch.device("cuda")# if torch.cuda.is_available() else torch.device("cpu")
#device = torch.device("cpu")


def load_data(tokenizer, params):

    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #
    #                   TODO: Implementation                      #
    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #

    def tokenize_function(examples):
        question = [[context] * 4 for context in examples["question"]]
        for i in range(len(examples["choices"])):
            if len(examples["choices"][i]["text"]) != 4:
                print(examples["choices"][i]["text"], len(examples["choices"][i]["text"]))

        sol = [[examples["choices"][i]["text"][0], examples["choices"][i]["text"][1],
                examples["choices"][i]["text"][2], examples["choices"][i]["text"][3]] for i in range(len(examples["choices"]))]
        question = sum(question, [])
        sol = sum(sol, [])
        tokenized_examples = tokenizer(question, sol, truncation=True, padding="max_length", max_length=128)
        out = {k: [v[i:i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
        out["labels"] = [examples["choices"][i]["label"].index(examples["answerKey"][i]) for i in range(len(examples["choices"]))]
        return out


    dataset = load_dataset(params.dataset, 'ARC-Challenge')
    dataset = dataset.filter(lambda example: len(example["choices"]["text"]) == 4)
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    tokenized_datasets = tokenized_datasets.remove_columns(['question', 'choices', "answerKey", "id"])
    tokenized_datasets.set_format("torch")

    train_dataloader = DataLoader(tokenized_datasets["train"], batch_size=params.batch_size, shuffle=True)
    eval_dataloader = DataLoader(tokenized_datasets["validation"], batch_size=params.batch_size, shuffle=True)
    test_dataloader = DataLoader(tokenized_datasets["test"], batch_size=params.batch_size, shuffle=True)
    ###################################################################################
    # train_dataset = tokenized_datasets["train"].shuffle(seed=SEED).select(range(20))
    # train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True)
    # eval_dataset = tokenized_datasets["validation"].shuffle(seed=SEED).select(range(20))
    # eval_dataloader = DataLoader(eval_dataset, batch_size=params.batch_size, shuffle=True)
    # test_dataset = tokenized_datasets["test"].shuffle(seed=SEED).select(range(20))
    # test_dataloader = DataLoader(test_dataset, batch_size=params.batch_size, shuffle=True)

    ###################################################################################



    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #

    return train_dataloader, eval_dataloader, test_dataloader


def finetune(model, train_dataloader, eval_dataloader, params):

    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #
    #                   TODO: Implementation                      #
    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #

    num_epochs = params.num_epochs
    num_training_steps = num_epochs * len(train_dataloader)

    optimizer = AdamW(model.parameters(), lr=params.lr)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    for epoch in range(params.num_epochs):
        print("This is epoch: ", epoch + 1)
        model.train()
        for i, batch in enumerate(train_dataloader):
            batch = {key: val.to(device) for key, val in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        # if (i + 1) % params.eval_epoch == 0:
        #     score = test(model, eval_dataloader, mode="eval")
        #     print("After ", i + 1, " steps, the accuracy: ", score)

    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #

    return model


def test(model, test_dataloader, prediction_save='predictions.torch', mode="test"):

    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #
    #                   TODO: Implementation                      #
    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #

    metric = load_metric('accuracy')
    model.eval()
    all_predictions = []

    for i, batch in enumerate(test_dataloader):
        batch = {key: val.to(device) for key, val in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    score = metric.compute()
    if mode == "test":
        #print()
        print('Test Accuracy:', score)
        torch.save(all_predictions, prediction_save)
        return score
    else:
        return score

    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #


def main(params):

    tokenizer = AutoTokenizer.from_pretrained(params.model)
    train_dataloader, eval_dataloader, test_dataloader = load_data(tokenizer, params)

    model = AutoModelForMultipleChoice.from_pretrained(params.model)
    model.to(device)
    model = finetune(model, train_dataloader, eval_dataloader, params)

    score = test(model, test_dataloader, prediction_save=f'predictions_{params.batch_size}_{params.lr}_{params.num_epochs}.torch')
    print("#################################")
    print(f"batch size: {params.batch_size}, lr: {params.lr}, number of epochs: {params.num_epochs}. Accuracy: {score}.")
    print("#################################")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune Language Model")

    parser.add_argument("--dataset", type=str, default="ai2_arc")
    parser.add_argument("--model", type=str, default="bert-base-cased")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--lr", type=int, default=1e-4)
    parser.add_argument("--eval_epoch", type=int, default=1)

    params, unknown = parser.parse_known_args()
    main(params)