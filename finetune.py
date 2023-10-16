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

SEED = 596
set_seed(SEED)
#device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cpu")


def load_data(tokenizer, params):

    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #
    #                   TODO: Implementation                      #
    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #

    # def tokenize_function(examples):
    #     question = [[context] * 4 for context in examples["question"]]
    #     for i in range(len(examples["choices"])):
    #         if len(examples["choices"][i]["text"]) != 4:
    #             print(examples["choices"][i]["text"], len(examples["choices"][i]["text"]))
    #
    #     sol = [[examples["choices"][i]["text"][0], examples["choices"][i]["text"][1],
    #             examples["choices"][i]["text"][2], examples["choices"][i]["text"][3]] for i in range(len(examples["choices"]))]
    #     question = sum(question, [])
    #     sol = sum(sol, [])
    #     tokenized_examples = tokenizer(question, sol, truncation=True, padding=True, max_length=128)
    #     return {k: [v[i:i + 4] for i in range(0, len(v), 2)] for k, v in tokenized_examples.items()}

    # def collate_fn(batch):
    #     #(labels, input_ids, token_type_ids, attention_mask) = zip(*batch)
    #     #print(labels, input_ids, token_type_ids, attention_mask)
    #     accepted_keys = ["input_ids", "attention_mask", "label"]
    #     features = [{k: v for k, v in batch[i].items() if k in accepted_keys} for i in range(len(batch))]
    #     label = batch["label"]
    #
    #     return torch.tensor(input_ids, dtype=torch.long, device=device), \
    #            torch.tensor(labels, dtype=torch.long, device=device)
    #
    # def collate_fn(features):
    #     label_name = "label" if "label" in features[0].keys() else "labels"
    #     labels = [feature.pop(label_name) for feature in features]
    #     batch_size = len(features)
    #     num_choices = len(features[0]["input_ids"])
    #     flattened_features = [
    #         [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
    #     ]
    #     batch = sum(flattened_features, [])
    #     print(batch[:2])
    #     batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
    #     batch["labels"] = torch.tensor(labels, dtype=torch.int64)
    #     return batch

    def tokenize_function(examples):
        len_map = []
        question = []
        sol = []
        for i in range(len(examples["choices"])):
            length = len(examples["choices"][i]["text"])
            len_map.append(length)
            question.extend([examples["question"][i]] * length)
            sol.extend([*examples["choices"][i]["text"]])

        tokenized_examples = tokenizer(question, sol, truncation=True, padding='max_length', max_length=128)
        out = {}
        cnt = 0
        out["labels"] = []
        for k, v in tokenized_examples.items():
            temp = []
            for i in len_map:
                temp.append(v[cnt: cnt + i])
                cnt += i
            out[k] = temp
            cnt = 0
        for i in range(len(len_map)):
            out["labels"].append(examples["choices"][i]["label"].index(examples["answerKey"][i]))
        return out


    dataset = load_dataset(params.dataset, 'ARC-Challenge')
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

    optimizer = AdamW(model.parameters(), lr=1e-4)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    for epoch in range(params.num_epochs):
        print("This is epoch: ", epoch + 1)
        model.train()
        for i, batch in enumerate(tqdm(train_dataloader)):
            batch = {key: val.to(device) for key, val in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        if (i + 1) % params.eval_epoch == 0:
            score = test(model, eval_dataloader, mode="eval")
            print("After the ", i + 1, " steps, the accuracy: ", score)

    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #

    return model


def test(model, test_dataloader, prediction_save='predictions.torch', mode="test"):

    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #
    #                   TODO: Implementation                      #
    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #

    metric = load_metric('accuracy')
    model.eval()
    all_predictions = []

    for i, batch in enumerate(tqdm(test_dataloader)):
        batch = {key: val.to(device) for key, val in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    score = metric.compute()
    if mode == "test":
        print('Test Accuracy:', score)
        torch.save(all_predictions, prediction_save)
    else:
        return score

    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #


def main(params):

    tokenizer = AutoTokenizer.from_pretrained(params.model)
    train_dataloader, eval_dataloader, test_dataloader = load_data(tokenizer, params)

    model = AutoModelForMultipleChoice.from_pretrained(params.model)
    model.to(device)
    model = finetune(model, train_dataloader, eval_dataloader, params)

    test(model, test_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune Language Model")

    parser.add_argument("--dataset", type=str, default="ai2_arc")
    parser.add_argument("--model", type=str, default="bert-base-cased")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--eval_epoch", type=int, default=1)

    params, unknown = parser.parse_known_args()
    main(params)