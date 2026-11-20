import pandas as pd, torch, os
from datasets import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling

MODEL="gpt2"
OUT="models/original"
os.makedirs(OUT,exist_ok=True)

def main():
    train=pd.read_csv("data/processed/train.csv")
    test=pd.read_csv("data/processed/test.csv")
    labels=sorted(train.label_name.unique().tolist())
    # Classification as generation: prompt + target label text.
    train["joined"] = train.apply(lambda r: "Clause: "+str(r.text)+"\nLabel: "+str(r.label_name),axis=1)
    test["joined"] = test.apply(lambda r: "Clause: "+str(r.text)+"\nLabel: "+str(r.label_name),axis=1)
    tokenizer=GPT2Tokenizer.from_pretrained(MODEL)
    tokenizer.pad_token=tokenizer.eos_token
    model=GPT2LMHeadModel.from_pretrained(MODEL)
    model.config.pad_token_id=tokenizer.pad_token_id
    train_ds=Dataset.from_pandas(train[["joined"]])
    test_ds=Dataset.from_pandas(test[["joined"]])
    def tok(batch):
        return tokenizer(batch["joined"], truncation=True, max_length=256, padding="max_length")
    train_ds=train_ds.map(tok,batched=True,remove_columns=["joined"])
    test_ds=test_ds.map(tok,batched=True,remove_columns=["joined"])
    train_ds=train_ds.map(lambda x: {"labels":x["input_ids"]})
    test_ds=test_ds.map(lambda x: {"labels":x["input_ids"]})
    args=TrainingArguments(
        output_dir="models/training_checkpoints", num_train_epochs=1,
        per_device_train_batch_size=2, per_device_eval_batch_size=2,
        logging_steps=50, save_strategy="no", report_to="none",
        fp16=torch.cuda.is_available(), learning_rate=5e-5,
        max_steps=300
    )
    trainer=Trainer(model=model,args=args,train_dataset=train_ds,eval_dataset=test_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False))
    trainer.train()
    model.save_pretrained(OUT); tokenizer.save_pretrained(OUT)
    with open("data/processed/labels.json","w") as f:
        import json; json.dump(labels,f)
    print("Saved fine-tuned model:",OUT)
if __name__=="__main__": main()
