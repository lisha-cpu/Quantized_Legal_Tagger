import pandas as pd, torch, json
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from sklearn.metrics import accuracy_score, f1_score, classification_report

def main():
    df=pd.read_csv("data/processed/test.csv")
    labels=sorted(pd.read_csv("data/processed/train.csv").label_name.unique().tolist())
    tokenizer=GPT2Tokenizer.from_pretrained("models/original")
    model=GPT2LMHeadModel.from_pretrained("models/original")
    model.eval()
    preds=[]
    for clause in df.text.astype(str).tolist():
        prompt="Clause: "+clause+"\nLabel:"
        inputs=tokenizer(prompt,return_tensors="pt",truncation=True,max_length=220)
        with torch.no_grad():
            out=model.generate(**inputs,max_new_tokens=16,do_sample=False,pad_token_id=tokenizer.eos_token_id)
        generated=tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],skip_special_tokens=True).strip()
        match=next((lab for lab in labels if lab.lower() in generated.lower()),None)
        preds.append(match if match else labels[0])
    y=df.label_name.astype(str).tolist()
    print("Accuracy:",accuracy_score(y,preds))
    print("Macro F1:",f1_score(y,preds,average="macro",zero_division=0))
    print(classification_report(y,preds,zero_division=0))
if __name__=="__main__":main()
