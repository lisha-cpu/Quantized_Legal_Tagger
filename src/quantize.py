import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from transformers.pytorch_utils import Conv1D

def convert_conv1d(module):
    for name, child in list(module.named_children()):
        if isinstance(child, Conv1D):
            new_layer = nn.Linear(
                child.weight.shape[0],
                child.weight.shape[1],
                bias=True
            )
            new_layer.weight.data.copy_(child.weight.data.T)
            new_layer.bias.data.copy_(child.bias.data)
            setattr(module, name, new_layer)
        else:
            convert_conv1d(child)

def main():
    model_path = "models/original"
    output_path = "models/dynamic_quantized"

    print("Loading fine-tuned GPT-2...")
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.eval()

    print("Converting GPT-2 Conv1D layers...")
    convert_conv1d(model)

    print("Applying dynamic INT8 quantization...")
    quantized_model = torch.ao.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8
    )

    os.makedirs(output_path, exist_ok=True)

    print("Saving quantized model...")
    torch.save(
        quantized_model.state_dict(),
        os.path.join(output_path, "quantized_state_dict.pt")
    )

    model.config.save_pretrained(output_path)

    original_size = sum(
        os.path.getsize(os.path.join(root, file))
        for root, _, files in os.walk(model_path)
        for file in files
    )

    quantized_size = os.path.getsize(
        os.path.join(output_path, "quantized_state_dict.pt")
    )

    print("\nQuantization finished!")
    print(f"Original model: {original_size / (1024**2):.2f} MB")
    print(f"Quantized file: {quantized_size / (1024**2):.2f} MB")

    if quantized_size < original_size:
        reduction = (1 - quantized_size / original_size) * 100
        print(f"Size reduction: {reduction:.2f}%")
    else:
        print("The quantized file did not reduce in size.")

if __name__ == "__main__":
    main()