import os
import time
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch import nn
from transformers.pytorch_utils import Conv1D

MODEL_PATH = "models/original"
QUANT_PATH = "models/dynamic_quantized/quantized_state_dict.pt"

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

def measure(model, inputs, runs=10):
    model.eval()

    with torch.no_grad():
        for _ in range(2):
            model(**inputs)

        timings = []
        for _ in range(runs):
            start = time.perf_counter()
            model(**inputs)
            timings.append((time.perf_counter() - start) * 1000)

    return sum(timings) / len(timings)

def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    text = "This agreement shall remain in force under applicable laws."
    inputs = tokenizer(text, return_tensors="pt")

    print("Benchmarking original model...")
    original = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    original.eval()

    original_latency = measure(original, inputs)

    original_size = sum(
        os.path.getsize(os.path.join(root, file))
        for root, _, files in os.walk(MODEL_PATH)
        for file in files
    ) / (1024 ** 2)

    del original

    print("Loading quantized model...")
    quantized = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    quantized.eval()

    convert_conv1d(quantized)

    quantized = torch.ao.quantization.quantize_dynamic(
        quantized,
        {nn.Linear},
        dtype=torch.qint8
    )

    state = torch.load(
        QUANT_PATH,
        map_location="cpu",
        weights_only=False
    )
    quantized.load_state_dict(state)
    quantized.eval()

    print("Benchmarking quantized model...")
    quantized_latency = measure(quantized, inputs)

    quantized_size = os.path.getsize(QUANT_PATH) / (1024 ** 2)

    results = pd.DataFrame([
        {
            "Model": "Original",
            "Size MB": original_size,
            "Mean latency ms": original_latency
        },
        {
            "Model": "Dynamic INT8",
            "Size MB": quantized_size,
            "Mean latency ms": quantized_latency
        }
    ])

    results.to_csv("benchmark_results.csv", index=False)

    print("\nBenchmark Results:")
    print(results.to_string(index=False))

    print("\nSize reduction: {:.2f}%".format(
        (1 - quantized_size / original_size) * 100
    ))

    print("Latency difference: {:.2f}%".format(
        (1 - quantized_latency / original_latency) * 100
    ))

    print("\nSaved to benchmark_results.csv")

if __name__ == "__main__":
    main()