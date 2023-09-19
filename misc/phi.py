import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


if __name__ == '__main__':
    device = torch.device('cuda')
    model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5", trust_remote_code=True, torch_dtype="auto").to(device)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5", trust_remote_code=True, torch_dtype="auto")
    inputs = tokenizer('''```python
    def print_prime(n):
       """
       Print all primes between 1 and n
       """''', return_tensors="pt", return_attention_mask=False)

    outputs = model.generate(**inputs, max_length=200)
    text = tokenizer.batch_decode(outputs)[0]
    print(text)
