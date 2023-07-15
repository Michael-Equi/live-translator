import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en")

input_text = "你好我叫张宝了"
encoded_input = tokenizer.encode(input_text, return_tensors="pt")
generated_text = "HHeyy"
encoded_pre_generation = tokenizer.encode(generated_text, return_tensors="pt")[0][:-1].unsqueeze(0)
encoded_pre_generation = torch.cat((torch.tensor(tokenizer.pad_token_id).unsqueeze(0), encoded_pre_generation[0]), dim=0).unsqueeze(0)
print(encoded_input)
print(encoded_pre_generation)

# Tokenize input + pre-output?
decoder_input_ids = torch.cat((encoded_input[0], encoded_pre_generation[0]), dim=0).unsqueeze(0) #type: ignore - huggingface screwed up their typing
print(decoder_input_ids)

# Generate output
outputs = model.generate(encoded_input, max_length=128, num_return_sequences=1, no_repeat_ngram_size=2, decoder_input_ids=encoded_pre_generation)
translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(translated_text)
print(outputs)

generated_ids = outputs[0].tolist()

# We'll compute the log probabilities one token at a time
individual_log_probs = []
for idx in range(1, len(generated_ids)):
    # Get the inputs up to (but not including) the current token
    inputs = torch.tensor(generated_ids[:idx]).unsqueeze(0)
    
    with torch.no_grad():
        logits = model(input_ids=encoded_input, decoder_input_ids=inputs).logits

    # Get the logits of the last token in the sequence
    logits = logits[:, -1, :]
    log_probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
    
    # Get the log probability of the current token
    token = generated_ids[idx]
    log_prob = log_probabilities[0, token].item()
    individual_log_probs.append(log_prob)



print(individual_log_probs)
