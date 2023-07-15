from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import dotenv
import os
import asyncio

dotenv.load_dotenv()

class Source:
    def __init__(self) -> None:
        pass
    pass

class Sink:

    def __init__(self, source):
        pass

class ConvoBuffer:
    def __init__(self) -> None:
        self.convo_buffer = ""
    
    def add(self, text):
        self.convo_buffer += text

    def pop(self, text):
        assert text in self.convo_buffer, ""
        assert text[0:len(text)] == self.convo_buffer[0:len(text)], "text must come from the front of the convo buffer"
        self.convo_buffer = self.convo_buffer[len(text):]

    def get_text(self):
        return self.convo_buffer

# print(x["choices"][0]["logprobs"]["token_logprobs"])
# print(x["choices"][0]["text"])

# def callgpt(prompt):
#     # Stream from openai
#     for res in openai.Completion.create(model="text-davinci-003", prompt=prompt,  max_tokens=7, temperature=0, stream=True, logprobs=5):
#         yield res["choices"][0]["text"], res["choices"][0]["logprobs"]["token_logprobs"]

async def process_buffer(text):
    #res = await openai.Completion.acreate(model="text-davinci-003", prompt=text,  max_tokens=32, temperature=0, stream=True, logprobs=5)
    #return res["choices"][0]["text"], res["choices"][0]["logprobs"]["token_logprobs"]

class Translator:
    def __init__(self) -> None:
        self.convbuff = ConvoBuffer()
        self.output_buffer = ConvoBuffer()
        self.tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en")


    def process_translation(self, input_text):
        encoded_input = self.tokenizer.encode(input_text, return_tensors="pt")
        translation = self.model.generate(encoded_input, max_length=128)
        return self.tokenizer.decode(translation[0], skip_special_tokens=True)

    def add_chunk(self, text):
        # Receive chunk from TTS stream
        self.convbuff.add(text["choices"][0])
        # Send to translation process
        task = asyncio.create_task(process_buffer(self.convbuff.get_text()))
        # Add done callback
        task.add_done_callback(self.process_translation)



def main():
    pass