import openai
import asyncio
import os 
import dotenv
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from elevenlabs import set_api_key, generate, play, stream
import threading
import torch
from multiprocessing import Queue



# dotenv.load_dotenv()
# openai.api_key = os.environ["OPENAI_API_KEY"]
# openai.organization = os.environ["OPENAI_ORG"]

dotenv.load_dotenv()
set_api_key(os.environ["ELEVEN_LABS_API_KEY"])

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
        assert text in self.convo_buffer, "text must be in conv buffer"
        assert text[0:len(text)] == self.convo_buffer[0:len(text)], "text must come from the front of the convo buffer"
        poped_buffer = self.convo_buffer[len(text):]
        self.convo_buffer[len(text):] = ""
        return poped_buffer

    def get_text(self):
        return self.convo_buffer

# print(x["choices"][0]["logprobs"]["token_logprobs"])
# print(x["choices"][0]["text"])

# def callgpt(prompt):
#     # Stream from openai
#     for res in openai.Completion.create(model="text-davinci-003", prompt=prompt,  max_tokens=7, temperature=0, stream=True, logprobs=5):
#         yield res["choices"][0]["text"], res["choices"][0]["logprobs"]["token_logprobs"]

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en")

"""async def process_buffer(text):
    res = await openai.Completion.acreate(model="text-davinci-003", prompt=text,  max_tokens=32, temperature=0, stream=True, logprobs=5)
    return text, res["choices"][0]["text"], res["choices"][0]["logprobs"]["token_logprobs"]
"""
def translate(left_to_translate, translated_conversation_history, untranslated_conversation_history):
    input_text = untranslated_conversation_history + left_to_translate
    encoded_input = tokenizer.encode(input_text, return_tensors="pt")

    # Tokenize and process forced start-of-generation tokens
    generated_ids = tokenizer.encode(translated_conversation_history, return_tensors="pt")[0][:-1].unsqueeze(0)
    generated_ids = torch.cat((torch.tensor(tokenizer.pad_token_id).unsqueeze(0), generated_ids[0]), dim=0).unsqueeze(0)

    # Initialize logits and tokens lists
    log_probabilities = []
    tokens_list = torch.empty(0)

    # Keep generating until EOS token
    while True:        
        outputs = model(input_ids=encoded_input, decoder_input_ids=generated_ids)
        
        # Get logits and token of the last position
        logits, token = outputs.logits[:, -1, :], torch.argmax(outputs.logits[:, -1, :], dim=1)
        
        # Add predicted token to tokens_list and update generated_ids
        tokens_list = torch.cat((tokens_list, token), dim=-1)
        generated_ids = torch.cat((generated_ids, token.unsqueeze(0)), dim=-1)

        # Appends log probability of most-likely token
        log_probabilities.append(torch.nn.functional.log_softmax(logits, dim=1)[0].max().item())

        # Breaks if EOS token is presented
        if token == tokenizer.eos_token_id:
            break

    # Getting the translated text
    translated_text = tokenizer.decode(tokens_list, skip_special_tokens=True)
    return translated_text, log_probabilities


class Translator:
    def __init__(self) -> None:
        self.translated_text = ""
        self.untranslated_text = ""
        self.untranslated_conversation_history = ""
        self.translated_conversation_history = ''
        # Output of stt
        # self.stt_buffer = ConvoBuffer()

        self.tts_q = Queue()
        # self.player = multiprocessing.Process(target=audio_to_play, args=(self.tts_q,))
        # self.player.start()
        # atexit.register(self.player.terminate)

    def __del__(self):
        # self.player.terminate()
        pass

    def send_to_tts(self, text):
        # Generate audio in another thread
        threading.Thread(target=generate_audio, args=(self.tts_q, text)).start() 

    def process_translation(self, text, translation, logprobs, threshold = 0.5):
        if sum(logprobs) / len(logprobs) > threshold:
            self.translated_text += translation
            self.untranslated_conversation_history += text
            self.translated_conversation_history += translation
            self.untranslated_text.replace(text, "")
            return translation
        return None
    
    def add_chunk(self, text):
        self.untranslated_text += text
        translated_text, logprobs = translate(text)
        translation = self.process_translation(text, translated_text, logprobs)
        if translation is not None:
            self.send_to_tts(translation)


    def process_translation_async(self, text, translation, logprobs, threshold = 0.5):
        # Check if the text has already been added to the conversation history
        if len(self.translated_text) >= len(text) and text == self.translated_text[-len(text):]:
            # Text has already been translated
            return 
        elif True:
            # TODO check if part of the text has already been translated
            # Sum the logprobs and divide by length
            if sum(logprobs) / len(logprobs) > threshold:
                self.conversation_history += self.convbuff.pop(translation)
                self.send_to_tts(translation)

    def add_chunk_async(self, text):
        # Receive chunk from TTS stream
        self.convbuff.add(text["choices"][0])
        # Send to translation process
        task = asyncio.create_task(process_buffer(self.convbuff.get_text()))
        # Add done callback
        task.add_done_callback(self.process_translation_async)



def main():
    translator = Translator()
    print("Sending to tts")
    translator.send_to_tts("What is your favorite")
    print("Sent to tts")
    translator.send_to_tts("color?")
    time.sleep(10)
    print("Sending to tts")
    translator.send_to_tts("What is your favorite")
    print("Sent to tts")
    translator.send_to_tts("color?")

if __name__ == '__main__':
    main()