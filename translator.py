import openai
import dotenv
import os
import asyncio

dotenv.load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]
openai.organization = os.environ["OPENAI_ORG"]

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

async def process_buffer(text):
    res = await openai.Completion.acreate(model="text-davinci-003", prompt=text,  max_tokens=32, temperature=0, stream=True, logprobs=5)
    return text, res["choices"][0]["text"], res["choices"][0]["logprobs"]["token_logprobs"]

class Translator:
    def __init__(self) -> None:
        self.conversation_history = ""
        self.translated_text = ""
        self.convbuff = ConvoBuffer()

    def process_translation(self, text, translation, logprobs, threshold = 0.5):
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

    def send_to_tts(self, text):
        print("Send to tts")

    def add_chunk(self, text):
        # Receive chunk from TTS stream
        self.convbuff.add(text["choices"][0])
        # Send to translation process
        task = asyncio.create_task(process_buffer(self.convbuff.get_text()))
        # Add done callback
        task.add_done_callback(self.process_translation)



def main():
    pass