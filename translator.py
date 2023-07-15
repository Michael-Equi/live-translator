import openai
import asyncio
import os 
import dotenv
import time
import atexit
from elevenlabs import set_api_key, generate, play, stream

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

async def process_buffer(text):
    res = await openai.Completion.acreate(model="text-davinci-003", prompt=text,  max_tokens=32, temperature=0, stream=True, logprobs=5)
    return text, res["choices"][0]["text"], res["choices"][0]["logprobs"]["token_logprobs"]

def translate(text, translated_conversation_history, untranslated_conversation_history):
    return None, None

import multiprocessing
import threading
from multiprocessing import Queue
import subprocess

# def audio_to_play(q: Queue):
#     while True:
#         if not q.empty():
#             print("playing audio")
#             audio = q.get()
#             play(audio)
#             print("played audio")


# def audio_to_play(q: Queue):
#     args = ["ffplay", "-autoexit", "-", "-nodisp"]
#     proc = subprocess.Popen(
#         args=args,
#         stdout=subprocess.PIPE,
#         stdin=subprocess.PIPE,
#         stderr=subprocess.PIPE,
#         bufsize=0
#     )
#     done = False
#     while not done:
#         # Feed audio data into the subprocess
#         if not q.empty():
#             chunk = q.get()
#             if chunk is None:
#                 print("Killing player")
#                 # Close the input stream to signal to ffplay that there is no more data coming
#                 proc.stdin.close()
#                 proc.wait()
#                 done = True
#             else:
#                 print("Sending audio chunk")
#                 proc.stdin.write(chunk)
#                 proc.stdin.flush()

# args = ["ffplay", "-autoexit", "-", "-nodisp"]
# proc = subprocess.Popen(
#     args=args,
#     stdout=subprocess.PIPE,
#     stdin=subprocess.PIPE,
#     stderr=subprocess.PIPE,
#     bufsize=0
# )
# audio = generate(
#     text="Getting started",
#     voice="Arnold",
#     model="eleven_monolingual_v1",
# )
# proc.stdin.write(audio)
# proc.stdin.flush()

# def audio_to_play(q: Queue):
#     mpv_command = ["mpv", "--no-cache", "--no-terminal", "--", "fd://0"]
#     mpv_process = subprocess.Popen(
#         mpv_command,
#         stdin=subprocess.PIPE,
#         stdout=subprocess.DEVNULL,
#         stderr=subprocess.DEVNULL,
#     )

#     done = False
#     while not done:
#         if not q.empty():
#             chunk = q.get()
#             if chunk is None:
#                 done = True
#                 print("Finished TTS")
#             else:
#                 print("Processing chunk")
#                 mpv_process.stdin.write(chunk)  # type: ignore
#                 mpv_process.stdin.flush()  # type: ignore

#     if mpv_process.stdin:
#         mpv_process.stdin.close()
#     mpv_process.wait()

lock = threading.Lock()

mpv_command = ["mpv", "--no-cache", "--no-terminal", "--", "fd://0"]
mpv_process = subprocess.Popen(
    mpv_command,
    stdin=subprocess.PIPE,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)

def generate_audio(q: Queue, text):
    print("Generating aufio for:", text)
    audio = generate(
        text=text,
        voice="Arnold",
        model="eleven_monolingual_v1",
        stream=True
    )
    lock.acquire()
    for chunk in audio:
        if chunk is not None:
            mpv_process.stdin.write(chunk)  # type: ignore
            mpv_process.stdin.flush()  # type: ignore

    # if mpv_process.stdin:
    #     mpv_process.stdin.close()
    # mpv_process.wait()    
    lock.release()


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