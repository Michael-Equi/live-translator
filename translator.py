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
from google.cloud import speech
from speech_to_text.stream_input import ResumableMicrophoneStream, STREAMING_LIMIT, SAMPLE_RATE, CHUNK_SIZE, RED, GREEN, YELLOW
import subprocess
import re
import sys

dotenv.load_dotenv()
set_api_key(os.environ["ELEVEN_LABS_API_KEY"])

class TTS:
    
    def __init__(self):
        self.lock = threading.Lock()
        mpv_command = ["mpv", "--no-cache", "--no-terminal", "--", "fd://0"]
        self.mpv_process = subprocess.Popen(
            mpv_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def send(self, text):
        # generate audio in a separate thread
        t = threading.Thread(target=self.generate_audio, args=(text,))
        t.start()
        return t

    def generate_audio(self, text):
        print(text)
        audio = generate(
            text=text,
            voice="Arnold",
            model="eleven_monolingual_v1",
            stream=True
        )
        self.lock.acquire()
        for chunk in audio:
            if chunk is not None:
                self.mpv_process.stdin.write(chunk)
                self.mpv_process.stdin.flush()
        self.lock.release()

    def __del__(self):
        if self.mpv_process.stdin:
            self.mpv_process.stdin.close()
        self.mpv_process.wait()    


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

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en")

def translate(left_to_translate, translated_conversation_history, untranslated_conversation_history):
    input_text = untranslated_conversation_history + left_to_translate
    encoded_input = tokenizer.encode(input_text, return_tensors="pt")

    # Tokenize and process forced start-of-generation tokens
    generated_ids = tokenizer.encode(translated_conversation_history, return_tensors="pt")[0][:-1].unsqueeze(0) # type: ignore
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
        self.untranslated_conversation_history = ""
        self.translated_conversation_history = ''

        self.tts = TTS()

    def send_to_tts(self, text):
        self.tts.send(text)

    def process_translation(self, text, translation, logprobs, threshold = -2):
        print(sum(logprobs) / len(logprobs))
        if sum(logprobs) / len(logprobs) > threshold:
            self.translated_text += translation
            self.untranslated_conversation_history += text
            self.translated_conversation_history += translation
            return translation
        return None
    
    def add_chunk(self, text):
        translated_text, logprobs = translate(text, self.translated_conversation_history, self.untranslated_conversation_history)
        translation = self.process_translation(text, translated_text, logprobs)
        if translation is not None and str.strip(translation) != "":
            print(translation)
            self.send_to_tts(translation)
            return True
        return False


    """def process_translation_async(self, text, translation, logprobs, threshold = 0.5):
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
"""

def get_current_time():
    """Return Current Time in MS."""

    return int(round(time.time() * 1000))

def translate_chunk(responses, stream, translator):
    """Iterates through server responses and prints them.
    The responses passed is a generator that will block until a response
    is provided by the server.
    Each response may contain multiple results, and each result may contain
    multiple alternatives; for details, see https://goo.gl/tjCPAU.  Here we
    print only the transcription for the top alternative of the top result.
    In this case, responses are provided for interim results as well. If the
    response is an interim one, print a line feed at the end of it, to allow
    the next result to overwrite it, until the response is a final one. For the
    final one, print a newline to preserve the finalized transcription.
    """

    for response in responses:

        if get_current_time() - stream.start_time > STREAMING_LIMIT:
            stream.start_time = get_current_time()
            break

        if not response.results:
            continue

        result = response.results[0]

        if not result.alternatives:
            continue

        transcript = result.alternatives[0].transcript

        result_seconds = 0
        result_micros = 0

        if result.result_end_time.seconds:
            result_seconds = result.result_end_time.seconds

        if result.result_end_time.microseconds:
            result_micros = result.result_end_time.microseconds

        stream.result_end_time = int((result_seconds * 1000) + (result_micros / 1000))

        corrected_time = (
            stream.result_end_time
            - stream.bridging_offset
            + (STREAMING_LIMIT * stream.restart_counter)
        )

        # Translate current results - if it passes the check, we'll keep
        trans, probs = translate(left_to_translate=transcript, translated_conversation_history="", untranslated_conversation_history="")
        print(transcript)
        print(trans)
        print(sum(probs) / len(probs))
        print(max(probs) / len(probs))
        print(min(probs) / len(probs))


        # Display interim results, but with a carriage return at the end of the
        # line, so subsequent lines will overwrite them.

        if result.is_final:

            """sys.stdout.write(GREEN)
            sys.stdout.write("\033[K")
            sys.stdout.write(str(corrected_time) + ": " + transcript + "\n")"""
            #translator.add_chunk(transcript)
            stream.is_final_end_time = stream.result_end_time
            stream.last_transcript_was_final = True

            # Exit recognition if any of the transcribed phrases could be
            # one of our keywords.
            if re.search(r"\b(exit|quit)\b", transcript, re.I):
                """sys.stdout.write(YELLOW)
                sys.stdout.write("Exiting...\n")"""
                stream.closed = True
                break
            
        else:
            """sys.stdout.write(RED)
            sys.stdout.write("\033[K")
            sys.stdout.write(str(corrected_time) + ": " + transcript + "\r")"""
            print(transcript)

            stream.last_transcript_was_final = False

translator = Translator()

def process_utterances():
    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=SAMPLE_RATE,
        language_code="zh",
        max_alternatives=1,
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config, interim_results=True
    )

    

    #voice.generate_and_play_audio("Hi my name is Michael Lutz, can I have 20 seconds?", playInBackground=True)

    mic_manager = ResumableMicrophoneStream(SAMPLE_RATE, CHUNK_SIZE)
    print(mic_manager.chunk_size)
    """sys.stdout.write(YELLOW)
    sys.stdout.write('\nListening, say "Quit" or "Exit" to stop.\n\n')
    sys.stdout.write("End (ms)       Transcript Results/Status\n")
    sys.stdout.write("=====================================================\n")"""

    with mic_manager as stream:

        while not stream.closed:
            """sys.stdout.write(YELLOW)
            sys.stdout.write(
                "\n" + str(STREAMING_LIMIT * stream.restart_counter) + ": NEW REQUEST\n"
            )"""

            stream.audio_input = []
            audio_generator = stream.generator()

            requests = (
                speech.StreamingRecognizeRequest(audio_content=content)
                for content in audio_generator
            )

            responses = client.streaming_recognize(streaming_config, requests)

            # Now, put the transcription responses to use.
            translate_chunk(responses, stream, translator)

            if stream.result_end_time > 0:
                stream.final_request_end_time = stream.is_final_end_time
            stream.result_end_time = 0
            stream.last_audio_input = []
            stream.last_audio_input = stream.audio_input
            stream.audio_input = []
            stream.restart_counter = stream.restart_counter + 1

            if not stream.last_transcript_was_final:
                sys.stdout.write("\n")
            stream.new_stream = True


def main():
    process_utterances()

if __name__ == '__main__':
    main()