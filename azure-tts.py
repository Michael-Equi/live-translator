import azure.cognitiveservices.speech as speechsdk
from azure.cognitiveservices.speech.audio import AudioOutputConfig, AudioOutputStream
import dotenv
import os
import azure.cognitiveservices.speech as speechsdk
dotenv.load_dotenv()

# This example requires environment variables named "SPEECH_KEY" and "SPEECH_REGION"
speech_config = speechsdk.SpeechConfig(subscription=os.environ.get('SPEECH_KEY'), region=os.environ.get('SPEECH_REGION'))

# The language of the voice that speaks.
speech_config.speech_synthesis_voice_name='en-US-JennyNeural'

def handle(chunk):
    print(chunk)

# Create an in-memory audio stream
pull_stream = speechsdk.audio.PullAudioOutputStream()
# Creates a speech synthesizer using pull stream as audio output.
stream_config = speechsdk.audio.AudioOutputConfig(stream=pull_stream)

# Initialize the Speech Synthesizer
speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=stream_config)

# Get text from the console and synthesize to the default speaker.
# print("Enter some text that you want to speak >")
# text = input()

text = "Hello, what is you favorite color?"
result = speech_synthesizer.speak_text_async(text).get()

del speech_synthesizer

import subprocess
# mpv_command = ["mpv", "--no-cache", "--no-terminal", "--", "fd://0"]
# mpv_process = subprocess.Popen(
#     mpv_command,
#     stdin=subprocess.PIPE,
#     stdout=subprocess.DEVNULL,
#     stderr=subprocess.DEVNULL,
# )

# # Reads(pulls) data from the stream
# audio_buffer = bytes(32000)
# total_size = 0
# filled_size = pull_stream.read(audio_buffer)
# while filled_size > 0:
#     print(audio_buffer)
#     mpv_process.stdin.write(audio_buffer)
#     mpv_process.stdin.flush()
#     print("{} bytes received.".format(filled_size))
#     total_size += filled_size
#     filled_size = pull_stream.read(audio_buffer)
# print("Totally {} bytes received.".format(total_size))

# Starts an 'mpv' process
mpv_command = ["mpv", "--no-cache", "--no-terminal", "--", "/dev/stdin"]
mpv_process = subprocess.Popen(mpv_command, stdin=subprocess.PIPE)

# Create a Data Stream from the result
# audio_data_stream = speechsdk.AudioDataStream(result)

# Read all data from the stream and write to 'mpv'
audio_buffer = bytes(32000)
total_size = 0
filled_size = pull_stream.read(audio_buffer)
mpv_process.stdin.write(audio_buffer)

mpv_process.stdin.close()

# Wait for 'mpv' to finish
mpv_process.wait()

# text = "color"
# speech_synthesis_result = speech_synthesizer.speak_text_async(text).get()

# if speech_synthesis_result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
#     print("Speech synthesized for text [{}]".format(text))
# elif speech_synthesis_result.reason == speechsdk.ResultReason.Canceled:
#     cancellation_details = speech_synthesis_result.cancellation_details
#     print("Speech synthesis canceled: {}".format(cancellation_details.reason))
#     if cancellation_details.reason == speechsdk.CancellationReason.Error:
#         if cancellation_details.error_details:
#             print("Error details: {}".format(cancellation_details.error_details))
#             print("Did you set the speech resource key and region values?")