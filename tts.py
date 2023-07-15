import dotenv
import numpy as np
import os
from elevenlabs import set_api_key, generate, play, stream

dotenv.load_dotenv()
set_api_key(os.environ["ELEVEN_LABS_API_KEY"])

audio = generate(
  text="Hi! My name is Bella, nice to meet you!",
  voice="Arnold",
  model="eleven_monolingual_v1",
  stream=True
)

# play(audio)

# from elevenlabs import generate, stream

# audio_stream = generate(
#   text="This is a... streaming voice!!",
#   voice="Arnold",
#   stream=True
# )

# stream(audio_stream)

import time 
import subprocess
import pyaudio

# Assume you have a generator to generate audio stream
# def generate(text, voice, model, stream):
#     # this method should return an Iterator[bytes] that yields audio data
#     pass

audio = generate(
    text="Hi! My name is Arnold, nice to meet you!",
    voice="Arnold",
    model="eleven_monolingual_v1",
    stream=True
)

args = ["ffplay", "-autoexit", "-", "-nodisp"]
proc = subprocess.Popen(
    args=args,
    stdout=subprocess.PIPE,
    stdin=subprocess.PIPE,
    stderr=subprocess.PIPE,
    bufsize=0
)

# Feed audio data into the subprocess
for i, chunk in enumerate(audio):
    print("chunk", i)
    proc.stdin.write(chunk)
    proc.stdin.flush()

# Close the input stream to signal to ffplay that there is no more data coming
proc.stdin.close()
proc.wait()

# Check for any output/errors
# out, err = proc.stderr.read(), proc.stdout.read()
# if out:
#     print("Output:", out)
# if err:
#     print("Error:", err)