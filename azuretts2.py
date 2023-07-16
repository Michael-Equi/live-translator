import azure.cognitiveservices.speech as speechsdk
import os, sys, pyaudio
pa = pyaudio.PyAudio()

import dotenv
dotenv.load_dotenv()

my_text = "My emotional experiences are varied"

voc_data = {
    'channels': 1, # Mono sound
    'rate': 16000, # 16KHz
    'width': 2, # 16 bit depth = 2 bytes
    'format': pyaudio.paInt16, # 16 bit depth
    'frames': [] # Placeholder for your frames data
}

my_stream = pa.open(format=voc_data['format'],
                    channels=voc_data['channels'],
                    rate=voc_data['rate'],
                    output=True)

speech_key = os.getenv('SPEECH_KEY')
service_region = os.getenv('SPEECH_REGION')

def speech_synthesis_to_push_audio_output_stream():
    """performs speech synthesis and push audio output to a stream"""
    class PushAudioOutputStreamSampleCallback(speechsdk.audio.PushAudioOutputStreamCallback):
        """
        Example class that implements the PushAudioOutputStreamCallback, which is used to show
        how to push output audio to a stream
        """
        def __init__(self) -> None:
            super().__init__()
            self._audio_data = bytes(0)
            self._closed = False

        def write(self, audio_buffer: memoryview) -> int:
            """
            The callback function which is invoked when the synthesizer has an output audio chunk
            to write out
            """
            self._audio_data += audio_buffer
            my_stream.write(audio_buffer.tobytes())  # Convert memoryview to bytes
            print("{} bytes received.".format(audio_buffer.nbytes))
            return audio_buffer.nbytes

        def close(self) -> None:
            """
            The callback function which is invoked when the synthesizer is about to close the
            stream.
            """
            self._closed = True
            print("Push audio output stream closed.")

        def get_audio_data(self) -> bytes:
            return self._audio_data

        def get_audio_size(self) -> int:
            return len(self._audio_data)

    # Creates an instance of a speech config with specified subscription key and service region.
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    # Creates customized instance of PushAudioOutputStreamCallback
    stream_callback = PushAudioOutputStreamSampleCallback()
    # Creates audio output stream from the callback
    push_stream = speechsdk.audio.PushAudioOutputStream(stream_callback)
    # Creates a speech synthesizer using push stream as audio output.
    stream_config = speechsdk.audio.AudioOutputConfig(stream=push_stream)
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=stream_config)

    # Receives a text from console input and synthesizes it to stream output.

    result = speech_synthesizer.speak_text_async(my_text).get()
    # Check result
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("Speech synthesized for text [{}], and the audio was written to output stream.".format(my_text))
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Speech synthesis canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))
    # Destroys result which is necessary for destroying speech synthesizer
    del result

    # Destroys the synthesizer in order to close the output stream.
    del speech_synthesizer

    print("Totally {} bytes received.".format(stream_callback.get_audio_size()))

speech_synthesis_to_push_audio_output_stream()