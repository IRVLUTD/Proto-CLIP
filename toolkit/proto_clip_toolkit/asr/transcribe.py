"""
Code credits: https://github.com/davabase/whisper_real_time
"""
import io
import speech_recognition as sr
import whisper
import torch
import time

from datetime import datetime, timedelta
from queue import Queue
from time import sleep
from sys import platform
from .asr_utils import Config

def transcribe(asr_config_path):
    """Prints the transcribed text to the console."""
    config = Config(asr_config_path)

    phrase_time = None
    last_sample = bytes()
    data_queue = Queue() # Queue containing the audio data from the callback.

    recorder = sr.Recognizer()
    recorder.energy_threshold = config.energy_threshold
    recorder.dynamic_energy_threshold = False
    
    if 'linux' in platform:
        mic_name = config.default_microphone
        if not mic_name or mic_name == 'list':
            print("Available microphone devices are: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"Microphone with name \"{name}\" found")   
            return
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    source = sr.Microphone(sample_rate=44100, device_index=index)
                    break
    else:
        source = sr.Microphone(sample_rate=44100)
        

    if "whisper" in config.model:
        _, model_type = config.model.split("-")
        if model_type != "large" and not config.non_english:
            model_type = model_type + ".en"
        audio_model = whisper.load_model(model_type)

    record_timeout = config.record_timeout
    phrase_timeout = config.phrase_timeout
    
    transcription = ['']
    
    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio:sr.AudioData) -> None:
        """
        Threaded callback function to recieve audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        # Grab the raw bytes and push it into the thread safe queue.
        data = audio.get_raw_data()
        data_queue.put(data)

    # Create a background thread that will pass us raw audio bytes.
    # We could do this manually but SpeechRecognizer provides a nice helper.
    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    # Cue the user that we're ready to go.
    print("Model loaded.\n")
    transcription_in_progress = True

    while transcription_in_progress:
        try:
            now = datetime.utcnow()
            # Pull raw recorded audio from the queue.
            if not data_queue.empty():
                phrase_complete = False
                # If enough time has passed between recordings, consider the phrase complete.
                # Clear the current working audio buffer to start over with the new data.
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    last_sample = bytes()
                    phrase_complete = True
                # This is the last time we received new audio data from the queue.
                phrase_time = now

                # Concatenate our current audio data with the latest audio data.
                while not data_queue.empty():
                    data = data_queue.get()
                    last_sample += data

                # Use AudioData to convert the raw data to wav data.
                audio_data = sr.AudioData(last_sample, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
                wav_data = io.BytesIO(audio_data.get_wav_data())

                time_stamp = int(time.time())
                file_name = f"./logs/{time_stamp}_audio_recording.wav"
                # Write wav data to the temporary file as bytes.
                with open(file_name, 'w+b') as f:
                    f.write(wav_data.read())

                # Read the transcription.
                result = audio_model.transcribe(file_name, fp16=torch.cuda.is_available())
                text = result['text'].strip()

                # If we detected a pause between recordings, add a new item to our transcripion.
                # Otherwise edit the existing one.
                if phrase_complete:
                    print(f"Transcribed text: {text}")
                    transcription.append(text)
                else:
                    transcription[-1] = text
                
                sleep(0.25)
        except KeyboardInterrupt:
            break