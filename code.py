import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import soundfile as sf
import numpy as np

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")

speech_array, sampling_rate = sf.read("4640-19187-0000.wav")

if len(speech_array.shape) > 1:
    speech_array = np.mean(speech_array, axis=1)

speech_array = speech_array / np.abs(speech_array).max()

input_values = processor(speech_array, sampling_rate=sampling_rate, return_tensors="pt", padding=True).input_values

with torch.no_grad():
    logits = model(input_values).logits

predicted_ids = torch.argmax(logits, dim=-1)

transcription = processor.batch_decode(predicted_ids, clean_up_tokenization_spaces=True)

print("Розпізнаний текст:", transcription[0])


"""
Розпізнаний текст: AND JUNE EIGHTEEN FORTY EIGHT KNEW A GREAT DEAL MORE ABOUT IT THAN JUNE EIGHTEEN THIRTY TWO SO THE BARRICADE OF THE
"""
