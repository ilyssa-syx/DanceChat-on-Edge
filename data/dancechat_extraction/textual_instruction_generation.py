import os
import random
from pathlib import Path
import librosa

from Music_Caption_Construction import extract_tempo_and_beats, extract_chord_progression, extract_key, extract_genre_from_filename

KW4_CHOICES=["This person", "A person", "A dancer"]


"""
Construct Prompting string
"""
def construct_prompt_for_file(audio_path: str) -> str:
    duration = librosa.get_duration(filename=audio_path)
    
    rhythm_info = extract_tempo_and_beats(audio_path)
    tempo = rhythm_info.get('tempo', 120.0)
    tempo_label = rhythm_info.get('tempo_marking', 'Unknown')
    signature = rhythm_info.get('signature', '4/4')

    chords = extract_chord_progression(audio_path)

    key = extract_key(audio_path)

    genre = extract_genre_from_filename(audio_path)

    description = f"The music has a tempo of {tempo} BPM, which is considered '{tempo_label}', follows a {signature} time signature, contains the chord progression {chords}, and is in the key of {key}."

    kw1 = f"{duration:.2f}"

    kw4 = random.choice(KW4_CHOICES)

    prompt = f"I have a {kw1}-second music clip with the following description: {description}"
    prompt += f"Can you provide a {genre} dance instruction that matches the music, focusing only on body movements (excluding the fingers)? "
    prompt += f"The instruction should begin with {kw4} in the third-person tone, avoiding any directional words, and be no longer than one sentence. YOu should generate only the instruction and nothing else."

    return prompt

