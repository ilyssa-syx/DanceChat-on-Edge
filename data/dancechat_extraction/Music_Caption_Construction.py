import os
import json
import random
from functools import partial
from pathlib import Path
import numpy as np
from tqdm import tqdm
import librosa
from chord_extractor.extractors import Chordino
from essentia.standard import MonoLoader, KeyExtractor
from BeatNet.BeatNet import BeatNet
import warnings
warnings.filterwarnings("ignore")


"""
Tempo and Beat Structure Analysis
"""
beat_estimator = BeatNet(1, mode='offline', inference_model='DBN', plot=[], thread=False)

def extract_tempo_and_beats(audio_path, sr=22050):
    """Extract tempo and beat structure using BeatNet"""
    try:
        
        # Extract beats and downbeats using BeatNet
        output = beat_estimator.process(audio_path)
        # output is a 2D array with shape (n_beats, 2)
        # where first column is the beat time (specific number) and second column indicates downbeat (0/1, 1 for downbeat)
        
        beats = output[:, 0]  # Beat times
        downbeats = output[output[:, 1] == 1, 0]  # Downbeat times (where beat=1)
        # 取“第1列为1”的idx的第0列

        # Calculate tempo (measured with BPM)
        if len(beats) > 1:
            beat_intervals = np.diff(beats)
            tempo = 60.0 / np.median(beat_intervals)
        else:
            tempo = 120.0  # Default tempo
        tempo_marking = classify_tempo(tempo)
        time_signature = analyze_time_signature(beats, downbeats)
        return {
            'tempo': round(float(tempo), 2),
            'tempo_marking': tempo_marking,
            'signature': time_signature
        }
    except Exception as e:
        print(f"Error extracting tempo/beats for {audio_path}: {e}")
        return {
            'tempo': 120.0,
            'tempo_marking': 'Moderato',
            'signature': '4/4'
        }
    
def analyze_time_signature(beats, downbeats):
    """Analyze time signature based on beat and downbeat positions"""
    if len(downbeats) < 2 or len(beats) < 4:
        return '4/4'
    
    # Find beats between consecutive downbeats
    measures_beat_counts = []
    measure_analysis = []
    
    for i in range(len(downbeats) - 1):
        downbeat_start = downbeats[i]
        downbeat_end = downbeats[i + 1]
        
        # Count beats between downbeats (including the first downbeat, excluding the last)
        beats_in_measure = np.sum((beats >= downbeat_start) & (beats < downbeat_end))
        measures_beat_counts.append(beats_in_measure)
        
        measure_analysis.append({
            'measure_start': float(downbeat_start),
            'measure_end': float(downbeat_end),
            'beats_in_measure': int(beats_in_measure),
            'duration': float(downbeat_end - downbeat_start)
        })
    
    if not measures_beat_counts:
        return '4/4'
    
    # Analyze the most common beat count per measure
    beat_counts = np.array(measures_beat_counts)
    unique_counts, count_frequencies = np.unique(beat_counts, return_counts=True)
    
    # Find the most common beat count
    most_common_idx = np.argmax(count_frequencies)
    most_common_beats = unique_counts[most_common_idx]
    
    # Map to common time signatures
    time_signature_map = {
        2: '2/4',
        3: '3/4', 
        4: '4/4',
        5: '5/4',
        6: '6/8',  # Could also be 6/4, would need more analysis
        7: '7/8',
        8: '8/8'   # Could be 4/4 with subdivisions
    }
    
    # Handle compound time signatures (like 6/8, 9/8, 12/8)
    if most_common_beats == 6:
        # Additional analysis needed to distinguish 6/8 from 6/4
        # Check if beats have strong-weak-medium pattern typical of 6/8
        signature = '6/8'
    elif most_common_beats == 9:
        signature = '9/8'
    elif most_common_beats == 12:
        signature = '12/8'
    else:
        signature = time_signature_map.get(most_common_beats, f'{most_common_beats}/4')
    
    return signature

def classify_tempo(tempo):
    TEMPO_MARKINGS = [
        (20, 40, "Grave"),
        (40, 60, "Largo"),
        (60, 66, "Larghetto"),
        (66, 76, "Adagio"),
        (76, 108, "Andante"),
        (108, 120, "Moderato"),
        (120, 168, "Allegro"),
        (168, 200, "Presto"),
        (200, float('inf'), "Prestissimo")
    ]
    for low, high, label in TEMPO_MARKINGS:
        if low <= tempo < high:
            return label
    return "Unknown"


"""
Chord Progression Analysis
return: str
"""
def extract_chord_progression(audio_path):
    chordino = Chordino(roll_on=1)
    chord_changes = chordino.extract(audio_path)
    chords = [change.chord for change in chord_changes]
    return '-'.join(chords)

"""
Key Analysis
return: str
"""
def extract_key(audio_path):
    loader = MonoLoader(filename=audio_path, sampleRate=44100)
    audio = loader()  # 返回 NumPy array，shape=(n_samples,)
    extractor = KeyExtractor()
    key, scale, strength = extractor(audio)
    key_label = f"{key} {scale}"
    return key_label

def extract_genre_from_filename(audio_path: str) -> str:
    GENRE_MAPPING = {
        'gBR': 'Break',
        'gPO': 'Pop',
        'gLO': 'Lock',
        'gWA': 'Waack',
        'gMH': 'Middle Hip-hop',
        'gLH': 'LA-style Hip-hop',
        'gHO': 'House',
        'gKR': 'Krump',
        'gJS': 'Street Jazz',
        'gJB': 'Ballet Jazz'
    }
    fname = os.path.basename(audio_path)
    prefix = fname.split('_')[0]
    return GENRE_MAPPING.get(prefix, "any")
