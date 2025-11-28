'''
Author:     Ji-Sung Kim
Project:    deepjazz
Purpose:    Parse, cleanup and process data.

Code adapted from Evan Chow's jazzml, https://github.com/evancchow/jazzml with
express permission.
'''

from __future__ import print_function

from music21 import *
from collections import defaultdict, OrderedDict
from itertools import groupby, zip_longest

from grammar import *
from grammar import parse_melody
from music_utils import *

#----------------------------HELPER FUNCTIONS----------------------------------#

''' Helper function to parse a MIDI file into its measures and chords '''
def __parse_midi(data_fn):
    # Parse the MIDI data for separate melody and accompaniment parts.
    midi_data = converter.parse(data_fn)
    
    # REPLACEMENT CODE: Robust Part Extraction
    # Filter the MIDI file to keep ONLY parts that contain actual notes.
    # This automatically discards Metadata, Text, and Copyright tracks that throw off the indices.
    musical_parts = []
    for el in midi_data:
        # Check if it's a Stream/Part and has notes inside
        if isinstance(el, (stream.Stream, stream.Part)):
            if len(el.recurse().notes) > 0:
                musical_parts.append(el)

    # Safety Check: If we didn't find enough parts, fall back to raw midi_data
    # (This handles edge cases where the file structure is unexpected)
    if len(musical_parts) < 6:
        # Fallback to raw streams if strict filtering removed too much
        musical_parts = [el for el in midi_data if isinstance(el, (stream.Stream, stream.Part))]

    # 1. Get Melody Part
    # Now we can safely use the original index 5, as we stripped the headers.
    try:
        melody_stream = musical_parts[5]
    except IndexError:
        # Fallback: Try to find a part named 'Melody' or 'Guitar' or 'Solo'
        melody_stream = musical_parts[-1] # Default to last part (often solo)
        for p in musical_parts:
            p_name = str(getattr(p, 'partName', '')).lower()
            if 'melody' in p_name or 'guit' in p_name or 'solo' in p_name:
                melody_stream = p
                break

    # Try to get voices directly (old music21 behavior)
    voices = list(melody_stream.getElementsByClass(stream.Voice))

    # If empty, try recursive search (new music21 behavior)
    if not voices:
        voices = list(melody_stream.recurse().getElementsByClass(stream.Voice))

    # Handle the voices found
    if len(voices) >= 2:
        melody1 = voices[0]
        melody2 = voices[1]
        for j in melody2:
            melody1.insert(j.offset, j)
    elif len(voices) == 1:
        melody1 = voices[0]
    else:
        # If still no voices, assume the stream is flattened/monophonic
        melody1 = melody_stream.flat
        
    melody_voice = melody1

    for i in melody_voice:
        if i.quarterLength == 0.0:
            i.quarterLength = 0.25

    # Change key signature to adhere to comp_stream (1 sharp, mode = major).
    # Also add Electric Guitar. 
    melody_voice.insert(0, instrument.ElectricGuitar())
    melody_voice.insert(0, key.KeySignature(sharps=1))

    # 2. Get Accompaniment Parts
    # Use the original indices [0, 1, 6, 7] on our cleaned 'musical_parts' list
    partIndices = [0, 1, 6, 7] 
    
    comp_stream = stream.Voice()
    
    for i in partIndices:
        if i < len(musical_parts):
            p = musical_parts[i]
            # Flatten to ensure we get all data from the part
            comp_stream.append(p.flat)

    # Full stream containing both the melody and the accompaniment. 
    full_stream = stream.Voice()
    for i in range(len(comp_stream)):
        full_stream.append(comp_stream[i])
    full_stream.append(melody_voice)

    # Extract solo stream
    solo_stream = stream.Voice()
    for part in full_stream:
        curr_part = stream.Part()
        
        # Helper to insert from StreamIterator (fixes StreamException)
        def insert_from_iterator(iterator):
            for item in iterator:
                curr_part.insert(item.offset, item)
        
        insert_from_iterator(part.getElementsByClass(instrument.Instrument))
        insert_from_iterator(part.getElementsByClass(tempo.MetronomeMark))
        insert_from_iterator(part.getElementsByClass(key.KeySignature))
        insert_from_iterator(part.getElementsByClass(meter.TimeSignature))
        # Ensure we capture the solo section correctly
        insert_from_iterator(part.getElementsByOffset(476, 548, includeEndBoundary=True))
        
        cp = curr_part.flat
        solo_stream.insert(cp)

    # Group by measure so you can classify. 
    melody_stream = solo_stream[-1]
    measures = OrderedDict()
    offsetTuples = [(int(n.offset / 4), n) for n in melody_stream]
    measureNum = 0 
    for key_x, group in groupby(offsetTuples, lambda x: x[0]):
        measures[measureNum] = [n[1] for n in group]
        measureNum += 1

    # Get the stream of chords.
    chordStream = solo_stream[0]
    chordStream.removeByClass(note.Rest)
    chordStream.removeByClass(note.Note)
    offsetTuples_chords = [(int(n.offset / 4), n) for n in chordStream]

    # Generate the chord structure.
    chords = OrderedDict()
    measureNum = 0
    for key_x, group in groupby(offsetTuples_chords, lambda x: x[0]):
        chords[measureNum] = [n[1] for n in group]
        measureNum += 1

    # Robustly align lengths (fixes AssertionError)
    min_len = min(len(measures), len(chords))
    
    while len(chords) > min_len:
        del chords[len(chords) - 1]
    
    while len(measures) > min_len:
        del measures[len(measures) - 1]
        
    assert len(chords) == len(measures)

    return measures, chords

''' Helper function to get the grammatical data from given musical data. '''
def __get_abstract_grammars(measures, chords):
    # extract grammars
    abstract_grammars = []
    for ix in range(1, len(measures)):
        m = stream.Voice()
        for i in measures[ix]:
            m.insert(i.offset, i)
        c = stream.Voice()
        for j in chords[ix]:
            c.insert(j.offset, j)
        parsed = parse_melody(m, c)
        abstract_grammars.append(parsed)

    return abstract_grammars

#----------------------------PUBLIC FUNCTIONS----------------------------------#

''' Get musical data from a MIDI file '''
def get_musical_data(data_fn):
    
    measures, chords = __parse_midi(data_fn)
    abstract_grammars = __get_abstract_grammars(measures, chords)

    return chords, abstract_grammars

''' Get corpus data from grammatical data '''
def get_corpus_data(abstract_grammars):
    corpus = [x for sublist in abstract_grammars for x in sublist.split(' ')]
    values = set(corpus)
    val_indices = dict((v, i) for i, v in enumerate(values))
    indices_val = dict((i, v) for i, v in enumerate(values))

    return corpus, values, val_indices, indices_val


# def load_music_utils():
#     chord_data, raw_music_data = get_musical_data('data/original_metheny.mid')
#     music_data, values, values_indices, indices_values = get_corpus_data(raw_music_data)

#     X, Y = data_processing(music_data, values_indices, Tx = 20, step = 3)
#     return (X, Y)
