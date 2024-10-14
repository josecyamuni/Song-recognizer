import streamlit as st
import numpy as np
from scipy import signal
from scipy.io.wavfile import write, read
import sounddevice as sd
import os
import pickle

# ---------- FUNCTION DEFINITIONS ----------
def create_constellation_map(audio, sample_rate):
    sample_window_duration = 0.5  # in seconds
    sample_window_size = int(sample_window_duration * sample_rate)
    sample_window_size += sample_window_size % 2  # Ensure even window size
    num_peaks = 15  # Number of peaks to select per window

    padding = sample_window_size - audio.size % sample_window_size
    padded_audio = np.pad(audio, (0, padding))

    frequencies, times, stft = signal.stft(
        padded_audio, sample_rate, nperseg=sample_window_size, nfft=sample_window_size, return_onesided=True
    )

    constellation_map = []
    for time_idx, window in enumerate(stft.T):
        spectrum = np.abs(window)
        peaks, properties = signal.find_peaks(spectrum, prominence=0, distance=200)
        n_peaks = min(num_peaks, len(peaks))
        prominent_peaks = np.argpartition(properties["prominences"], -n_peaks)[-n_peaks:]
        for peak in peaks[prominent_peaks]:
            frequency = frequencies[peak]
            constellation_map.append([time_idx, frequency])

    return constellation_map

def create_hashes(constellation_map, song_id=None):
    hashes = {}
    upper_frequency = 23_000  # Upper frequency limit in Hz
    frequency_bits = 10

    for idx, (time, freq) in enumerate(constellation_map):
        for other_time, other_freq in constellation_map[idx : idx + 100]:
            time_diff = other_time - time
            if time_diff <= 1 or time_diff > 10:
                continue

            freq_conv = freq / upper_frequency * (2 ** frequency_bits)
            other_freq_conv = other_freq / upper_frequency * (2 ** frequency_bits)
            hash_val = int(freq_conv) | (int(other_freq_conv) << 10) | (int(time_diff) << 20)

            hashes[hash_val] = (time, song_id)
    return hashes

def score_songs(hashes, database):
    song_matches = {}
    for hash_val, (sample_time, _) in hashes.items():
        if hash_val in database:
            match_list = database[hash_val]
            for ref_time, song_index in match_list:
                if song_index not in song_matches:
                    song_matches[song_index] = []
                song_matches[song_index].append((hash_val, sample_time, ref_time))

    song_scores = {}
    for song_index, matches in song_matches.items():
        offset_scores = {}
        for hash_val, sample_time, ref_time in matches:
            delta = ref_time - sample_time
            if delta not in offset_scores:
                offset_scores[delta] = 0
            offset_scores[delta] += 1

        max_score = (0, 0)
        for offset, score in offset_scores.items():
            if score > max_score[1]:
                max_score = (offset, score)

        song_scores[song_index] = max_score

    sorted_scores = sorted(song_scores.items(), key=lambda x: x[1][1], reverse=True)
    return sorted_scores

def record_audio(duration, sample_rate=44100, filename="output.wav"):
    st.write("Recording...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=2, dtype='float64')
    sd.wait()  # Esperar a que termine la grabación
    write(filename, sample_rate, np.int16(audio_data * 32767))  # Guardar como archivo WAV
    st.write(f"Audio guardado como {filename}")
    return filename

# ---------- STREAMLIT APP ----------

st.title("Song Identification App")

# Get the absolute path to the pickle directory
base_dir = os.path.dirname(os.path.abspath(__file__))
pickle_dir = os.path.join(base_dir, "database_pickles")

# Define database file paths using absolute paths
database_file = os.path.join(pickle_dir, "database.pickle")
song_index_file = os.path.join(pickle_dir, "song_index.pickle")

# Load database if exists
if os.path.exists(database_file) and os.path.exists(song_index_file):
    with open(database_file, 'rb') as db_file:
        database = pickle.load(db_file)

    with open(song_index_file, 'rb') as idx_file:
        song_index = pickle.load(idx_file)

    st.success("Database loaded successfully!")

    # Option to either upload a file or record audio
    st.write("Select an option:")
    option = st.radio("", ("Upload an audio file", "Record audio"))

    if option == "Upload an audio file":
        uploaded_file = st.file_uploader("Upload an audio file (.wav)", type=["wav"])

        if uploaded_file is not None:
            st.audio(uploaded_file)

            # Read the uploaded file as a WAV file
            sample_rate, audio_input = read(uploaded_file)

            # If the audio is stereo, take the first channel
            if len(audio_input.shape) > 1:
                audio_input = audio_input[:, 0]

            # Generate the constellation map and hashes
            constellation_map = create_constellation_map(audio_input, sample_rate)
            hashes = create_hashes(constellation_map)

            # Compare hashes with the database
            scores = score_songs(hashes, database)

            if scores:
                song_id, (offset, match_score) = scores[0]
                song_name = os.path.basename(song_index[song_id])
                song_name = os.path.splitext(song_name)[0]

                st.subheader(f"Identified Song: {song_name.replace('-', ' ')}")
                st.write(f"Match Score: {match_score}")
            else:
                st.error("No matching song found in the database.")

    elif option == "Record audio":
        # Input duration for recording
        duration = st.slider("Select duration of recording (seconds)", 1, 10, 5)

        if st.button("Record"):
            # Record audio for the selected duration
            recorded_file = record_audio(duration, filename="recorded_output.wav")
            st.audio(recorded_file)

            # Process the recorded file
            sample_rate, audio_input = read(recorded_file)

            # If the audio is stereo, take the first channel
            if len(audio_input.shape) > 1:
                audio_input = audio_input[:, 0]

            # Generate the constellation map and hashes
            constellation_map = create_constellation_map(audio_input, sample_rate)
            hashes = create_hashes(constellation_map)

            # Compare hashes with the database
            scores = score_songs(hashes, database)

            if scores:
                song_id, (offset, match_score) = scores[0]
                song_name = os.path.basename(song_index[song_id])
                song_name = os.path.splitext(song_name)[0]

                st.subheader(f"Identified Song: {song_name.replace('-', ' ')}")
                st.write(f"Match Score: {match_score}")
            else:
                st.error("No matching song found in the database.")
else:
    st.warning("Database not found. Please ensure the database is created and available.")
