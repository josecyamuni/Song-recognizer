import numpy as np
from scipy import signal
from scipy.io.wavfile import read
from glob import glob
import os
import pickle

# ---------- FUNCTION DEFINITIONS ----------
def create_constellation_map(audio, sample_rate):
    """
    Creates a constellation map from an audio signal using STFT and peak detection.

    Parameters:
    - audio: The input audio signal.
    - sample_rate: The sampling frequency of the audio signal.

    Returns:
    - A list of time-frequency pairs representing the constellation map.
    """
    sample_window_duration = 0.5  # in seconds
    sample_window_size = int(sample_window_duration * sample_rate)
    sample_window_size += sample_window_size % 2  # Ensure even window size
    num_peaks = 15  # Number of peaks to select per window

    # Padding audio to fit full windows
    padding = sample_window_size - audio.size % sample_window_size
    padded_audio = np.pad(audio, (0, padding))

    # Compute Short-Time Fourier Transform (STFT)
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
    """
    Creates hashes from a constellation map to uniquely identify time-frequency pairs.

    Parameters:
    - constellation_map: The time-frequency pairs representing the constellation map.
    - song_id: An optional song identifier to associate with the hashes.

    Returns:
    - A dictionary where each key is a hash and the value is a tuple (time, song_id).
    """
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

def score_songs(hashes):
    """
    Scores songs by comparing the hashes of the input audio with the database.

    Parameters:
    - hashes: The hashes generated from the input audio.

    Returns:
    - A list of songs ranked by their match score.
    """
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

    # Sort by highest score
    sorted_scores = sorted(song_scores.items(), key=lambda x: x[1][1], reverse=True)

    return sorted_scores

# ---------- MAIN SCRIPT ----------

if __name__ == "__main__":
    # Crear el directorio si no existe
    pickle_dir = "database_pickles"
    if not os.path.exists(pickle_dir):
        os.makedirs(pickle_dir)

    # ---------- Verificar si la base de datos y song_index ya existen ----------
    database_file = os.path.join(pickle_dir, "database.pickle")
    song_index_file = os.path.join(pickle_dir, "song_index.pickle")


    if os.path.exists(database_file) and os.path.exists(song_index_file):
        print("Loading existing database and song index...")

        with open(database_file, 'rb') as db_file:
            database = pickle.load(db_file)

        with open(song_index_file, 'rb') as idx_file:
            song_index = pickle.load(idx_file)

    else:
        print("No database found, creating new database...")

        # ---------- INPUT: Base de datos de canciones ----------
        song_files = glob('database_songs/*.wav')  

        song_index = {}
        database = {}

        # Procesar cada canción en la base de datos para generar los hashes
        for idx, file in enumerate(sorted(song_files)):
            song_index[idx] = file
            sample_rate, audio_input = read(file)

            # Verificar si el audio es estéreo (2 canales) o monoaural (1 canal)
            if len(audio_input.shape) > 1:
                audio_input = audio_input[:, 0]  

            constellation_map = create_constellation_map(audio_input, sample_rate)
            hashes = create_hashes(constellation_map, idx)

            for hash_val, time_idx_tuple in hashes.items():
                if hash_val not in database:
                    database[hash_val] = []
                database[hash_val].append(time_idx_tuple)

        # Guardar la base de datos y el índice de canciones para uso futuro
        with open(database_file, 'wb') as db_file:
            pickle.dump(database, db_file, pickle.HIGHEST_PROTOCOL)

        with open(song_index_file, 'wb') as idx_file:
            pickle.dump(song_index, idx_file, pickle.HIGHEST_PROTOCOL)

    # ---------- INPUT: Canciones a identificar ----------
    identify_files = glob('identify_songs/*.wav')

    for identify_file in identify_files:
        sample_rate, audio_input = read(identify_file)

        # Verificar si el audio es estéreo (2 canales) o monoaural (1 canal)
        if len(audio_input.shape) > 1:
            audio_input = audio_input[:, 0] 

        constellation_map = create_constellation_map(audio_input, sample_rate)
        hashes = create_hashes(constellation_map)

        # Comparar los hashes generados con los de la base de datos
        scores = score_songs(hashes)
        print(f"Results for {identify_file}:")
        song_id, (offset, match_score) = scores[0]
        song_name = os.path.basename(song_index[song_id])
        song_name = os.path.splitext(song_name)[0]
        print(f"Your song is {song_name.replace('-', ' ')}!!!")
