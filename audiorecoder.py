import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write

def record_audio(duration, sample_rate=44100, filename="output.wav"):
    """
    Graba audio durante un tiempo especificado y guarda el archivo como WAV.
    
    Args:
        duration (int): Duración de la grabación en segundos.
        sample_rate (int): Frecuencia de muestreo, por defecto 44100.
        filename (str): Nombre del archivo WAV de salida.
    """
    print("Recording...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=2, dtype='float64')
    sd.wait()  # Esperar a que termine la grabación
    write(filename, sample_rate, np.int16(audio_data * 32767))  # Guardar como archivo WAV
    print(f"Audio guardado como {filename}")

# Parámetros de grabación
duration = 20  # Duración de la grabación en segundos
wav_file = "output.wav"

# Grabar y guardar como WAV
record_audio(duration, filename=wav_file)