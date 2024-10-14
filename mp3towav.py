import os
from pydub import AudioSegment

# Tests directory
directory = "identify_songs/"

for filename in os.listdir(directory):
    if filename.endswith(".mp3"):

        mp3_path = os.path.join(directory, filename)
        
        audio = AudioSegment.from_mp3(mp3_path)
        
        wav_filename = filename.replace(".mp3", ".wav")
        wav_path = os.path.join(directory, wav_filename)
        
        audio.export(wav_path, format="wav")
        
        # Eliminar el archivo MP3 original
        os.remove(mp3_path)
        
        print(f"Convertido y eliminado: {filename} -> {wav_filename}")
