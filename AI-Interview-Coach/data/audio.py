import openai
from pydub import AudioSegment
import os

def transcribe_audio(file_path: str) -> str:
    """
    Transcribe audio file using OpenAI Whisper API.
    Returns the text transcript.
    """
    with open(file_path, "rb") as audio_file:
        transcript = openai.Audio.transcribe(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )
    return transcript.strip()


def get_audio_duration(file_path: str) -> float:
    """
    Get duration of audio/video file in seconds using pydub.
    Works for mp3, wav, m4a, mp4, etc.
    """
    audio = AudioSegment.from_file(file_path)
    return len(audio) / 1000.0  # seconds


def process_uploaded_file(uploaded_file) -> tuple[str, float, str]:
    """
    Save uploaded file temporarily, transcribe + get duration.
    Returns: (transcript, duration_sec, temp_path)
    Caller is responsible for os.remove(temp_path)
    """
    if not uploaded_file:
        return "", 0.0, ""

    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    try:
        transcript = transcribe_audio(temp_path)
        duration = get_audio_duration(temp_path)
        return transcript, duration, temp_path
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise RuntimeError(f"Audio processing failed: {e}")