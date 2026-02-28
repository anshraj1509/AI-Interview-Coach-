import cv2
import base64
import openai
from deepface import DeepFace
import numpy as np
from typing import List, Dict, Any

def extract_frames(file_path: str, num_frames: int = 8) -> List[np.ndarray]:
    """
    Extract evenly spaced frames from video file as OpenCV BGR arrays.
    Returns empty list if not a video or failed to open.
    """
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 1:
        cap.release()
        return []

    positions = [int(i * total_frames / num_frames) for i in range(num_frames)]
    frames = []

    for pos in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()
    return frames


def frames_to_base64(frames: List[np.ndarray], max_images: int = 5) -> List[str]:
    """Convert selected cv2 frames to base64 JPEG strings for GPT-4o vision."""
    if not frames:
        return []

    step = max(1, len(frames) // max_images)
    selected = frames[::step][:max_images]

    base64_list = []
    for frame in selected:
        success, buffer = cv2.imencode('.jpg', frame)
        if success:
            base64_list.append(base64.b64encode(buffer).decode('utf-8'))
    return base64_list


def analyze_body_language(base64_frames: List[str]) -> Dict[str, Any]:
    """Use GPT-4o vision to analyze posture, gestures, eye contact etc."""
    if not base64_frames:
        return {
            "body_language_score": 50,
            "body_language_strengths": [],
            "body_language_improvements": [],
            "body_language_summary": "No video frames available."
        }

    prompt = """
You are an expert at reading body language in job interviews.
Analyze the provided video frames for:
- posture (slouching/upright/open/closed)
- hand gestures & fidgeting
- eye contact toward camera
- head movements & nodding
- overall non-verbal confidence level

Return **JSON only**:
{
  "confidence": <int 0-100>,
  "strengths": ["point 1", "point 2", ...],
  "improvements": ["point 1", "point 2", ...],
  "summary": "one short paragraph summary"
}
"""

    content = [{"type": "text", "text": prompt}]
    for img_b64 in base64_frames:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
        })

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": content}],
        temperature=0.55,
        max_tokens=500
    )

    try:
        result = eval(response.choices[0].message.content.strip())
        return {
            "body_language_score": int(result.get("confidence", 60)),
            "body_language_strengths": result.get("strengths", []),
            "body_language_improvements": result.get("improvements", []),
            "body_language_summary": result.get("summary", "")
        }
    except Exception:
        return {
            "body_language_score": 60,
            "body_language_strengths": [],
            "body_language_improvements": [],
            "body_language_summary": "Body language analysis could not be completed."
        }


def analyze_facial_emotions(frames: List[np.ndarray]) -> Dict[str, Any]:
    """Average emotion detection across frames using DeepFace."""
    if not frames:
        return {
            "dominant_emotion": "neutral",
            "emotion_scores": {},
            "emotion_confidence": 50.0
        }

    all_emotions = []
    for frame in frames:
        try:
            analysis = DeepFace.analyze(
                frame,
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )
            emo_dict = analysis[0]['emotion'] if isinstance(analysis, list) else analysis['emotion']
            all_emotions.append(emo_dict)
        except:
            continue

    if not all_emotions:
        return {
            "dominant_emotion": "neutral",
            "emotion_scores": {},
            "emotion_confidence": 50.0
        }

    # Average probabilities
    emotions = list(all_emotions[0].keys())
    avg_scores = {emo: sum(d[emo] for d in all_emotions) / len(all_emotions) for emo in emotions}
    dominant = max(avg_scores, key=avg_scores.get)

    return {
        "dominant_emotion": dominant.capitalize(),
        "emotion_scores": {k.capitalize(): round(v, 1) for k, v in avg_scores.items()},
        "emotion_confidence": round(avg_scores[dominant], 1)
    }s