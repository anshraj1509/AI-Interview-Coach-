import streamlit as st
import openai
import re
import os
from pydub import AudioSegment
import cv2
import base64
from deepface import DeepFace
import numpy as np

# ────────────────────────────────────────────────
#   CONFIG
# ────────────────────────────────────────────────

openai.api_key = st.secrets.get("OPENAI_API_KEY", "your-api-key-here").strip()

FILLER_WORDS = ['um', 'uh', 'like', 'you know', 'so', 'basically', 'right', 'okay', 'alright']

# ────────────────────────────────────────────────
#   HELPERS
# ────────────────────────────────────────────────

def transcribe_audio(file_path):
    with open(file_path, "rb") as f:
        transcript = openai.Audio.transcribe(
            model="whisper-1",
            file=f,
            response_format="text"
        )
    return transcript

def extract_frames(file_path, num_frames=8):
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    positions = [int(i * total / num_frames) for i in range(num_frames)]
    frames = []
    for pos in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames

def frames_to_base64(frames, max_images=5):
    if not frames:
        return []
    step = max(1, len(frames) // max_images)
    selected = frames[::step][:max_images]
    return [
        base64.b64encode(cv2.imencode('.jpg', f)[1]).decode('utf-8')
        for f in selected
    ]

def analyze_body_language(base64_frames):
    if not base64_frames:
        return {"body_language_score": 50, "body_language_strengths": [], "body_language_improvements": [], "body_language_summary": "No video available."}

    prompt = """
You are an expert interview coach. Analyze these video frames for:
- posture, gestures, eye contact, facial expressions, fidgeting
- overall body language confidence (0–100)

Return **JSON only**:
{
  "confidence": number,
  "strengths": ["..."],
  "improvements": ["..."],
  "summary": "short paragraph"
}
"""
    content = [{"type": "text", "text": prompt}]
    for b64 in base64_frames:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

    resp = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": content}],
        temperature=0.6
    )
    try:
        d = eval(resp.choices[0].message.content.strip())
        return {
            "body_language_score": d.get("confidence", 60),
            "body_language_strengths": d.get("strengths", []),
            "body_language_improvements": d.get("improvements", []),
            "body_language_summary": d.get("summary", "")
        }
    except:
        return {"body_language_score": 60, "body_language_strengths": [], "body_language_improvements": [], "body_language_summary": "Analysis failed."}

def analyze_facial_emotions(cv2_frames):
    if not cv2_frames:
        return {"dominant_emotion": "neutral", "emotion_scores": {}, "emotion_confidence": 50}

    emotions = []
    for frame in cv2_frames:
        try:
            r = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, silent=True)
            emotions.append(r[0]['emotion'] if isinstance(r, list) else r['emotion'])
        except:
            pass

    if not emotions:
        return {"dominant_emotion": "neutral", "emotion_scores": {}, "emotion_confidence": 50}

    avg = {k: sum(d[k] for d in emotions)/len(emotions) for k in emotions[0]}
    dom = max(avg, key=avg.get)
    return {
        "dominant_emotion": dom.capitalize(),
        "emotion_scores": {k.capitalize(): round(v,1) for k,v in avg.items()},
        "emotion_confidence": round(avg[dom],1)
    }

def analyze_transcript(text, duration_sec):
    words = re.split(r'\s+', text.lower().strip())
    word_count = len([w for w in words if w.strip()])
    fillers = sum(1 for w in words if w.strip('.,!?') in FILLER_WORDS)
    wpm = round(word_count / duration_sec * 60) if duration_sec > 0 else 0

    prompt = f"""Analyze this interview response:
{text}

Return **JSON only**:
{{
  "star_details": "S: ?/5  T: ?/5  A: ?/5  R: ?/5 (Total: ?/20)",
  "star_score": int,
  "verbal_confidence": int,     // 0-100 based on clarity, fillers ({fillers}), wpm ({wpm})
  "overall_score": int,         // 0-100
  "strengths": ["..."],
  "improvements": ["..."],
  "rewritten": "full rewritten STAR answer",
  "roadmap": ["step 1", "step 2", ...]
}}
"""
    try:
        r = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        d = eval(r.choices[0].message.content.strip())
    except:
        d = {
            "star_details": "S:3/5 T:3/5 A:4/5 R:3/5 (Total:13/20)",
            "star_score": 13,
            "verbal_confidence": 70,
            "overall_score": 75,
            "strengths": ["Good structure", "Relevant example"],
            "improvements": ["Reduce fillers", "Quantify results"],
            "rewritten": "[Simulated perfect answer]",
            "roadmap": ["Practice STAR format", "Record & review"]
        }

    return {
        "transcript": text,
        "word_count": word_count,
        "filler_count": fillers,
        "wpm": wpm,
        "duration": duration_sec,
        **d
    }

# ────────────────────────────────────────────────
#   STREAMLIT UI
# ────────────────────────────────────────────────

st.set_page_config(page_title="AI Interview Coach", layout="wide")
st.title("🎤 AI Interview Coach  (OpenAI + DeepFace)")
st.caption("Transcribes • Analyzes STAR • Fillers • Speed • Body Language • Facial Emotions")

input_mode = st.radio("Input", ["Paste transcript", "Upload audio/video"])

transcript = ""
duration = 0
is_video = False

if input_mode == "Paste transcript":
    transcript = st.text_area("Your interview answer", height=140)
    duration = st.number_input("Duration (seconds)", 10, 600, 60)
else:
    file = st.file_uploader("Upload .mp3 .wav .m4a .mp4", type=["mp3","wav","m4a","mp4"])
    if file:
        is_video = file.name.lower().endswith(".mp4")

if st.button("Analyze →", type="primary"):
    if "your-api-key-here" in openai.api_key:
        st.error("Please set a real OpenAI API key in Streamlit secrets.")
        st.stop()

    with st.spinner("Transcribing + analyzing video + emotions..."):
        results = {}
        body = {}
        emotion = {}

        if input_mode == "Upload audio/video" and file:
            path = f"temp_{file.name}"
            with open(path, "wb") as f:
                f.write(file.read())

            audio = AudioSegment.from_file(path)
            duration = len(audio) / 1000
            transcript = transcribe_audio(path)

            if is_video:
                cv_frames = extract_frames(path)
                b64_frames = frames_to_base64(cv_frames)
                body = analyze_body_language(b64_frames)
                emotion = analyze_facial_emotions(cv_frames)

            os.remove(path)

        if not transcript.strip():
            st.error("No text to analyze.")
            st.stop()

        trans_results = analyze_transcript(transcript, duration)
        results = {**trans_results, **body, **emotion}

    # ──── DISPLAY ───────────────────────────────────────

    st.header("Results")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall", f"{results.get('overall_score', 0)}/100")
    c2.metric("STAR", f"{results.get('star_score', 0)}/20")
    c3.metric("Body Language", f"{results.get('body_language_score', 50)}/100")
    c4.metric("Facial Emotion", f"{results.get('emotion_confidence', 50)}% – {results.get('dominant_emotion', 'neutral').title()}")

    if results.get('emotion_scores'):
        st.subheader("Emotion Distribution")
        st.bar_chart(results['emotion_scores'])

    st.subheader("Body Language Summary")
    st.info(results.get("body_language_summary", "—"))

    st.subheader("Key Metrics")
    st.write(f"• WPM: **{results.get('wpm', 0)}**")
    st.write(f"• Fillers: **{results.get('filler_count', 0)}**")
    st.write(f"• Verbal Confidence: **{results.get('verbal_confidence', 0)}%**")
    st.write(f"• Duration: **{results.get('duration', 0):.1f} s**")

    with st.expander("Transcript"):
        st.text_area("", results["transcript"], height=120)

    colA, colB = st.columns(2)
    with colA:
        st.subheader("Strengths")
        for x in results.get("strengths", []):
            st.success(f"✓ {x}")
    with colB:
        st.subheader("Improvements")
        for x in results.get("improvements", []):
            st.warning(f"⚠ {x}")

    if results.get("body_language_strengths"):
        st.subheader("Body Language Strengths")
        for x in results["body_language_strengths"]:
            st.success(f"✓ {x}")

    st.subheader("Rewritten Perfect Answer")
    st.markdown(results.get("rewritten", "—"))

    st.subheader("Improvement Roadmap")
    for i, step in enumerate(results.get("roadmap", []), 1):
        st.write(f"{i}. {step}")

st.markdown("---")
st.caption("Built with Streamlit • OpenAI (Whisper + GPT-4o) • DeepFace • OpenCV • pydub")