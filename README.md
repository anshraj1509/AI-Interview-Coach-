# AI Interview Coach

Advanced mock interview analyzer using **Streamlit**, **OpenAI**, and **DeepFace**.

**Features**
- Audio/video upload or paste transcript
- Whisper transcription
- STAR method scoring
- Filler words, WPM, verbal confidence
- Body language analysis (GPT-4o vision)
- Facial emotion detection (DeepFace – happy, sad, neutral, angry, etc.)
- Rewritten perfect answer + roadmap

## Quick Start

```bash
# 1. Clone / download
git clone https://github.com/YOUR-USERNAME/AI-Interview-Coach.git
cd AI-Interview-Coach

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set OpenAI key (recommended: use .streamlit/secrets.toml)
#    [secrets]
#    OPENAI_API_KEY = "sk-..."

# 4. Run
streamlit run app.py
