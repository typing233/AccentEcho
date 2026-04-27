# AccentEcho

**AccentEcho** is a browser-based accent practice tool. Upload or record a short reference audio clip, enter a sentence you want to practice, and the app will synthesize that sentence in a voice that reflects the pitch and speaking-rate characteristics extracted from your reference audio.

![AccentEcho UI](https://github.com/user-attachments/assets/0dcbd463-f5e5-42da-9d53-9f8c4271fd3e)

## Features

| Feature | Details |
|---|---|
| Reference audio | Upload a file (WAV / MP3 / OGG / M4A …) **or** record directly in the browser (15–30 s recommended) |
| Voice analysis | Extracts pitch (F0), pitch range, speaking rate, and energy from the reference clip |
| Accent synthesis | Synthesises the target text and applies pitch-shifting + time-stretching to approximate the reference voice |
| Playback speed | Slider from **0.5×** to **1.5×** for slow-follow practice |
| Loop playback | One-click toggle |
| Download | Download the synthesised WAV file |
| Offline TTS | Powered by **eSpeak-NG** — no internet connection required for synthesis |

## Tech stack

- **Backend**: Python 3.12 · Flask · librosa · soundfile · NumPy · SciPy  
- **TTS**: eSpeak-NG (offline, supports Mandarin Chinese & English)  
- **Frontend**: Vanilla HTML / CSS / JavaScript · Web Audio API · MediaRecorder API

## Quick start

### Prerequisites

```bash
# Ubuntu / Debian
sudo apt-get install espeak-ng espeak-ng-data

# macOS (Homebrew)
brew install espeak-ng
```

### Install Python dependencies

```bash
pip install -r requirements.txt
```

### Run the server

```bash
python app.py
# Open http://localhost:5000 in your browser
```

## Usage

1. **Upload or record** a 15–30 second reference audio in your target accent  
2. **Enter the sentence** you want to practice  
3. Click **合成口音语音** (Synthesise) — takes ~20–40 s on a typical CPU  
4. Use the **speed slider** (0.5×–1.5×) to slow down for follow-along practice  
5. Toggle **循环** (Loop) for repeated playback, then **download** the WAV when done
