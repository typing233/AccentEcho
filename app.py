"""
AccentEcho – accent-practice web application.

Routes
------
GET  /                  – serve main UI
POST /analyze           – receive WAV, extract voice characteristics
POST /synthesize        – generate TTS + apply voice transformation
GET  /audio/<audio_id>  – stream synthesised WAV
GET  /download/<audio_id> – download synthesised WAV as attachment
"""

import os
import uuid
import subprocess
import shutil

import numpy as np
import soundfile as sf
import librosa
from flask import Flask, render_template, request, jsonify, send_file

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB

_BASE = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(_BASE, "uploads")
OUTPUT_FOLDER = os.path.join(_BASE, "outputs")

for _d in (UPLOAD_FOLDER, OUTPUT_FOLDER):
    os.makedirs(_d, exist_ok=True)


# ---------------------------------------------------------------------------
# Audio analysis
# ---------------------------------------------------------------------------

def analyze_voice(audio_path: str) -> dict:
    """Return a dict of perceptual voice characteristics from *audio_path*."""
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    # Fundamental frequency (F0) via pYIN
    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=float(librosa.note_to_hz("C2")),
        fmax=float(librosa.note_to_hz("C7")),
        sr=sr,
    )
    valid_f0 = f0[voiced_flag & ~np.isnan(f0)]
    if len(valid_f0) > 5:
        mean_pitch = float(np.mean(valid_f0))
        pitch_std = float(np.std(valid_f0))
        pitch_range = float(np.max(valid_f0) - np.min(valid_f0))
    else:
        mean_pitch, pitch_std, pitch_range = 150.0, 30.0, 80.0

    # Approximate syllable/onset rate
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time")
    speaking_rate = len(onsets) / duration if duration > 0 else 4.0

    # Energy & spectral brightness
    rms = float(np.mean(librosa.feature.rms(y=y)))
    spec_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))

    return {
        "duration": round(duration, 2),
        "mean_pitch": round(mean_pitch, 1),
        "pitch_std": round(pitch_std, 1),
        "pitch_range": round(pitch_range, 1),
        "speaking_rate": round(speaking_rate, 2),
        "rms_energy": round(rms, 5),
        "spectral_centroid": round(spec_centroid, 1),
    }


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# TTS via espeak-ng subprocess (offline, no network required)
# ---------------------------------------------------------------------------

#: espeak-ng executable path
_ESPEAK = shutil.which("espeak-ng") or "espeak-ng"

#: Language code → espeak-ng voice name
_VOICE_MAP: dict[str, str] = {
    "zh-TW": "cmn",   # Mandarin Chinese
    "zh-CN": "cmn",
    "en":    "en",
}


def _tts_generate(text: str, voice: str, out_path: str, wpm: int = 150) -> None:
    """Synthesise *text* to a WAV file using espeak-ng."""
    cmd = [_ESPEAK, "-v", voice, "-s", str(wpm), "-w", out_path, text]
    result = subprocess.run(cmd, capture_output=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(
            f"espeak-ng failed (rc={result.returncode}): {result.stderr.decode()}"
        )


# ---------------------------------------------------------------------------
# Voice-aware TTS synthesis
# ---------------------------------------------------------------------------

def synthesize_with_accent(text: str, lang: str, characteristics: dict) -> str:
    """
    Generate TTS for *text* then shift pitch / stretch time to approximate
    the voice described by *characteristics*.  Returns the output audio_id.
    """
    out_id = str(uuid.uuid4())
    tts_tmp = os.path.join(OUTPUT_FOLDER, out_id + "_tts.wav")
    out_wav = os.path.join(OUTPUT_FOLDER, out_id + ".wav")

    # 1. Base TTS using espeak-ng (offline, no threading issues)
    voice = _VOICE_MAP.get(lang, "en")
    _tts_generate(text, voice, tts_tmp)

    # 2. Load TTS audio
    y, sr = librosa.load(tts_tmp, sr=22050, mono=True)
    try:
        os.remove(tts_tmp)
    except OSError:
        pass

    # 3. Estimate TTS pitch
    f0, voiced_flag, _ = librosa.pyin(y, fmin=80.0, fmax=500.0, sr=sr)
    valid_f0 = f0[voiced_flag & ~np.isnan(f0)]
    tts_pitch = float(np.mean(valid_f0)) if len(valid_f0) > 5 else 200.0

    # 4. Pitch shift to match reference speaker
    target_pitch = characteristics.get("mean_pitch", 150.0)
    if tts_pitch > 0 and target_pitch > 0:
        semitones = float(np.clip(12 * np.log2(target_pitch / tts_pitch), -12, 12))
    else:
        semitones = 0.0

    if abs(semitones) > 0.5:
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=semitones)

    # 5. Time-stretch to approximate reference speaking rate
    ref_rate = characteristics.get("speaking_rate", 4.5)
    tts_rate = 4.5  # approximate default espeak rate (onsets/sec)
    if ref_rate > 0:
        ratio = float(np.clip(tts_rate / ref_rate, 0.7, 1.4))
        if abs(ratio - 1.0) > 0.05:
            y = librosa.effects.time_stretch(y, rate=ratio)


    # 6. Save
    sf.write(out_wav, y, sr)
    return out_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_uuid(value: str) -> bool:
    try:
        uuid.UUID(value)
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "audio" not in request.files:
        return jsonify({"error": "未提供音频文件"}), 400

    f = request.files["audio"]
    if not f.filename:
        return jsonify({"error": "文件为空"}), 400

    ref_id = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_FOLDER, ref_id + ".wav")
    f.save(save_path)

    try:
        chars = analyze_voice(save_path)
        return jsonify({"success": True, "ref_id": ref_id, "characteristics": chars})
    except Exception as exc:
        if os.path.exists(save_path):
            os.remove(save_path)
        return jsonify({"error": str(exc)}), 500


@app.route("/synthesize", methods=["POST"])
def synthesize():
    data = request.get_json(force=True) or {}
    text = (data.get("text") or "").strip()
    ref_id = (data.get("ref_id") or "").strip()

    if not text:
        return jsonify({"error": "文本不能为空"}), 400

    # Detect language (CJK → zh-TW, otherwise en)
    lang = "zh-TW" if any("\u4e00" <= c <= "\u9fff" for c in text) else "en"

    # Load reference characteristics or use safe defaults
    chars: dict
    if ref_id and _valid_uuid(ref_id):
        ref_path = os.path.join(UPLOAD_FOLDER, ref_id + ".wav")
        if os.path.isfile(ref_path):
            try:
                chars = analyze_voice(ref_path)
            except Exception:
                chars = {"mean_pitch": 150.0, "pitch_std": 30.0, "speaking_rate": 4.5}
        else:
            chars = {"mean_pitch": 150.0, "pitch_std": 30.0, "speaking_rate": 4.5}
    else:
        chars = {"mean_pitch": 150.0, "pitch_std": 30.0, "speaking_rate": 4.5}

    try:
        audio_id = synthesize_with_accent(text, lang, chars)
        return jsonify({"success": True, "audio_id": audio_id})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/audio/<audio_id>")
def serve_audio(audio_id):
    if not _valid_uuid(audio_id):
        return jsonify({"error": "无效的音频 ID"}), 400
    path = os.path.join(OUTPUT_FOLDER, audio_id + ".wav")
    if not os.path.isfile(path):
        return jsonify({"error": "找不到文件"}), 404
    return send_file(path, mimetype="audio/wav")


@app.route("/download/<audio_id>")
def download_audio(audio_id):
    if not _valid_uuid(audio_id):
        return jsonify({"error": "无效的音频 ID"}), 400
    path = os.path.join(OUTPUT_FOLDER, audio_id + ".wav")
    if not os.path.isfile(path):
        return jsonify({"error": "找不到文件"}), 404
    return send_file(
        path,
        as_attachment=True,
        download_name="accent_echo.wav",
        mimetype="audio/wav",
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
