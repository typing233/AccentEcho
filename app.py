"""
AccentEcho – accent-practice web application.

Routes
------
GET  /                  – serve main UI
POST /analyze           – receive WAV, extract voice characteristics
POST /synthesize        – generate TTS + apply voice transformation
POST /synthesize-blend  – TTS with blend coefficient between two voices
GET  /audio/<audio_id>  – stream synthesised WAV
GET  /download/<audio_id> – download synthesised WAV as attachment
POST /compare           – compare voices for scoring
POST /llm/config        – save LLM configuration
POST /llm/test          – test LLM API connectivity
GET  /fingerprint/<id>  – get voice fingerprint data
GET  /fingerprint/compare – compare two fingerprints
GET  /fingerprint/export/<id> – export fingerprint as PNG
"""

import os
import json
import uuid
import logging
import subprocess
import shutil
import base64
import io

import numpy as np
import soundfile as sf
import librosa
from scipy.spatial.distance import cosine
from flask import Flask, render_template, request, jsonify, send_file
import requests

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Circle
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not installed, fingerprint export disabled")

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB

_BASE = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(_BASE, "uploads")
OUTPUT_FOLDER = os.path.join(_BASE, "outputs")
FINGERPRINT_FOLDER = os.path.join(_BASE, "fingerprints")

for _d in (UPLOAD_FOLDER, OUTPUT_FOLDER, FINGERPRINT_FOLDER):
    os.makedirs(_d, exist_ok=True)

logger = logging.getLogger(__name__)

# LLM Configuration storage (in-memory with file backup)
_LLM_CONFIG = {
    "base_url": "",
    "api_key": "",
    "model_name": "",
    "connected": False
}
_LLM_CONFIG_FILE = os.path.join(_BASE, ".llm_config.json")

def _load_llm_config():
    global _LLM_CONFIG
    if os.path.exists(_LLM_CONFIG_FILE):
        try:
            with open(_LLM_CONFIG_FILE, "r") as f:
                saved = json.load(f)
                _LLM_CONFIG.update(saved)
        except Exception:
            pass

def _save_llm_config():
    try:
        with open(_LLM_CONFIG_FILE, "w") as f:
            json.dump(_LLM_CONFIG, f)
    except Exception:
        pass

_load_llm_config()


# ---------------------------------------------------------------------------
# Voice fingerprint storage
# ---------------------------------------------------------------------------
_FINGERPRINTS = {}
_FP_CONFIG_FILE = os.path.join(_BASE, ".fingerprints.json")

def _load_fingerprints():
    global _FINGERPRINTS
    if os.path.exists(_FP_CONFIG_FILE):
        try:
            with open(_FP_CONFIG_FILE, "r") as f:
                _FINGERPRINTS = json.load(f)
        except Exception:
            _FINGERPRINTS = {}

def _save_fingerprints():
    try:
        with open(_FP_CONFIG_FILE, "w") as f:
            json.dump(_FINGERPRINTS, f)
    except Exception:
        pass

_load_fingerprints()


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
# MFCC Feature Comparison for Speech Evaluation
# ---------------------------------------------------------------------------

def extract_mfcc(audio_path: str, n_mfcc: int = 13) -> np.ndarray:
    """Extract MFCC features from audio file."""
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    features = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
    return features


def dtw_distance(seq1: np.ndarray, seq2: np.ndarray) -> float:
    """Compute Dynamic Time Warping distance between two sequences."""
    n, m = seq1.shape[1], seq2.shape[1]
    if max(n, m) == 0:
        return 0.0
    dtw_matrix = np.zeros((n + 1, m + 1))
    dtw_matrix[0, 1:] = np.inf
    dtw_matrix[1:, 0] = np.inf

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.linalg.norm(seq1[:, i-1] - seq2[:, j-1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],
                dtw_matrix[i, j-1],
                dtw_matrix[i-1, j-1]
            )

    return float(dtw_matrix[n, m] / max(n, m))


def compare_voices(ref_path: str, user_path: str) -> dict:
    """
    Compare user's recording with reference/synthesized audio.
    Returns similarity score and deviations in pitch/speaking rate.
    """
    ref_chars = analyze_voice(ref_path)
    user_chars = analyze_voice(user_path)
    
    try:
        ref_mfcc = extract_mfcc(ref_path)
        user_mfcc = extract_mfcc(user_path)
        
        dtw_dist = dtw_distance(ref_mfcc, user_mfcc)
        
        ref_mean = np.mean(ref_mfcc, axis=1)
        user_mean = np.mean(user_mfcc, axis=1)
        
        cos_sim = 1 - cosine(ref_mean, user_mean)
        
        raw_similarity = (cos_sim * 0.6 + max(0, 1 - dtw_dist / 100) * 0.4)
        similarity = max(0, min(100, int(raw_similarity * 100)))
        
    except Exception as e:
        logger.warning(f"MFCC comparison failed: {e}")
        similarity = 50
    
    pitch_ref = ref_chars["mean_pitch"]
    pitch_user = user_chars["mean_pitch"]
    pitch_deviation = ((pitch_user - pitch_ref) / pitch_ref) * 100 if pitch_ref > 0 else 0
    
    rate_ref = ref_chars["speaking_rate"]
    rate_user = user_chars["speaking_rate"]
    rate_deviation = ((rate_user - rate_ref) / rate_ref) * 100 if rate_ref > 0 else 0
    
    return {
        "similarity_score": similarity,
        "pitch_deviation_percent": round(pitch_deviation, 1),
        "rate_deviation_percent": round(rate_deviation, 1),
        "reference_pitch_hz": pitch_ref,
        "user_pitch_hz": pitch_user,
        "reference_rate": rate_ref,
        "user_rate": rate_user,
    }


# ---------------------------------------------------------------------------
# Voice Blending (Accent Morphing)
# ---------------------------------------------------------------------------

def blend_characteristics(ref_chars: dict, target_chars: dict, blend: float) -> dict:
    """
    Blend two voice characteristics using a blend factor.
    blend=0.0 → ref_chars, blend=1.0 → target_chars
    Uses linear interpolation on pitch-related features,
    and geometric blending for rate/energy.
    """
    blend = np.clip(blend, 0.0, 1.0)
    
    def lerp(a, b, t):
        return a * (1 - t) + b * t
    
    def glerp(a, b, t):
        return (a ** (1 - t)) * (b ** t)
    
    return {
        "duration": lerp(ref_chars.get("duration", 1.0), 
                         target_chars.get("duration", 1.0), blend),
        "mean_pitch": lerp(ref_chars.get("mean_pitch", 150.0),
                           target_chars.get("mean_pitch", 150.0), blend),
        "pitch_std": lerp(ref_chars.get("pitch_std", 30.0),
                          target_chars.get("pitch_std", 30.0), blend),
        "pitch_range": lerp(ref_chars.get("pitch_range", 80.0),
                            target_chars.get("pitch_range", 80.0), blend),
        "speaking_rate": glerp(ref_chars.get("speaking_rate", 4.5),
                               target_chars.get("speaking_rate", 4.5), blend),
        "rms_energy": glerp(ref_chars.get("rms_energy", 0.01),
                            target_chars.get("rms_energy", 0.01), blend),
        "spectral_centroid": lerp(ref_chars.get("spectral_centroid", 2000.0),
                                   target_chars.get("spectral_centroid", 2000.0), blend),
        "_blend_factor": blend,
    }


# ---------------------------------------------------------------------------
# Voice Fingerprint Generation
# ---------------------------------------------------------------------------

def generate_fingerprint(chars: dict, audio_id: str = None) -> dict:
    """
    Generate a unique voice fingerprint from characteristics.
    Maps acoustic dimensions to polar coordinates with color/texture mapping.
    """
    # Normalize dimensions to 0-1 range for polar plot
    dims = {
        "pitch": np.clip((chars.get("mean_pitch", 150.0) - 80.0) / 350.0, 0.0, 1.0),
        "pitch_variance": np.clip(chars.get("pitch_std", 30.0) / 80.0, 0.0, 1.0),
        "speaking_rate": np.clip(chars.get("speaking_rate", 4.5) / 8.0, 0.0, 1.0),
        "energy": np.clip(chars.get("rms_energy", 0.01) * 8000.0, 0.0, 1.0),
        "brightness": np.clip((chars.get("spectral_centroid", 2000.0) - 800.0) / 4000.0, 0.0, 1.0),
        "pitch_range": np.clip(chars.get("pitch_range", 80.0) / 200.0, 0.0, 1.0),
    }
    
    # Create polar coordinates for the fingerprint shape
    keys = ["pitch", "pitch_variance", "speaking_rate", "energy", "brightness", "pitch_range"]
    n = len(keys)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    
    # Generate polar points
    radii = [dims[k] for k in keys]
    
    # Calculate color based on overall characteristics
    hue = (
        dims["pitch"] * 0.6 + 
        dims["brightness"] * 0.2 + 
        dims["energy"] * 0.2
    ) * 360.0
    saturation = 60.0 + dims["pitch_variance"] * 20.0
    lightness = 50.0 + dims["energy"] * 15.0
    
    # Generate unique pattern for texture
    seed = int(
        chars.get("mean_pitch", 150) * 100 + 
        chars.get("spectral_centroid", 2000)
    ) % (2**32)
    rng = np.random.Generator(np.random.PCG64(seed))
    texture_vertices = rng.random(24).tolist()
    
    # Calculate uniqueness score
    uniqueness = np.std([dims[k] for k in keys]) * 100.0
    
    return {
        "id": audio_id or str(uuid.uuid4())[:8],
        "dimensions": dims,
        "labels": {
            "pitch": "音调",
            "pitch_variance": "音调变化",
            "speaking_rate": "语速",
            "energy": "音量",
            "brightness": "音色亮度",
            "pitch_range": "音域",
        },
        "polar_coords": {
            "angles": angles.tolist(),
            "radii": radii,
        },
        "color": {
            "hsl": [round(hue, 1), round(saturation, 1), round(lightness, 1)],
            "hex": hsl_to_hex(hue, saturation, lightness),
        },
        "texture": texture_vertices,
        "uniqueness": round(uniqueness, 1),
        "raw_characteristics": chars,
    }


def hsl_to_hex(h: float, s: float, l: float) -> str:
    """Convert HSL (0-360, 0-100, 0-100) to hex color."""
    h /= 360.0
    s /= 100.0
    l /= 100.0
    
    if s == 0:
        r = g = b = l
    else:
        def hue2rgb(p, q, t):
            if t < 0: t += 1
            if t > 1: t -= 1
            if t < 1/6: return p + (q - p) * 6 * t
            if t < 1/2: return q
            if t < 2/3: return p + (q - p) * (2/3 - t) * 6
            return p
        
        q = l * (1 + s) if l < 0.5 else l + s - l * s
        p = 2 * l - q
        r = hue2rgb(p, q, h + 1/3)
        g = hue2rgb(p, q, h)
        b = hue2rgb(p, q, h - 1/3)
    
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def compare_fingerprints(fp1: dict, fp2: dict) -> dict:
    """Compare two voice fingerprints and return similarity metrics."""
    dims1 = fp1.get("dimensions", {})
    dims2 = fp2.get("dimensions", {})
    
    keys = ["pitch", "pitch_variance", "speaking_rate", "energy", "brightness", "pitch_range"]
    
    diffs = []
    per_dim = {}
    
    for k in keys:
        v1 = dims1.get(k, 0.5)
        v2 = dims2.get(k, 0.5)
        diff = abs(v1 - v2)
        diffs.append(diff)
        per_dim[k] = {
            "value1": round(v1, 3),
            "value2": round(v2, 3),
            "difference": round(diff, 3),
            "similarity": round((1 - diff) * 100, 1),
        }
    
    avg_diff = np.mean(diffs)
    overall_similarity = round((1 - avg_diff) * 100, 1)
    
    return {
        "overall_similarity": overall_similarity,
        "average_difference": round(avg_diff, 4),
        "per_dimension": per_dim,
        "color1": fp1.get("color", {}).get("hex", "#6366f1"),
        "color2": fp2.get("color", {}).get("hex", "#06b6d4"),
    }


def export_fingerprint_to_png(fp: dict, size: int = 400) -> bytes:
    """Export a voice fingerprint to PNG bytes using matplotlib."""
    if not HAS_MATPLOTLIB:
        raise RuntimeError("matplotlib is required for PNG export")
    
    fig, ax = plt.subplots(figsize=(size/100, size/100), dpi=100)
    ax.set_aspect('equal')
    
    # Clear axes
    ax.set_facecolor('#f8fafc')
    ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Get fingerprint data
    angles = np.array(fp["polar_coords"]["angles"])
    radii = np.array(fp["polar_coords"]["radii"])
    color = fp["color"]["hex"]
    
    # Close the polygon
    angles = np.append(angles, angles[0])
    radii = np.append(radii, radii[0])
    
    # Convert polar to Cartesian
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    
    # Plot background grid
    for r_level in [0.25, 0.5, 0.75]:
        circle = Circle((0, 0), r_level, fill=False, color='#e2e8f0', linewidth=0.5)
        ax.add_patch(circle)
    
    # Plot angular lines
    for angle in angles[:-1]:
        ax.plot([0, np.cos(angle)], [0, np.sin(angle)], color='#e2e8f0', linewidth=0.5)
    
    # Fill the fingerprint shape
    ax.fill(x, y, color=color, alpha=0.3)
    ax.plot(x, y, color=color, linewidth=2.5)
    
    # Add texture dots
    texture = fp.get("texture", [])
    for i in range(0, min(len(texture), 24), 2):
        tx = (texture[i] - 0.5) * 1.8
        ty = (texture[i+1] - 0.5) * 1.8
        ax.scatter([tx], [ty], s=15, color=color, alpha=0.6)
    
    # Set limits
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    
    # Add labels
    labels = fp.get("labels", {})
    keys = list(labels.keys())
    for i, key in enumerate(keys):
        angle = fp["polar_coords"]["angles"][i]
        label_r = 0.95
        lx = label_r * np.cos(angle)
        ly = label_r * np.sin(angle)
        ax.text(lx, ly, labels[key], 
                fontsize=8, 
                ha='center', 
                va='center',
                color='#64748b')
    
    # Save to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#f8fafc')
    plt.close(fig)
    buf.seek(0)
    
    return buf.getvalue()


def export_dual_fingerprint_to_png(fp1: dict, fp2: dict, size: int = 500) -> bytes:
    """Export two overlapping fingerprints for comparison."""
    if not HAS_MATPLOTLIB:
        raise RuntimeError("matplotlib is required for PNG export")
    
    fig, ax = plt.subplots(figsize=(size/100, size/100), dpi=100)
    ax.set_aspect('equal')
    
    ax.set_facecolor('#f8fafc')
    ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
    color1 = fp1["color"]["hex"]
    color2 = fp2["color"]["hex"]
    
    def plot_fp(fp, color, alpha=0.3):
        angles = np.array(fp["polar_coords"]["angles"])
        radii = np.array(fp["polar_coords"]["radii"])
        angles = np.append(angles, angles[0])
        radii = np.append(radii, radii[0])
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        ax.fill(x, y, color=color, alpha=alpha)
        ax.plot(x, y, color=color, linewidth=2.5)
    
    for r_level in [0.25, 0.5, 0.75]:
        circle = Circle((0, 0), r_level, fill=False, color='#e2e8f0', linewidth=0.5)
        ax.add_patch(circle)
    
    plot_fp(fp1, color1, 0.25)
    plot_fp(fp2, color2, 0.25)
    
    # Add intersection highlight
    comp = compare_fingerprints(fp1, fp2)
    ax.text(0, -1.25, f"相似度: {comp['overall_similarity']}%", 
            fontsize=12, ha='center', va='center', color='#1e293b', fontweight='bold')
    
    # Legend
    ax.text(-0.8, -1.0, "● 参考口音", fontsize=10, ha='center', color=color1)
    ax.text(0.8, -1.0, "● 目标口音", fontsize=10, ha='center', color=color2)
    
    labels = fp1.get("labels", {})
    keys = list(labels.keys())
    for i, key in enumerate(keys):
        angle = fp1["polar_coords"]["angles"][i]
        label_r = 0.95
        lx = label_r * np.cos(angle)
        ly = label_r * np.sin(angle)
        ax.text(lx, ly, labels[key], 
                fontsize=8, 
                ha='center', 
                va='center',
                color='#64748b')
    
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.4, 1.3)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#f8fafc')
    plt.close(fig)
    buf.seek(0)
    
    return buf.getvalue()


# ---------------------------------------------------------------------------
# LLM API Connectivity Test
# ---------------------------------------------------------------------------

def test_llm_connection(base_url: str, api_key: str, model_name: str) -> dict:
    """
    Test connection to an OpenAI-compatible LLM API.
    Returns status and latency information.
    """
    if not base_url:
        return {"success": False, "error": "Base URL 不能为空"}
    
    # Normalize base URL
    base_url = base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = base_url + "/v1"
    
    try:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        # First try list models to test connectivity
        try:
            models_url = f"{base_url}/models"
            resp = requests.get(models_url, headers=headers, timeout=10)
            if resp.status_code == 200:
                return {
                    "success": True, 
                    "message": "连接成功！",
                    "models_available": True,
                }
        except requests.RequestException:
            pass
        
        # If /models fails, try a simple chat completion
        chat_url = f"{base_url}/chat/completions"
        payload = {
            "model": model_name or "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 5,
        }
        
        import time
        start_time = time.time()
        
        resp = requests.post(
            chat_url, 
            headers=headers, 
            json=payload, 
            timeout=30
        )
        
        elapsed = time.time() - start_time
        
        if resp.status_code == 200:
            return {
                "success": True,
                "message": f"连接成功！响应时间: {elapsed*1000:.0f}ms",
                "latency_ms": round(elapsed * 1000, 0),
            }
        else:
            error_msg = resp.text[:200] if resp.text else f"HTTP {resp.status_code}"
            return {
                "success": False,
                "error": f"API 返回错误: {error_msg}"
            }
            
    except requests.exceptions.Timeout:
        return {"success": False, "error": "连接超时，请检查网络或 API 地址"}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "无法连接到 API，请检查地址是否正确"}
    except Exception as e:
        return {"success": False, "error": f"连接测试失败: {str(e)[:100]}"}


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# TTS via espeak-ng subprocess (offline, no network required)
# ---------------------------------------------------------------------------

#: espeak-ng executable path
_ESPEAK = shutil.which("espeak-ng") or "espeak-ng"

#: Language code → espeak-ng voice name.
#: Both zh-TW and zh-CN map to the same espeak "cmn" (Mandarin) voice because
#: espeak-ng does not distinguish Traditional from Simplified Chinese phonetically.
#: The language detection logic outputs "zh-TW" for any CJK text.
_VOICE_MAP: dict[str, str] = {
    "zh-TW": "cmn",   # Mandarin Chinese (Traditional / Simplified share the same voice)
    "zh-CN": "cmn",
    "en":    "en",
}


def _tts_generate(text: str, voice: str, out_path: str, wpm: int = 150) -> None:
    """Synthesise *text* to a WAV file using espeak-ng.

    Text is supplied via stdin (not as a CLI argument) to prevent any
    possibility of argument injection.
    """
    cmd = [_ESPEAK, "-v", voice, "-s", str(wpm), "-w", out_path, "--stdin"]
    result = subprocess.run(
        cmd,
        input=text.encode("utf-8"),
        capture_output=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"espeak-ng failed (rc={result.returncode}): {result.stderr.decode('utf-8', errors='replace').strip()}"
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
    try:
        y, sr = librosa.load(tts_tmp, sr=22050, mono=True)
    finally:
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


def _safe_path(folder: str, filename: str) -> str | None:
    """Return the resolved path only if it stays inside *folder*.

    Uses ``os.path.commonpath`` so the check works correctly on all
    platforms (including Windows drives on different roots).
    Returns None when the resolved path escapes the expected directory
    (path traversal guard).
    """
    resolved    = os.path.realpath(os.path.join(folder, filename))
    folder_real = os.path.realpath(folder)
    try:
        if os.path.commonpath([resolved, folder_real]) != folder_real:
            return None
    except ValueError:
        # commonpath raises ValueError on Windows when paths are on different drives
        return None
    return resolved


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
        logger.exception("Audio analysis failed")
        if os.path.exists(save_path):
            os.remove(save_path)
        return jsonify({"error": "音频分析失败，请检查文件格式"}), 500


@app.route("/synthesize", methods=["POST"])
def synthesize():
    data = request.get_json(force=True) or {}
    text = (data.get("text") or "").strip()
    ref_id = (data.get("ref_id") or "").strip()

    if not text:
        return jsonify({"error": "文本不能为空"}), 400
    if len(text) > 500:
        return jsonify({"error": "文本过长，请限制在 500 字以内"}), 400

    # Detect language (CJK → zh-TW, otherwise en)
    lang = "zh-TW" if any("\u4e00" <= c <= "\u9fff" for c in text) else "en"

    # Load reference characteristics or use safe defaults
    chars: dict
    if ref_id and _valid_uuid(ref_id):
        ref_path = _safe_path(UPLOAD_FOLDER, ref_id + ".wav")
        if ref_path and os.path.isfile(ref_path):
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
        logger.exception("Synthesis failed")
        return jsonify({"error": "语音合成失败，请稍后重试"}), 500


@app.route("/audio/<audio_id>")
def serve_audio(audio_id):
    if not _valid_uuid(audio_id):
        return jsonify({"error": "无效的音频 ID"}), 400
    path = _safe_path(OUTPUT_FOLDER, audio_id + ".wav")
    if not path or not os.path.isfile(path):
        return jsonify({"error": "找不到文件"}), 404
    return send_file(path, mimetype="audio/wav")


@app.route("/download/<audio_id>")
def download_audio(audio_id):
    if not _valid_uuid(audio_id):
        return jsonify({"error": "无效的音频 ID"}), 400
    path = _safe_path(OUTPUT_FOLDER, audio_id + ".wav")
    if not path or not os.path.isfile(path):
        return jsonify({"error": "找不到文件"}), 404
    return send_file(
        path,
        as_attachment=True,
        download_name="accent_echo.wav",
        mimetype="audio/wav",
    )


@app.route("/compare", methods=["POST"])
def compare():
    """Compare user's recording with synthesized audio and return score."""
    if "user_audio" not in request.files:
        return jsonify({"error": "未提供用户录音"}), 400
    
    audio_id = request.form.get("audio_id", "").strip()
    if not audio_id or not _valid_uuid(audio_id):
        return jsonify({"error": "无效的音频 ID"}), 400
    
    synth_path = _safe_path(OUTPUT_FOLDER, audio_id + ".wav")
    if not synth_path or not os.path.isfile(synth_path):
        return jsonify({"error": "找不到合成音频"}), 404
    
    user_file = request.files["user_audio"]
    user_id = str(uuid.uuid4())
    user_tmp = os.path.join(UPLOAD_FOLDER, user_id + "_user.wav")
    
    try:
        user_file.save(user_tmp)
        result = compare_voices(synth_path, user_tmp)
        return jsonify({"success": True, "result": result})
    except Exception as exc:
        logger.exception("Comparison failed")
        return jsonify({"error": "评分计算失败"}), 500
    finally:
        if os.path.exists(user_tmp):
            try:
                os.remove(user_tmp)
            except OSError:
                pass


@app.route("/analyze-output/<audio_id>")
def analyze_output(audio_id):
    """Get voice characteristics for synthesized audio (for radar chart)."""
    if not _valid_uuid(audio_id):
        return jsonify({"error": "无效的音频 ID"}), 400
    
    path = _safe_path(OUTPUT_FOLDER, audio_id + ".wav")
    if not path or not os.path.isfile(path):
        return jsonify({"error": "找不到文件"}), 404
    
    try:
        chars = analyze_voice(path)
        return jsonify({"success": True, "characteristics": chars})
    except Exception as exc:
        logger.exception("Output analysis failed")
        return jsonify({"error": "特征分析失败"}), 500


# ---------------------------------------------------------------------------
# New V2.0 Routes
# ---------------------------------------------------------------------------

@app.route("/synthesize-blend", methods=["POST"])
def synthesize_blend():
    """
    TTS with blending between two voices.
    Expects: text, ref_id1 (reference), ref_id2 (target), blend (0.0-1.0)
    blend=0.0 = reference voice, blend=1.0 = target voice
    """
    data = request.get_json(force=True) or {}
    text = (data.get("text") or "").strip()
    ref_id1 = (data.get("ref_id1") or "").strip()
    ref_id2 = (data.get("ref_id2") or "").strip()
    blend = float(data.get("blend") or 0.5)
    
    if not text:
        return jsonify({"error": "文本不能为空"}), 400
    if len(text) > 500:
        return jsonify({"error": "文本过长，请限制在 500 字以内"}), 400
    
    lang = "zh-TW" if any("\u4e00" <= c <= "\u9fff" for c in text) else "en"
    
    def get_chars(ref_id):
        if ref_id and _valid_uuid(ref_id):
            path = _safe_path(UPLOAD_FOLDER, ref_id + ".wav")
            if path and os.path.isfile(path):
                try:
                    return analyze_voice(path)
                except Exception:
                    pass
        return {"mean_pitch": 150.0, "pitch_std": 30.0, "speaking_rate": 4.5}
    
    chars1 = get_chars(ref_id1)
    chars2 = get_chars(ref_id2)
    
    blended_chars = blend_characteristics(chars1, chars2, blend)
    
    try:
        audio_id = synthesize_with_accent(text, lang, blended_chars)
        return jsonify({
            "success": True, 
            "audio_id": audio_id,
            "blend_factor": blend,
            "blended_characteristics": blended_chars,
            "source1_characteristics": chars1,
            "source2_characteristics": chars2,
        })
    except Exception as exc:
        logger.exception("Blended synthesis failed")
        return jsonify({"error": "混合语音合成失败，请稍后重试"}), 500


# ---------------------------------------------------------------------------
# LLM Configuration Routes
# ---------------------------------------------------------------------------

@app.route("/llm/config", methods=["GET"])
def get_llm_config():
    """Get current LLM configuration (without exposing API key)."""
    return jsonify({
        "success": True,
        "config": {
            "base_url": _LLM_CONFIG.get("base_url", ""),
            "model_name": _LLM_CONFIG.get("model_name", ""),
            "has_api_key": bool(_LLM_CONFIG.get("api_key", "")),
            "connected": _LLM_CONFIG.get("connected", False),
        }
    })


@app.route("/llm/config", methods=["POST"])
def save_llm_config():
    """Save LLM configuration."""
    global _LLM_CONFIG
    
    data = request.get_json(force=True) or {}
    base_url = (data.get("base_url") or "").strip()
    api_key = (data.get("api_key") or "").strip()
    model_name = (data.get("model_name") or "").strip()
    
    _LLM_CONFIG["base_url"] = base_url
    _LLM_CONFIG["model_name"] = model_name
    if api_key:
        _LLM_CONFIG["api_key"] = api_key
    
    _save_llm_config()
    
    return jsonify({
        "success": True,
        "message": "配置已保存",
    })


@app.route("/llm/test", methods=["POST"])
def test_llm():
    """Test LLM API connectivity with current or provided config."""
    global _LLM_CONFIG
    
    data = request.get_json(force=True) or {}
    
    base_url = (data.get("base_url") or "").strip() or _LLM_CONFIG.get("base_url", "")
    api_key = (data.get("api_key") or "").strip() or _LLM_CONFIG.get("api_key", "")
    model_name = (data.get("model_name") or "").strip() or _LLM_CONFIG.get("model_name", "")
    
    result = test_llm_connection(base_url, api_key, model_name)
    
    if result.get("success"):
        _LLM_CONFIG["connected"] = True
        _save_llm_config()
    else:
        _LLM_CONFIG["connected"] = False
        _save_llm_config()
    
    return jsonify(result)


# ---------------------------------------------------------------------------
# Voice Fingerprint Routes
# ---------------------------------------------------------------------------

@app.route("/fingerprint/<audio_id>", methods=["GET"])
def get_fingerprint(audio_id):
    """Get voice fingerprint for an audio file."""
    # Check both uploads and outputs folders
    path = _safe_path(UPLOAD_FOLDER, audio_id + ".wav")
    if not path or not os.path.isfile(path):
        path = _safe_path(OUTPUT_FOLDER, audio_id + ".wav")
    
    if not path or not os.path.isfile(path):
        return jsonify({"error": "找不到音频文件"}), 404
    
    try:
        chars = analyze_voice(path)
        fp = generate_fingerprint(chars, audio_id)
        
        _FINGERPRINTS[audio_id] = fp
        _save_fingerprints()
        
        return jsonify({
            "success": True,
            "fingerprint": fp,
        })
    except Exception as exc:
        logger.exception("Fingerprint generation failed")
        return jsonify({"error": "指纹生成失败"}), 500


@app.route("/fingerprint/compare", methods=["POST"])
def compare_fingerprints_route():
    """Compare two voice fingerprints."""
    data = request.get_json(force=True) or {}
    id1 = (data.get("audio_id1") or "").strip()
    id2 = (data.get("audio_id2") or "").strip()
    
    def get_fp_data(aid):
        if aid in _FINGERPRINTS:
            return _FINGERPRINTS[aid]
        
        for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER]:
            path = _safe_path(folder, aid + ".wav")
            if path and os.path.isfile(path):
                chars = analyze_voice(path)
                fp = generate_fingerprint(chars, aid)
                _FINGERPRINTS[aid] = fp
                return fp
        return None
    
    fp1 = get_fp_data(id1)
    fp2 = get_fp_data(id2)
    
    if not fp1 or not fp2:
        return jsonify({"error": "无法找到音频数据进行对比"}), 400
    
    _save_fingerprints()
    
    comparison = compare_fingerprints(fp1, fp2)
    
    return jsonify({
        "success": True,
        "comparison": comparison,
        "fingerprint1": fp1,
        "fingerprint2": fp2,
    })


@app.route("/fingerprint/export/<audio_id>", methods=["GET"])
def export_fingerprint_png(audio_id):
    """Export a single fingerprint as PNG."""
    if not HAS_MATPLOTLIB:
        return jsonify({"error": "服务器未安装 matplotlib，无法导出 PNG"}), 500
    
    path = _safe_path(UPLOAD_FOLDER, audio_id + ".wav")
    if not path or not os.path.isfile(path):
        path = _safe_path(OUTPUT_FOLDER, audio_id + ".wav")
    
    if not path or not os.path.isfile(path):
        return jsonify({"error": "找不到音频文件"}), 404
    
    try:
        chars = analyze_voice(path)
        fp = generate_fingerprint(chars, audio_id)
        
        png_data = export_fingerprint_to_png(fp, size=500)
        
        return send_file(
            io.BytesIO(png_data),
            mimetype="image/png",
            as_attachment=True,
            download_name=f"fingerprint_{audio_id}.png"
        )
    except Exception as exc:
        logger.exception("Fingerprint export failed")
        return jsonify({"error": f"导出失败: {str(exc)[:100]}"}), 500


@app.route("/fingerprint/export-compare", methods=["GET"])
def export_compare_fingerprint_png():
    """Export two overlapping fingerprints as PNG for comparison."""
    if not HAS_MATPLOTLIB:
        return jsonify({"error": "服务器未安装 matplotlib，无法导出 PNG"}), 500
    
    id1 = request.args.get("id1", "").strip()
    id2 = request.args.get("id2", "").strip()
    
    if not id1 or not id2:
        return jsonify({"error": "请提供两个音频 ID"}), 400
    
    def get_path(aid):
        for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER]:
            p = _safe_path(folder, aid + ".wav")
            if p and os.path.isfile(p):
                return p
        return None
    
    path1 = get_path(id1)
    path2 = get_path(id2)
    
    if not path1 or not path2:
        return jsonify({"error": "找不到音频文件"}), 404
    
    try:
        chars1 = analyze_voice(path1)
        chars2 = analyze_voice(path2)
        
        fp1 = generate_fingerprint(chars1, id1)
        fp2 = generate_fingerprint(chars2, id2)
        
        png_data = export_dual_fingerprint_to_png(fp1, fp2, size=600)
        
        return send_file(
            io.BytesIO(png_data),
            mimetype="image/png",
            as_attachment=True,
            download_name=f"fingerprint_compare_{id1}_{id2}.png"
        )
    except Exception as exc:
        logger.exception("Dual fingerprint export failed")
        return jsonify({"error": f"导出失败: {str(exc)[:100]}"}), 500


# ---------------------------------------------------------------------------
# F0 Curve Extraction for Pitch Chase Mode
# ---------------------------------------------------------------------------

@app.route("/extract-f0/<audio_id>", methods=["GET"])
def extract_f0_curve(audio_id):
    """Extract F0 (pitch) curve from audio for pitch chase mode."""
    path = _safe_path(OUTPUT_FOLDER, audio_id + ".wav")
    if not path or not os.path.isfile(path):
        path = _safe_path(UPLOAD_FOLDER, audio_id + ".wav")
    
    if not path or not os.path.isfile(path):
        return jsonify({"error": "找不到音频文件"}), 404
    
    try:
        y, sr = librosa.load(path, sr=22050, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)
        
        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=float(librosa.note_to_hz("C2")),
            fmax=float(librosa.note_to_hz("C7")),
            sr=sr,
            hop_length=512,
        )
        
        hop_duration = 512 / sr
        time_points = np.arange(0, len(f0)) * hop_duration
        
        f0_clean = []
        times_clean = []
        for t, f, voiced in zip(time_points, f0, voiced_flag):
            if voiced and not np.isnan(f):
                f0_clean.append(float(f))
                times_clean.append(float(t))
        
        f0_interp = []
        valid_indices = [i for i, f in enumerate(f0) if voiced_flag[i] and not np.isnan(f)]
        valid_times = [time_points[i] for i in valid_indices]
        valid_f0 = [f0[i] for i in valid_indices]
        
        from scipy import interpolate
        if len(valid_times) > 2:
            f = interpolate.interp1d(valid_times, valid_f0, kind='cubic', fill_value="extrapolate")
            for t in time_points:
                if t >= valid_times[0] and t <= valid_times[-1]:
                    f0_interp.append(float(f(t)))
                else:
                    f0_interp.append(None)
        else:
            f0_interp = [float(v) if v is not None else None for v in f0.tolist()]
        
        stats = {
            "min": float(np.nanmin(f0)) if np.any(~np.isnan(f0)) else 0,
            "max": float(np.nanmax(f0)) if np.any(~np.isnan(f0)) else 0,
            "mean": float(np.nanmean(f0)) if np.any(~np.isnan(f0)) else 0,
            "median": float(np.nanmedian(f0)) if np.any(~np.isnan(f0)) else 0,
        }
        
        return jsonify({
            "success": True,
            "audio_id": audio_id,
            "duration_seconds": round(duration, 3),
            "hop_duration_seconds": round(hop_duration, 4),
            "time_points": [round(t, 4) for t in time_points.tolist()],
            "f0_raw": [float(v) if not np.isnan(v) else None for v in f0.tolist()],
            "f0_interpolated": f0_interp,
            "voiced_frames": [bool(v) for v in voiced_flag.tolist()],
            "statistics": {k: round(v, 2) for k, v in stats.items()},
        })
        
    except Exception as exc:
        logger.exception("F0 extraction failed")
        return jsonify({"error": f"F0 曲线提取失败: {str(exc)[:100]}"}), 500


if __name__ == "__main__":
    _debug = os.environ.get("FLASK_DEBUG", "0").lower() in ("1", "true", "yes", "on", "t", "y")
    app.run(debug=_debug, host="0.0.0.0", port=5000)
