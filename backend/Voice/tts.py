"""
backend/Voice/tts.py

Text-to-Speech using piper-tts.
Loads the voice model once (lazy, thread-safe).
Exposes a single blocking function: synthesize(text) -> bytes (WAV)
"""

import io
import os
import threading
import wave
import logging
import subprocess
import tempfile
import struct
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = os.path.join(
    os.path.dirname(__file__), "models", "en_US-lessac-medium.onnx"
)
PIPER_MODEL_PATH = os.environ.get("PIPER_MODEL_PATH", _DEFAULT_MODEL)
PIPER_EXECUTABLE = os.environ.get("PIPER_EXECUTABLE", "piper")

# ---------------------------------------------------------------------------
# Lazy singleton
# ---------------------------------------------------------------------------

_voice = None
_voice_lock = threading.Lock()
_available = None


def is_available() -> bool:
    """Return True if piper and its model file are both present."""
    global _available
    if _available is None:
        # Check if piper command exists
        piper_exists = False
        try:
            result = subprocess.run(
                [PIPER_EXECUTABLE, "--help"],
                capture_output=True,
                text=True,
                timeout=2
            )
            piper_exists = result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        
        _available = piper_exists and os.path.isfile(PIPER_MODEL_PATH)
        
        if not _available:
            if not piper_exists:
                logger.warning(
                    "Piper executable not found. TTS disabled. "
                    "Install with: pip install piper-tts"
                )
            elif not os.path.isfile(PIPER_MODEL_PATH):
                logger.warning(
                    "Piper model not found at '%s'. TTS disabled.",
                    PIPER_MODEL_PATH,
                )
    return _available


def preload() -> None:
    """Preload TTS model (no-op, we'll use subprocess)."""
    if is_available():
        logger.info("Piper TTS ready (using subprocess)")


def _create_wav_header(sample_rate: int, bits_per_sample: int, channels: int, data_size: int) -> bytes:
    """Create a WAV file header."""
    byte_rate = sample_rate * channels * (bits_per_sample // 8)
    block_align = channels * (bits_per_sample // 8)
    
    header = struct.pack('<4sI4s', b'RIFF', 36 + data_size, b'WAVE')
    header += struct.pack('<4sI', b'fmt ', 16)
    header += struct.pack('<HHIIHH', 1, channels, sample_rate, byte_rate, block_align, bits_per_sample)
    header += struct.pack('<4sI', b'data', data_size)
    
    return header

# backend/Voice/tts.py - Modified to return raw PCM

def synthesize_raw(text: str) -> bytes:
    """Return raw PCM audio bytes (16-bit mono)."""
    if not is_available():
        raise RuntimeError("Piper TTS not available")
    
    with tempfile.NamedTemporaryFile(suffix='.raw', delete=False) as tmp:
        raw_path = tmp.name
    
    try:
        cmd = [
            PIPER_EXECUTABLE,
            "--model", PIPER_MODEL_PATH,
            "--output_file", raw_path,
            "--output-raw",
        ]
        
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        process.communicate(input=text, timeout=15)
        
        if process.returncode != 0:
            raise RuntimeError("Piper failed")
        
        with open(raw_path, 'rb') as f:
            return f.read()
            
    finally:
        if os.path.exists(raw_path):
            os.unlink(raw_path)


def get_audio_config() -> dict:
    """Return audio configuration for the client."""
    sample_rate = 22050  # Default
    model_json = PIPER_MODEL_PATH + '.json'
    if os.path.exists(model_json):
        try:
            import json
            with open(model_json, 'r') as f:
                config = json.load(f)
                sample_rate = config.get('audio', {}).get('sample_rate', 22050)
        except:
            pass
    
    return {
        "sample_rate": sample_rate,
        "channels": 1,
        "bit_depth": 16
    }
    
def synthesize(text: str) -> bytes:
    """Convert text to WAV audio bytes using piper subprocess.
    
    Parameters
    ----------
    text : str
        The sentence or phrase to speak.
        
    Returns
    -------
    bytes
        A valid WAV file in memory.
        
    Raises
    ------
    RuntimeError
        If piper is not installed or the model file is missing.
    """
    if not is_available():
        raise RuntimeError(
            "Piper TTS is not available. Install piper-tts and download the model file."
        )
    
    if not text or not text.strip():
        raise ValueError("Text to synthesize cannot be empty")
    
    # Use subprocess approach which is more reliable across versions
    return _synthesize_with_subprocess(text)


def _synthesize_with_subprocess(text: str) -> bytes:
    """Use piper command-line tool to synthesize."""
    
    # Create temporary files for raw audio and WAV
    with tempfile.NamedTemporaryFile(suffix='.raw', delete=False) as raw_tmp:
        raw_path = raw_tmp.name
    
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as json_tmp:
        json_path = json_tmp.name
    
    try:
        # Get model config to know sample rate
        # First, try to read the JSON config file
        sample_rate = 22050  # Default fallback
        model_json_path = PIPER_MODEL_PATH + '.json'
        
        if os.path.exists(model_json_path):
            try:
                import json
                with open(model_json_path, 'r') as f:
                    config = json.load(f)
                    sample_rate = config.get('audio', {}).get('sample_rate', 22050)
            except Exception as e:
                logger.warning(f"Could not read model config: {e}")
        
        # Run piper command to generate raw audio
        cmd = [
            PIPER_EXECUTABLE,
            "--model", PIPER_MODEL_PATH,
            "--output_file", raw_path,
            "--output-raw",  # Output raw audio (no WAV header)
        ]
        
        logger.debug(f"Running piper command: {' '.join(cmd)}")
        
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Send text to stdin
        stdout, stderr = process.communicate(input=text, timeout=15)
        
        if process.returncode != 0:
            error_msg = stderr if stderr else "Unknown error"
            raise RuntimeError(f"Piper failed with code {process.returncode}: {error_msg}")
        
        # Read the raw audio data
        if not os.path.exists(raw_path) or os.path.getsize(raw_path) == 0:
            raise RuntimeError("Piper produced no audio output")
        
        with open(raw_path, 'rb') as f:
            raw_audio = f.read()
        
        # Wrap raw audio in WAV header
        # Piper outputs 16-bit mono PCM
        channels = 1
        bits_per_sample = 16
        data_size = len(raw_audio)
        
        # Create WAV header
        wav_header = _create_wav_header(sample_rate, bits_per_sample, channels, data_size)
        
        # Combine header and raw audio
        wav_bytes = wav_header + raw_audio
        
        return wav_bytes
        
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        raise RuntimeError("Piper synthesis timed out after 15 seconds")
    except Exception as e:
        raise RuntimeError(f"Piper synthesis failed: {e}")
    finally:
        # Clean up temporary files
        for path in [raw_path, json_path]:
            if os.path.exists(path):
                try:
                    os.unlink(path)
                except Exception:
                    pass


# Alternative implementation using piper's Python API if available
def _synthesize_with_piper_package(text: str) -> bytes:
    """Alternative: Use piper Python package if installed."""
    try:
        from piper import PiperVoice
        
        voice = PiperVoice.load(PIPER_MODEL_PATH)
        
        # Create WAV file in memory
        wav_buffer = io.BytesIO()
        
        # Write WAV header
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit = 2 bytes
            wav_file.setframerate(voice.config.sample_rate)
            
            # Synthesize and write audio chunks
            # Different piper versions have different APIs
            if hasattr(voice, 'synthesize'):
                result = voice.synthesize(text)
                if isinstance(result, bytes):
                    wav_file.writeframes(result)
                elif hasattr(result, '__iter__'):
                    for chunk in result:
                        wav_file.writeframes(chunk)
            elif hasattr(voice, 'synthesize_stream'):
                for chunk in voice.synthesize_stream(text):
                    wav_file.writeframes(chunk)
            else:
                # Fallback: use pipe method
                audio_bytes = voice.pipe(text.encode('utf-8'))
                wav_file.writeframes(audio_bytes)
        
        return wav_buffer.getvalue()
        
    except ImportError:
        raise RuntimeError("piper Python package not installed")
    except Exception as e:
        raise RuntimeError(f"Piper Python API failed: {e}")


# For testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test synthesis
    test_text = "Hello, this is a test of the text to speech system."
    
    if is_available():
        print(f"TTS is available. Testing with: '{test_text}'")
        try:
            audio_bytes = synthesize(test_text)
            print(f"Generated {len(audio_bytes)} bytes of WAV audio")
            
            # Save to file for testing
            with open("test_output.wav", "wb") as f:
                f.write(audio_bytes)
            print("Saved to test_output.wav")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("TTS not available. Check piper installation and model file.")