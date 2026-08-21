# How to run:
$ source venv/bin/activate
$ python app.py


# Project Documentation

## Overview

This bear project implements a **real-time speech pipeline** using WebSockets, microphone input, noise gating, transcription forwarding, and text-to-speech (TTS) playback. Designed to run on single microphone, RPI with Ubuntu.  

It consists of two main scripts:

- **`app.py`**:  
  Handles audio capture from a microphone, noise-gating, sending audio to a speech-to-text service, forwarding transcriptions to another service, and receiving and speaking back responses using TTS.

- **`audio.py`**:  
  A standalone TTS client for testing. It streams synthesized speech audio via WebSocket and plays it back in real time.

---

## Features

- **Microphone input with PyAudio**  
- **Noise Gate processing** (with configurable threshold, attack, release)  
- **Real-time audio streaming over WebSockets**  
- **Forwarding of transcription results to another WebSocket**  
- **TTS synthesis and playback with interruption handling**  
- **Audio device listing (input & output)**  
- **Debugging via saving to `.wav` files**  

---

## Dependencies
Conda not installed, so just use venv. For your own testing just use conda.
$ pip install -r requirements.txt

Also uses 2 websockets linked to 2 models, ConvoAI and WhisperX STT. HTTPS calls for Orpheus TTS. These services hosted on another private server.
If you want to change the personality of the bear avatar, need to update on ConvoAI side, not done here!

---

## Configuration

Defined in `app.py`:

- **Audio Settings**
  - `FORMAT = pyaudio.paInt16`
  - `CHANNELS = 1` (mono)
  - `RATE = 16000` Hz
  - `CHUNK = 480` (30 ms @ 16kHz)

- **WebSocket Endpoints**
  - `AUDIO_WS_URL`: audio stream destination
  - `FORWARD_WS_URL`: text forward destination
  - TTS endpoint is fixed inside `tts_client`.

- **Noise Gate**
  - `NOISE_GATE_ENABLED = True`
  - `NOISE_GATE_THRESHOLD = 500`
  - `ATTACK_TIME = 0.05`
  - `RELEASE_TIME = 0.25`

- **Global Flag**
  - `MUTE`: silences audio output/input when set to `True`.

---

## Module: `app.py`

### Classes

#### `NoiseGate`
Implements a simple noise gate for microphone input.

- **Constructor**:  
  `__init__(rate, chunk_size, threshold, attack, release)`  
  Configures RMS threshold, attack, and release.

- **Method**:  
  `process(samples: np.ndarray) -> bool`  
  Takes a NumPy array of audio samples and returns whether the gate is open.

---

### Functions

#### `show_audio_devices()`
Prints all available output devices.

#### `list_microphones(p: pyaudio.PyAudio) -> int | None`
Lists all input devices and returns the default device index.

#### `audio_capture_loop(audio_queue, loop)`
Captures audio from the microphone, applies noise gate & mute, and pushes data into an asyncio queue.

#### `tts_client(prompt: str, stop_event: threading.Event)`
Connects to the TTS WebSocket, streams audio playback in real-time, and supports interruption via `stop_event`.

#### `send_audio(websocket, audio_queue)`
Consumes audio chunks from a queue and sends them to the speech WebSocket.

#### `receive_messages(audio_ws, forward_ws)`
Handles server messages:
- Detects speech start → stops ongoing TTS.
- Forwards transcriptions to `forward_ws`.
- Receives responses and starts new TTS.

#### `main()`
Entry point:
- Starts audio capture thread.
- Connects to both WebSockets (`AUDIO_WS_URL`, `FORWARD_WS_URL`).
- Runs `send_audio` and `receive_messages` concurrently.

---

## Module: `audio.py`

A simpler standalone TTS client for debugging.

### Functions

#### `show_audio_devices()`
Lists all available audio output devices.

#### `tts_client(prompt: str)`
Connects to the TTS WebSocket:
- Sends a text prompt.
- Receives metadata (sample width, rate, channels).
- Streams audio playback with PyAudio.
- Saves audio to `debug_output.wav`.

#### `main()`
- Shows available devices.
- Loops for user input.
- Calls `tts_client(text)` until `"exit"` is entered.

---

## Execution Flow

### In `app.py`
1. **Startup**:
   - Lists input devices.
   - Starts a thread capturing audio → queue.

2. **WebSocket Connections**:
   - `audio_ws`: receives transcription results, sends audio.
   - `forward_ws`: forwards transcription to an external service, receives responses.

3. **Message Handling**:
   - On `"proper_speech_start"`: cancels current TTS.
   - On `"text"`: forwards transcription, gets response, starts new TTS.

4. **TTS Playback**:
   - Streams audio in real time with optional interruption.

---

## Debugging

- Both modules save TTS audio into `debug_output.wav` for analysis.
- Console logs provide insights into:
  - Device detection
  - Noise gate state
  - WebSocket connection events
  - Received transcription and responses

---

## Example Usage

### Running full pipeline (speech-to-text + forward + TTS):
```bash
python app.py
```

### Running standalone TTS client:
```bash
python audio.py
```
Then enter text prompts to synthesize speech.

---

## Potential Improvements

- Currently using full orpheus TTS model, inference is slow. To switch over to quantized model and test.
- STT is multi-lingual, but TTS is stuck in english only. There should be other models to simulate other languages also under orpheus.
- Video streaming will be pretty cool. Just stream extracted features tho, don't try streaming video feed as thats only production feasible via local network connections. (Trust us we tried).
- Add physical buttons / lights to control microphone & display whenever bear is speaking / listening.
- Theres no screeeeen. We can't see whats going on. Should add that. 
