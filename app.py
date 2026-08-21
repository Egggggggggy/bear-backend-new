import ast
import asyncio
import json
import queue
import websockets
import pyaudio
import threading
import pyaudio
import wave
import numpy as np


# Audio Configuration
FORMAT = pyaudio.paInt16  # 16-bit PCM
CHANNELS = 1              # Mono audio
RATE = 16000              # Sample rate (Hz)
CHUNK = 480               # Samples per chunk (30ms at 16kHz)


# WebSocket endpoints
AUDIO_WS_URL = "wss://sia-nus-dev.karyolymphkinetograph.uk/audio"
FORWARD_WS_URL = "wss://sia-nus.karyolymphkinetograph.uk/listen/test/kobold/daisy/0/0/start"


# Add these configurations at the top with other audio settings
NOISE_GATE_ENABLED = True    # Set to False to disable noise gate
NOISE_GATE_THRESHOLD = 500   # RMS threshold (0-32767), adjust based on environment
ATTACK_TIME = 0.05           # Seconds to open gate after threshold crossed
RELEASE_TIME = 0.25          # Seconds to close gate after falling below threshold


MUTE = False  # Global mute flag


# Add this class definition before the audio_capture_loop function
class NoiseGate:
    def __init__(self, rate, chunk_size, threshold, attack, release):
        self.gate_open = False
        self.samples_per_chunk = chunk_size
        self.attack_samples = int(rate * attack)
        self.release_samples = int(rate * release)
        self.hold_counter = 0
        self.threshold = threshold
        
    def process(self, samples):
        # Handle empty/invalid input
        if samples.size == 0:
            self.gate_open = False
            self.hold_counter = 0
            return False
            
        # Ensure we don't calculate RMS of negative values (should never happen with squared audio)
        squared = samples.astype(np.float32)**2
        mean_square = np.mean(squared)
        
        # Handle potential NaN/Infinite values
        if not np.isfinite(mean_square) or mean_square < 0:
            self.gate_open = False
            self.hold_counter = 0
            return False
            
        rms = np.sqrt(mean_square)
        
        if self.gate_open:
            if rms < self.threshold:
                self.hold_counter -= self.samples_per_chunk
                if self.hold_counter <= 0:
                    self.gate_open = False
            else:
                self.hold_counter = self.release_samples
        else:
            if rms >= self.threshold:
                self.hold_counter += self.samples_per_chunk
                if self.hold_counter >= self.attack_samples:
                    self.gate_open = True
            else:
                self.hold_counter = max(0, self.hold_counter - self.samples_per_chunk)
        
        return self.gate_open
def show_audio_devices():
    p = pyaudio.PyAudio()
    print("\nAvailable audio outputs:")
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if dev['maxOutputChannels'] > 0:
            print(f"{dev['index']}: {dev['name']} (Channels: {dev['maxOutputChannels']})")
    p.terminate()

async def tts_client(prompt: str, stop_event: threading.Event):
    """TTS client with buffered audio playback"""
    global MUTE
    try:
        async with websockets.connect("wss://sia-nus-orpheus.karyolymphkinetograph.uk/ws") as ws:
            await ws.send(json.dumps({"prompt": prompt, "voice": "tara"}))
            metadata = json.loads(await ws.recv())
            print(f"TTS Format: {metadata}")

            p = pyaudio.PyAudio()
            stream = None
            
            try:
                stream = p.open(
                    format=p.get_format_from_width(metadata['sample_width']),
                    channels=metadata['channels'],
                    rate=metadata['sample_rate'],
                    output=True,
                    frames_per_buffer=4096
                )

                with wave.open('debug_output.wav', 'wb') as wav_file:
                    MUTE = False
                    wav_file.setnchannels(metadata['channels'])
                    wav_file.setsampwidth(metadata['sample_width'])
                    wav_file.setframerate(metadata['sample_rate'])

                    # Audio buffering configuration
                    buffer_time = 0.1  # 100ms buffer
                    bytes_per_frame = metadata['channels'] * metadata['sample_width']
                    buffer_threshold = int(metadata['sample_rate'] * buffer_time) * bytes_per_frame
                    audio_buffer = b''

                    async def monitor_stop():
                        while not stop_event.is_set():
                            await asyncio.sleep(0.1)
                        await ws.close()

                    monitor_task = asyncio.create_task(monitor_stop())

                    try:
                        while True:
                            try:
                                audio_data = await ws.recv()
                            except websockets.ConnectionClosedOK:
                                break
                                
                            if stop_event.is_set():
                                break

                            audio_buffer += audio_data

                            # Write complete chunks to stream
                            while len(audio_buffer) >= buffer_threshold:
                                chunk = audio_buffer[:buffer_threshold]
                                stream.write(chunk)
                                wav_file.writeframes(chunk)
                                audio_buffer = audio_buffer[buffer_threshold:]

                        # Flush remaining audio
                        if audio_buffer:
                            stream.write(audio_buffer)
                            wav_file.writeframes(audio_buffer)

                    finally:
                        monitor_task.cancel()
            finally:
                if stream:
                    stream.close()
                p.terminate()
                stop_event.clear()

    except Exception as e:
        print(f"TTS Error: {e}")

        
def list_microphones(p):
    """List all available input devices and return default device index."""
    print("\nAvailable input devices:")
    input_devices = []
    for i in range(p.get_device_count()):
        dev_info = p.get_device_info_by_index(i)
        if dev_info['maxInputChannels'] > 0:
            input_devices.append(i)
            print(f"Index {i}: {dev_info['name']}")
            print(f"   Channels: {dev_info['maxInputChannels']}")
            print(f"   Default Sample Rate: {dev_info['defaultSampleRate']:.1f} Hz")
            print(f"   Host API: {p.get_host_api_info_by_index(dev_info['hostApi'])['name']}")

    if not input_devices:
        print("No input devices found!")
        return None
        
    default_dev = p.get_default_input_device_info()
    print(f"\nDefault input device: Index {default_dev['index']} - {default_dev['name']}")
    return default_dev['index']

def audio_capture_loop(audio_queue, loop):
    """Capture audio from the microphone and populate the queue with noise gating."""
    p = pyaudio.PyAudio()
    global MUTE
    
    # List devices and validate configuration
    device_index = list_microphones(p)
    if device_index is None:
        return

    try:
        # Validate device supports our format
        if not p.is_format_supported(
            rate=RATE,
            input_device=device_index,
            input_channels=CHANNELS,
            input_format=FORMAT
        ):
            print(f"\nError: Device {device_index} doesn't support:")
            print(f"Sample rate: {RATE} Hz")
            print(f"Channels: {CHANNELS}")
            print(f"Format: {FORMAT} ({p.get_sample_size(FORMAT)*8}-bit PCM)")
            return

        # Initialize noise gate if enabled
        noise_gate = None
        if NOISE_GATE_ENABLED:
            noise_gate = NoiseGate(
                rate=RATE,
                chunk_size=CHUNK,
                threshold=NOISE_GATE_THRESHOLD,
                attack=ATTACK_TIME,
                release=RELEASE_TIME
            )
            print(f"\nNoise Gate Enabled | Threshold: {NOISE_GATE_THRESHOLD} | Attack: {ATTACK_TIME}s | Release: {RELEASE_TIME}s")


        # Print final configuration
        print(f"\nOpening audio stream with:")
        print(f"Format: {FORMAT} ({p.get_sample_size(FORMAT)*8}-bit PCM)")
        print(f"Channels: {CHANNELS}")
        print(f"Sample Rate: {RATE} Hz")
        print(f"Chunk Size: {CHUNK} samples ({CHUNK/RATE*1000:.1f} ms)")

        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=CHUNK,
            start=False
        )

        print("Recording audio... Press Ctrl+C to stop.")
        stream.start_stream()

        try:
            while True:
                data = stream.read(CHUNK, exception_on_overflow=False)
                
                # Apply noise gate if enabled
                if NOISE_GATE_ENABLED:
                    samples = np.frombuffer(data, dtype=np.int16)
                    gate_open = noise_gate.process(samples)
                    
                    if not gate_open:
                        # Replace with silence while gate is closed
                        data = np.zeros_like(samples).tobytes()
                        
                # Global mute override (applies after noise gating)
                if MUTE:
                    data = b'\x00' * len(data)  # Efficient silence generation

                loop.call_soon_threadsafe(audio_queue.put_nowait, data)

                loop.call_soon_threadsafe(audio_queue.put_nowait, data)

        except KeyboardInterrupt:
            print("Audio capture interrupted.")
        except Exception as e:
            print(f"Audio capture error: {e}")
        finally:
            stream.stop_stream()
            stream.close()
    finally:
        p.terminate()


# Rest of the code remains the same as previous version
# (send_audio, receive_messages, and main functions)

async def send_audio(websocket, audio_queue):
    try:
        while True:
            data = await audio_queue.get()
            await websocket.send(data)
    except asyncio.CancelledError:
        print("Audio sending task cancelled.")
        
async def receive_messages(audio_ws, forward_ws):
    """Message handler with TTS interruption"""
    current_tts_task = None
    current_stop_event = None
    tts_lock = asyncio.Lock()
    
    global MUTE

    try:
        while True:
            response = await audio_ws.recv()
            print(f"\nServer Message: {response}")
            
            try:
                response_dict = ast.literal_eval(response)

                # Handle speech detection
                if response_dict.get('detection') == 'proper_speech_start':
                    async with tts_lock:
                        if current_stop_event:
                            current_stop_event.set()
                            current_stop_event = None
                        if current_tts_task:
                            current_tts_task.cancel()
                            current_tts_task = None
                    continue

                # Handle transcription forwarding
                if 'text' in response_dict:
                    MUTE = True
                    print(f"Transcription: {response_dict['text']}")
                    await forward_ws.send(response_dict['text'])
                    try:
                        forward_response = await asyncio.wait_for(forward_ws.recv(), 10)
                        response_data = json.loads(forward_response)
                        
                        if 'message' in response_data:
                            print(response_data)
                            async with tts_lock:
                                # Cancel existing TTS
                                if current_tts_task:
                                    current_tts_task.cancel()
                                if current_stop_event:
                                    current_stop_event.set()
                                
                                # Start new TTS
                                new_stop_event = threading.Event()

                                current_tts_task = asyncio.create_task(
                                    tts_client(response_data['message'], new_stop_event)
                                )
                                current_stop_event = new_stop_event

                    except asyncio.TimeoutError:
                        print("Forward response timeout")
            except Exception as e:
                print(f"Message error: {e}")

    except asyncio.CancelledError:
        print("Message receiver stopped")
        async with tts_lock:
            if current_tts_task:
                current_tts_task.cancel()
            if current_stop_event:
                current_stop_event.set()

async def main():
    """Main connection handler with dual WebSocket support"""
    loop = asyncio.get_running_loop()
    audio_queue = asyncio.Queue()

    # Start audio capture thread
    capture_thread = threading.Thread(
        target=audio_capture_loop,
        args=(audio_queue, loop),
        daemon=True
    )
    capture_thread.start()

    try:
        # Connect to both WebSockets
        async with websockets.connect(AUDIO_WS_URL) as audio_ws, \
                   websockets.connect(FORWARD_WS_URL) as forward_ws:
            
            print(f"Connected to audio WS: {AUDIO_WS_URL}")
            print(f"Connected to forwarding WS: {FORWARD_WS_URL}")

            # Start communication tasks
            send_task = asyncio.create_task(send_audio(audio_ws, audio_queue))
            recv_task = asyncio.create_task(receive_messages(audio_ws, forward_ws))
            
            await asyncio.gather(send_task, recv_task)

    except websockets.exceptions.ConnectionClosedError as e:
        print(f"Connection closed: {e.code} - {e.reason}")
    except Exception as e:
        print(f"Critical error: {str(e)}")
    finally:
        print("Closing all connections...")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nClient terminated by user")