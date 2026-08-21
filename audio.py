import asyncio
import websockets
import pyaudio
import json
import wave
import numpy as np

def show_audio_devices():
    p = pyaudio.PyAudio()
    print("\nAvailable audio outputs:")
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if dev['maxOutputChannels'] > 0:
            print(f"{dev['index']}: {dev['name']} (Channels: {dev['maxOutputChannels']})")
    p.terminate()

async def tts_client(prompt: str):
    """Connect to WebSocket and handle audio streaming with real-time playback."""
    try:
        async with websockets.connect("wss://sia-nus-orpheus.karyolymphkinetograph.uk/ws") as ws:
            # Send TTS request
            await ws.send(json.dumps({"prompt": prompt, "voice": "tara"}))
            
            # Receive audio format metadata
            metadata = json.loads(await ws.recv())
            print(f"Received audio format: {metadata}")

            # Initialize audio playback
            p = pyaudio.PyAudio()
            
            try:
                # Create output stream with larger chunk size
                stream = p.open(
                    format=p.get_format_from_width(metadata['sample_width']),
                    channels=metadata['channels'],
                    rate=metadata['sample_rate'],
                    output=True,
                    frames_per_buffer=4096  # Larger buffer reduces underflow
                )

                # Create WAV file for debugging
                with wave.open('debug_output.wav', 'wb') as wav_file:
                    wav_file.setnchannels(metadata['channels'])
                    wav_file.setsampwidth(metadata['sample_width'])
                    wav_file.setframerate(metadata['sample_rate'])

                    # Real-time playback loop
                    try:
                        while True:
                            audio_data = await ws.recv()
                            print(f"Received {len(audio_data)} bytes")  # Debug output
                            
                            # Play audio chunk
                            stream.write(audio_data)
                            
                            # Save to WAV file
                            wav_file.writeframes(audio_data)

                    except websockets.exceptions.ConnectionClosedOK:
                        print("Streaming completed successfully")

            finally:
                stream.stop_stream()
                stream.close()
                p.terminate()

    except Exception as e:
        print(f"Client error: {e}")

def main():
    show_audio_devices()
    
    while True:
        text = input("\nEnter text to synthesize (or 'exit' to quit): ").strip()
        if text.lower() == 'exit':
            break
        if text:
            asyncio.run(tts_client(text))
    
    print("Client terminated")

if __name__ == "__main__":
    # Install requirements: pip install websockets pyaudio wave numpy
    main()
