# Robot companion runtime

This repository contains an asynchronous Python package for a Jetson Orin Nano
healthcare companion. The package is under [`src/robot_companion`](src/robot_companion)
and uses bounded queues plus worker tasks so camera/audio capture is not blocked
by model inference or cloud dialogue.

## Run

The default process is intentionally safe on a development machine: missing
Picamera2, audio, and model backends are logged and the application remains
idle. On Jetson, install the JetPack-compatible extras from
[`requirements.txt`](requirements.txt), configure the model adapters, then run:

```bash
PYTHONPATH=src python -m robot_companion.main
```

Set `ROBOT_LLM_ENDPOINT`, `ROBOT_LLM_API_KEY`, optional `ROBOT_TTS_WS_URL`, and optionally `LOG_LEVEL` for
cloud dialogue and speech output. Model weights are not downloaded implicitly at import time.

## Tests

Run the unit and integration test suite:

```bash
pytest
```