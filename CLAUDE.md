# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project contains webcam-to-LLM applications with synthwave-themed GUIs that capture video frames and send them to LM Studio for AI analysis. There are three main implementations:

- `webcamllm.py`: Uses OpenAI SDK pointed at LM Studio's REST API
- `synthwave_cam_lmstudio.py`: Uses official LM Studio SDK with camera scanning
- `synthwave_cam_lmstudio2.py`: Enhanced version with improved camera handling

## Environment Setup

This project uses a Python virtual environment (Python 3.12.2). The virtual environment is already configured in the project directory.

### Dependencies

Install dependencies using:
```bash
pip install -r requirements.txt
```

Core dependencies:
- `lmstudio>=1.5.0` - Official LM Studio SDK
- `opencv-python>=4.9.0` - Computer vision and webcam capture
- `Pillow>=10.3.0` - Image processing

The OpenAI SDK is also used by `webcamllm.py` but not listed in requirements.txt.

## Running the Applications

### Using LM Studio SDK (Recommended)
```bash
python synthwave_cam_lmstudio2.py
```
This version includes robust camera detection and error handling.

### Using OpenAI SDK
```bash
python webcamllm.py
```
This version connects to LM Studio via OpenAI-compatible REST API.

## Architecture

All applications follow a similar pattern:

### Core Components

1. **SynthwaveApp Class**: Main GUI application class with synthwave color scheme
   - Handles webcam initialization and frame capture
   - Manages GUI layout with tkinter/ttk
   - Processes AI model communication

2. **Camera Management**:
   - Platform-specific OpenCV backend selection (`_pick_cv_backend()`)
   - Camera scanning and validation (`_scan_working_cameras()`)
   - Robust camera opening with fallback (`_try_open_camera()`)

3. **GUI Layout**:
   - Left panel: Live video feed, capture controls, captured image preview
   - Right panel: Model selection, text prompt, AI response display
   - Synthwave color palette: deep purple background, magenta/cyan neon accents

4. **LM Studio Integration**:
   - Model enumeration and selection
   - Text + image prompt processing
   - Streaming response handling

### Key Functions

- `_build_styles()`: Configures synthwave theme for ttk widgets
- `capture_frame()`: Captures current video frame for AI analysis
- `send_to_model()`: Sends text prompt and image to selected AI model
- `refresh_models()`/`rescan_cameras()`: Updates available models/cameras

## Configuration

### LM Studio Connection

**For LM Studio SDK versions**: Uses default LM Studio client configuration. Customize with:
```python
lms.configure_default_client("localhost:1234")  # Custom host:port
```

**For OpenAI SDK version**: Configured via constants:
```python
LMSTUDIO_BASE_URL = "http://localhost:1234/v1"
LMSTUDIO_API_KEY = "lm-studio"  # Placeholder, ignored by LM Studio
```

### Platform Considerations

Camera backends are automatically selected:
- macOS: `CAP_AVFOUNDATION`
- Windows: `CAP_DSHOW`
- Linux: `CAP_V4L2` or `CAP_ANY`

## Development Notes

- All applications use threading for non-blocking AI model calls
- Error handling includes webcam access failures and model communication issues
- GUI uses grid layout manager with proper weight distribution for responsiveness
- Images are converted between OpenCV BGR, PIL RGB, and base64 formats as needed