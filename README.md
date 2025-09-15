# Darren's AI Video Recorder 🎬

A synthwave-themed AI video analysis application that combines real-time webcam capture with advanced AI vision models for both single-frame and video sequence analysis.

## ✨ Features

### 🎥 Dual Analysis Modes
- **Single Frame Analysis**: Capture and analyze individual webcam frames using LM Studio models
- **Video Sequence Analysis**: Record and analyze video sequences using Qwen2.5-VL with MPS acceleration

### 🎨 Synthwave Interface
- Retro-futuristic neon color scheme with deep purples, magentas, and cyans
- Real-time video preview with recording indicators
- Intuitive controls for both capture modes

### 🧠 AI Integration
- **LM Studio**: Compatible with locally served vision language models
- **MLX-VLM**: Optimized Apple Silicon integration with Qwen2.5-VL models
- **Quantized Models**: 8-bit models for efficient memory usage

### 🎬 Video Recording
- Configurable recording duration (1-10 seconds)
- 2 FPS frame capture for efficient processing
- Real-time recording status with countdown
- Automatic temporary file management

## 🚀 Installation

### Prerequisites
- Python 3.12+
- macOS with Apple Silicon (for MPS acceleration)
- Webcam/camera access

### Setup
1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd webcam_llm
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **For LM Studio integration** (optional):
   - Install and run LM Studio
   - Load a vision language model (e.g., Qwen2-VL)
   - Ensure server is running on localhost:1234

## 📦 Dependencies

- `lmstudio>=1.5.0` - Official LM Studio SDK
- `opencv-python>=4.9.0` - Webcam capture and computer vision
- `Pillow>=10.3.0` - Image processing
- `mlx-vlm` - Apple Silicon optimized Vision Language Models

## 🎮 Usage

### Launch the Application
```bash
python synthwave_cam_lmstudio.py
```

### Single Frame Analysis
1. Select **📸 SINGLE FRAME** mode
2. Choose your camera from the dropdown
3. Select a model from LM Studio
4. Enter your prompt (e.g., "What do you see?")
5. Click **◢ CAPTURE & ANALYZE ◣**

### Video Sequence Analysis
1. Select **🎬 VIDEO SEQUENCE** mode
2. Set recording duration using the slider
3. Click **🔴 RECORD VIDEO** to start recording
4. Wait for recording to complete automatically
5. Enter your prompt (e.g., "Describe what happens in this video")
6. Click **◢ CAPTURE & ANALYZE ◣**

## 🏗️ Architecture

### Core Components

- **SynthwaveApp**: Main GUI application with tkinter
- **Camera Management**: Cross-platform OpenCV backend selection
- **Dual Inference Paths**:
  - LM Studio SDK for single-frame analysis
  - MLX-VLM integration for video sequence analysis
- **Apple Silicon Optimization**: Native MLX framework with quantized models

### Video Processing Pipeline

1. **Frame Capture**: Real-time webcam frames at 2 FPS during recording
2. **Temporary Storage**: Frames saved as JPEG files in temp directory
3. **MLX-VLM Processing**: Video frames processed as image sequence with quantized models
4. **Cleanup**: Automatic temporary file removal on exit

## ⚙️ Configuration

### Camera Settings
- Automatic backend selection (AVFoundation on macOS)
- Camera scanning and validation
- Fallback handling for camera access issues

### Model Configuration
- **MLX Model**: Uses mlx-community/Qwen2.5-VL-32B-Instruct-8bit (quantized)
- **Apple Silicon**: Native MLX framework optimization
- **LM Studio**: Configure custom host/port in code if needed

## 🎨 UI Elements

### Color Palette
- Background: `#0d0221` (Deep Space Blue)
- Panels: `#1a0040` (Dark Purple)
- Accents: `#2d0a5c` (Medium Purple)
- Neon: `#ff2bd6` (Hot Pink)
- Cyan: `#00ffff` (Electric Cyan)
- Text: `#e8e6ff` (Light Purple)

### Controls
- **Camera Selection**: Dropdown with auto-detection
- **Model Selection**: LM Studio model picker
- **Recording Duration**: 1-10 second slider
- **Mode Toggle**: Radio buttons for analysis type
- **Status Display**: Real-time operation feedback

## 🔧 Troubleshooting

### Camera Issues
- Ensure no other applications are using the camera
- Check system privacy settings for camera access
- Try running as administrator if needed

### Model Loading
- First run will download MLX quantized model (~8GB)
- Much smaller and faster than full precision models
- Native Apple Silicon optimization with MLX framework

### Performance Tips
- Use smaller models for faster inference
- Reduce recording duration for quicker processing
- Ensure adequate disk space for temporary files

## 🎯 Future Enhancements

- [ ] Multiple video format export
- [ ] Batch video processing
- [ ] Real-time video streaming analysis
- [ ] Custom model integration
- [ ] Audio analysis capabilities
- [ ] Cloud deployment options

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Credits

- Built with ❤️ using Python, tkinter, OpenCV, and Transformers
- Qwen2.5-VL by Alibaba Cloud
- Synthwave aesthetic inspired by retro-futurism
- Created for Darren's AI exploration journey

---

*Experience the future of AI video analysis with synthwave style* ✨🎬