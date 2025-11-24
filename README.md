# ComfyUI Qwen Image-Edit 2509 + UltimateSDUpscale Pipeline

[![Python](https://img.shields.io/badge/Python-3.11.13-blue.svg)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-Enabled-green.svg)](https://developer.nvidia.com/cuda-zone)
[![Kaggle](https://img.shields.io/badge/Platform-Kaggle-orange.svg)](https://www.kaggle.com/)
[![GPU](https://img.shields.io/badge/GPU-NVIDIA%20Tesla%20T4-red.svg)](https://www.nvidia.com/en-in/data-center/tesla-t4/)

## Project Overview

This project provides a complete, cloud-deployed ComfyUI-based pipeline combining **Qwen-Image-Edit-2509** advanced AI model with **UltimateSDUpscale** for high-resolution image editing and upscaling. Built for Kaggle's GPU environment, it offers professional-grade image transformation capabilities through an intuitive web interface.

---

##  Key Features

This pipeline represents the cutting edge of AI image processing technology, making professional-grade tools accessible to users without specialized hardware or technical expertise.

### Advanced Image Editing
- **Qwen-Image-Edit-2509**: State-of-the-art AI model for intelligent image transformation
- **GGUF Model Support**: Quantized models for efficient inference with reduced memory usage
- **Aesthetic Transformation**: Apply sophisticated style transfers and visual enhancements
- **LoRA Support**: Transfer learning models for enhanced customization
- **Text Prompt Control**: Fine-tune editing with positive and negative prompt conditioning

### UltimateSDUpscale Integration
- **4x Image Upscaling**: Enhance resolution up to 4x original size
- **Professional Upscaling**: Uses advanced 4x_NMKD-Superscale-SP_178000_G model
- **Quality Preservation**: Maintains image details during upscaling process
- **Configurable Parameters**: Customizable upscaling settings
 
### Technical Features
- **Web-Based Interface**: Real-time workflow management through ComfyUI's node editor
- **Cloud-Based**: Runs entirely on Kaggle's T4 GPU infrastructure , No local installation required
- **Multi-Model Pipeline**: Combines multiple specialized AI models
- **Secure Access**: LocalTunnel encrypted web interface

### Use Cases
- **Digital Art Enhancement**: Transform artistic works with AI assistance
- **Photo Restoration**: Upscale and enhance vintage or low-quality images
- **Style Transfer**: Apply sophisticated aesthetic transformations
- **Professional Workflows**: Integration into existing design pipelines

---

##  Setup Process

### Getting Started
The project is ready for immediate use with provided setup instructions. Just follow the step-by-step guide to begin creating professional-quality image transformations.

### Prerequisites
- Kaggle account with GPU access
- Stable internet connection
- Modern web browser

### Step-by-Step Installation

1. **Upload to Kaggle**
   ```
   - Upload the notebook to your Kaggle account
   - Navigate to the notebook in your Kaggle workspace
   ```

2. **Configure Runtime**
   ```
   - Enable GPU: Accelerator → GPU T4
   - Enable Internet: Settings → Internet → On
   - Verify settings are applied correctly
   ```

3. **Execute Pipeline**
   ```
   - Click "Run All" in the notebook
   - Monitor output for model downloads
   ```

4. **Access Interface**
   ```
   - Copy the LocalTunnel URL from the output
   - Access the URL in your web browser
   - Enter password when prompted (displayed in output , Lookout for an IPv4 address)
   - Wait for ComfyUI initialization (Blank page shown , typically 5-7 minutes)
   ```

5. **Upload Workflow**
   ```
   - Import the provided JSON workflow file
   - Upload your target images to the workflow
   - Enable supplemenatry image nodes if needed 
   ```

6. **Configure Parameters** 
   ```
   - Edit Positive Text Prompt for desired transformation
   - Modify Negative Text Prompt to avoid unwanted effects
   ```

7. **Generate Results**
   ```
   - Click "Run" to start processing
   - Wait for completion (~10 minutes per image)
   - Download results from output directory
   ```

---

# Architecture Logic Explanation

### Core Models
1. **Main Model**: `Qwen-Image-Edit-2509-Q3_K_M.gguf`
   - Quantized image editing model (Q3_K_M)
   - Size: ~4.2GB compressed
   - Source: HuggingFace QuantStack

2. **Text Encoder**: `Qwen2.5-VL-7B-Instruct-Q3_K_M.gguf`
   - Vision-language text encoder
   - Multi-modal understanding
   - Source: Unsloth HuggingFace

3. **Multimodal Projection**: `Qwen2.5-VL-7B-Instruct-mmproj-BF16.gguf`
   - Vision-language alignment layer
   - BF16 precision for quality

4. **VAE**: `qwen_image_vae.safetensors`
   - Variational Autoencoder for image generation
   - Source: Comfy-Org

#### Enhancement Models
5. **Qwen Image Lightning**: `Qwen-Image-Lightning-8steps-V2.0.safetensors`
   - Fast inference model (8 steps)
   - Optimized for speed
   - Source: Lightx2v

6. **Transfer LoRA**: `Transfer_Qwen_Image_Edit_2509.safetensors`
   - Style transfer and enhancement
   - Civitai model repository

#### UltimateSDUpscale Models
7. **Checkpoint**: `majicMIX.safetensors`
   - High-quality upscaling base model
   - Civitai model #176425

8. **Upscaler**: `4x_NMKD-Superscale-SP_178000_G.pth`
   - 4x upscaling neural network
   - Super-resolution algorithm

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     ComfyUI Web Interface                   │
│                  (Port 8188 + LocalTunnel)                  │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────┐
│                  ComfyUI Core Engine                        │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Node Processing Pipeline               │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │    │
│  │  │    Input    │  │ Processing  │  │   Output    │  │    │
│  │  │   Images    │  │   Nodes     │  │   Results   │  │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────┐
│                    Model Layer                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │     Qwen     │  │ UltimateSD   │  │    VAE &     │       │
│  │   Image-Edit │  │   Upscale    │  │   CLIP       │       │
│  │   2509 GGUF  │  │   Models     │  │   Models     │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────┘

```

### Processing Pipeline

1. **Image Loading Phase**
   ```
   Load Image (MAIN/Option1/Option2) → Image Preprocessing
   ```

2. **Text Encoding Phase**
   ```
   TextEncodeQwenImageEditPlus → CLIP Text Encoder → Conditioning Vectors
   ```

3. **Model Loading Phase**
   ```
   Qwen-Image-Edit-2509 → Lightning LoRA → ModelSamplingAuraFlow
   ```

4. **Generation Phase**
   ```
   KSampler → VAE Encoding → Latent Space Processing → VAE Decoding
   ```

5. **Upscaling Phase**
   ```
   UltimateSDUpscale → majicMIX Checkpoint → 4x Upscaler → Final Output
   ```

### Data Flow Architecture

```
Input Images → Text Prompts → Qwen Model → Generation → Upscaling → Output
     ↓             ↓           ↓          ↓          ↓          ↓
   3 Max        Positive    GGUF       Latent     4x Scale   High-Res
   Images      + Negative   Models     Sample    + Quality   Results
```

---

##  License
This project is designed for educational and research purposes.

---


