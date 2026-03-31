# SAM3 Masks

Lichtfeld plugin for text-prompted mask extraction using SAM3.

## Requirements

- NVIDIA GPU with CUDA 12.6
- Hugging Face CLI auth (`huggingface-cli login`) completed before use
- Access to SAM3 (https://huggingface.co/facebook/sam3)

## Usage

1. Click **Get SAM3 Weights** to download the model.
2. Enter comma-separated text prompts (e.g. `person, car`).
3. Adjust confidence, inference resolution, dilation, and fill holes as needed.
4. Click **Extract Masks** to generate masks for all images in the loaded scene.
