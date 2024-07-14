# Dog Breed Classification Project

## Overview

This project aims to classify dog breeds using pretrained deep learning models (ResNet, AlexNet, and VGG) and OpenAI's GPT-4 model for image analysis. The classifier can identify the breed of a dog from an image and provide a description of the image.

## Features

- Image classification using pretrained models: ResNet18, AlexNet, and VGG16.
- Image description using OpenAI's GPT-4 model.
- Supports image input in local file format.

## Prerequisites

- Python 3.7
- Anaconda or Miniconda for environment management

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/dog-breed-classification.git
    cd dog-breed-classification
    ```

2. **Create and activate a conda environment:**

    ```bash
    conda create --name dog-breed-classifier python=3.6
    conda activate dog-breed-classifier
    ```

3. **Install the required packages:**

    ```bash
    conda install pytorch==1.4.0 torchvision==0.2.1 pillow==7.1.2 -c pytorch
    pip install requests openai
    ```

4. **Set up OpenAI API Key:**

    - Get your API key from [OpenAI](https://platform.openai.com/account/api-keys).
    - Set the `OPENAI_API_KEY` environment variable:

    ```bash
    export OPENAI_API_KEY=your-api-key-here
    ```

    Alternatively, you can set the API key directly in the code:

    ```python
    openai.api_key = 'your-api-key-here'
    ```

## Usage

1. **Prepare the image:**

   Place the image you want to classify in a folder.

2. **Run the classifier:**

    ```
    python check_images.py --dir pet_images/ --arch <model> --dogfile dognames.txt
    ```

## Acknowledgements

This project was created as part of the AWS AI ML Nanodegree program.

