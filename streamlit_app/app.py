"""
Streamlit application for interactive image inpainting.
"""
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.generator import Generator
from utils.data_loader import MaskGenerator

def load_model(checkpoint_path):
    """Load the trained generator model."""
    generator = Generator(in_channels=4, out_channels=3)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()
    return generator

def process_image(image, mask, model):
    """Process image through the model."""
    # Prepare image
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    image_tensor = transform(image).unsqueeze(0)
    mask_tensor = torch.from_numpy(mask).permute(2, 0, 1).unsqueeze(0)
    
    # Apply mask to image
    masked_image = image_tensor * (1 - mask_tensor)
    
    # Generate inpainted image
    with torch.no_grad():
        inpainted = model(masked_image, mask_tensor)
    
    # Convert back to image
    inpainted = inpainted.squeeze(0).permute(1, 2, 0)
    inpainted = ((inpainted * 0.5 + 0.5) * 255).numpy().astype(np.uint8)
    return Image.fromarray(inpainted)

def main():
    st.title('Image Inpainting Demo')
    st.write('Upload an image and select a region to inpaint')
    
    # Load model
    model = load_model('checkpoints/latest.pt')
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file).convert('RGB')
        image = image.resize((256, 256))  # Resize to match model input
        st.image(image, caption='Original Image', use_column_width=True)
        
        # Mask options
        mask_type = st.selectbox(
            'Select mask type',
            ('Rectangle', 'Ellipse')
        )
        
        # Mask parameters
        st.sidebar.header('Mask Parameters')
        if mask_type == 'Rectangle':
            x = st.sidebar.slider('X position', 0, 256, 128)
            y = st.sidebar.slider('Y position', 0, 256, 128)
            width = st.sidebar.slider('Width', 10, 128, 64)
            height = st.sidebar.slider('Height', 10, 128, 64)
        else:  # Ellipse
            center_x = st.sidebar.slider('Center X', 0, 256, 128)
            center_y = st.sidebar.slider('Center Y', 0, 256, 128)
            axis_x = st.sidebar.slider('X axis', 10, 128, 64)
            axis_y = st.sidebar.slider('Y axis', 10, 128, 64)
            angle = st.sidebar.slider('Angle', 0, 360, 0)
        
        # Create mask
        mask_generator = MaskGenerator(256, 256)
        if mask_type == 'Rectangle':
            mask = np.zeros((256, 256, 1), np.float32)
            mask[y:y+height, x:x+width] = 1
        else:
            mask = np.zeros((256, 256, 1), np.float32)
            cv2.ellipse(mask, (center_x, center_y), (axis_x, axis_y),
                       angle, 0, 360, 1, -1)
        
        # Display masked image
        image_array = np.array(image)
        mask_3channel = np.repeat(mask, 3, axis=2)  # Repeat mask for RGB channels
        masked_image = image_array * (1 - mask_3channel)
        st.image(masked_image.astype(np.uint8), caption='Masked Image', use_column_width=True)
        
        if st.button('Inpaint'):
            with st.spinner('Inpainting...'):
                inpainted = process_image(image, mask, model)
                st.image(inpainted, caption='Inpainted Image', use_column_width=True)

if __name__ == '__main__':
    main()
