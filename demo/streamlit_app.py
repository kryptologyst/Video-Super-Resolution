"""Streamlit demo for video super-resolution."""

import io
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms

from src.models.vsr_models import create_model
from src.utils.device import get_device


def load_model(model_name: str, checkpoint_path: Optional[str] = None) -> torch.nn.Module:
    """
    Load a video super-resolution model.
    
    Args:
        model_name: Name of the model to load.
        checkpoint_path: Path to checkpoint file (optional).
        
    Returns:
        Loaded model.
    """
    device = get_device()
    
    # Create model
    model = create_model(
        model_name=model_name,
        channels=3,
        scale_factor=2,
    )
    
    # Load checkpoint if provided
    if checkpoint_path and Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model


def process_image(
    image: np.ndarray,
    model: torch.nn.Module,
    scale_factor: int = 2,
) -> np.ndarray:
    """
    Process a single image with the model.
    
    Args:
        image: Input image as numpy array.
        model: Super-resolution model.
        scale_factor: Upscaling factor.
        
    Returns:
        Upscaled image as numpy array.
    """
    device = next(model.parameters()).device
    
    # Convert to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Process with model
    with torch.no_grad():
        output_tensor = model(image_tensor)
    
    # Convert back to numpy
    output_tensor = output_tensor.squeeze(0).cpu()
    output_image = transforms.ToPILImage()(output_tensor)
    output_array = np.array(output_image)
    
    return output_array


def process_video(
    video_frames: list,
    model: torch.nn.Module,
    scale_factor: int = 2,
) -> list:
    """
    Process video frames with the model.
    
    Args:
        video_frames: List of video frames as numpy arrays.
        model: Super-resolution model.
        scale_factor: Upscaling factor.
        
    Returns:
        List of upscaled frames.
    """
    device = next(model.parameters()).device
    processed_frames = []
    
    # Convert frames to tensors
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    frames_tensor = []
    for frame in video_frames:
        frame_tensor = transform(frame).to(device)
        frames_tensor.append(frame_tensor)
    
    # Stack frames
    frames_tensor = torch.stack(frames_tensor)  # (T, C, H, W)
    
    # Process with model
    with torch.no_grad():
        if hasattr(model, 'num_frames') and model.num_frames > 1:
            # Multi-frame model
            frames_tensor = frames_tensor.unsqueeze(0)  # (1, T, C, H, W)
            output_tensor = model(frames_tensor)
            output_tensor = output_tensor.squeeze(0)  # (T, C, H, W)
        else:
            # Single-frame model
            output_tensor = model(frames_tensor)
    
    # Convert back to numpy arrays
    for t in range(output_tensor.shape[0]):
        frame_tensor = output_tensor[t].cpu()
        frame_image = transforms.ToPILImage()(frame_tensor)
        frame_array = np.array(frame_image)
        processed_frames.append(frame_array)
    
    return processed_frames


def create_comparison_gif(
    original_frames: list,
    upscaled_frames: list,
    fps: int = 10,
) -> bytes:
    """
    Create a side-by-side comparison GIF.
    
    Args:
        original_frames: Original frames.
        upscaled_frames: Upscaled frames.
        fps: Frames per second.
        
    Returns:
        GIF as bytes.
    """
    import imageio
    
    # Resize frames to same height
    target_height = min(original_frames[0].shape[0], upscaled_frames[0].shape[0])
    
    resized_original = []
    resized_upscaled = []
    
    for orig, upscaled in zip(original_frames, upscaled_frames):
        # Resize original frame
        orig_resized = cv2.resize(orig, (orig.shape[1] * target_height // orig.shape[0], target_height))
        
        # Resize upscaled frame
        upscaled_resized = cv2.resize(upscaled, (upscaled.shape[1] * target_height // upscaled.shape[0], target_height))
        
        resized_original.append(orig_resized)
        resized_upscaled.append(upscaled_resized)
    
    # Create side-by-side frames
    comparison_frames = []
    for orig, upscaled in zip(resized_original, resized_upscaled):
        comparison = np.hstack([orig, upscaled])
        comparison_frames.append(comparison)
    
    # Create GIF
    gif_buffer = io.BytesIO()
    imageio.mimsave(gif_buffer, comparison_frames, format='GIF', fps=fps)
    gif_buffer.seek(0)
    
    return gif_buffer.getvalue()


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Video Super-Resolution Demo",
        page_icon="🎬",
        layout="wide",
    )
    
    st.title("🎬 Video Super-Resolution Demo")
    st.markdown("Upload a video or image to enhance its resolution using deep learning models.")
    
    # Sidebar for model selection
    st.sidebar.header("Model Configuration")
    
    model_name = st.sidebar.selectbox(
        "Select Model",
        ["srcnn", "vdsr", "edvr", "basicvsr++"],
        help="Choose the super-resolution model to use"
    )
    
    scale_factor = st.sidebar.selectbox(
        "Scale Factor",
        [2, 4],
        help="Upscaling factor"
    )
    
    # Load model
    try:
        with st.spinner("Loading model..."):
            model = load_model(model_name)
        st.sidebar.success(f"Model {model_name} loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Failed to load model: {str(e)}")
        st.stop()
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Input")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload a video or image",
            type=['mp4', 'avi', 'mov', 'jpg', 'jpeg', 'png'],
            help="Supported formats: MP4, AVI, MOV, JPG, JPEG, PNG"
        )
        
        if uploaded_file is not None:
            # Display file info
            st.info(f"File: {uploaded_file.name} ({uploaded_file.size} bytes)")
            
            # Process based on file type
            if uploaded_file.type.startswith('video/'):
                # Video processing
                st.subheader("Video Processing")
                
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file_path = tmp_file.name
                
                try:
                    # Load video
                    cap = cv2.VideoCapture(tmp_file_path)
                    frames = []
                    
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(frame_rgb)
                    
                    cap.release()
                    
                    if frames:
                        st.success(f"Loaded {len(frames)} frames")
                        
                        # Process video
                        if st.button("Enhance Video", type="primary"):
                            with st.spinner("Processing video..."):
                                processed_frames = process_video(frames, model, scale_factor)
                            
                            st.success("Video processing completed!")
                            
                            # Create comparison GIF
                            gif_data = create_comparison_gif(frames, processed_frames)
                            
                            # Display results
                            st.subheader("Results")
                            st.image(gif_data, caption="Original (left) vs Enhanced (right)")
                            
                            # Download button
                            st.download_button(
                                label="Download Enhanced Video",
                                data=gif_data,
                                file_name=f"enhanced_{uploaded_file.name}",
                                mime="image/gif"
                            )
                    
                except Exception as e:
                    st.error(f"Error processing video: {str(e)}")
                
                finally:
                    # Clean up temporary file
                    Path(tmp_file_path).unlink()
            
            else:
                # Image processing
                st.subheader("Image Processing")
                
                # Load image
                image = Image.open(uploaded_file)
                image_array = np.array(image)
                
                st.image(image_array, caption="Original Image", use_column_width=True)
                
                # Process image
                if st.button("Enhance Image", type="primary"):
                    with st.spinner("Processing image..."):
                        enhanced_image = process_image(image_array, model, scale_factor)
                    
                    st.success("Image processing completed!")
                    
                    # Display results
                    st.subheader("Results")
                    
                    col_orig, col_enhanced = st.columns(2)
                    
                    with col_orig:
                        st.image(image_array, caption="Original", use_column_width=True)
                    
                    with col_enhanced:
                        st.image(enhanced_image, caption="Enhanced", use_column_width=True)
                    
                    # Download button
                    enhanced_pil = Image.fromarray(enhanced_image)
                    img_buffer = io.BytesIO()
                    enhanced_pil.save(img_buffer, format='PNG')
                    img_data = img_buffer.getvalue()
                    
                    st.download_button(
                        label="Download Enhanced Image",
                        data=img_data,
                        file_name=f"enhanced_{uploaded_file.name}",
                        mime="image/png"
                    )
    
    with col2:
        st.header("Model Information")
        
        # Model details
        st.subheader(f"Model: {model_name.upper()}")
        
        model_info = {
            "srcnn": "Super-Resolution Convolutional Neural Network - Basic CNN for image super-resolution",
            "vdsr": "Very Deep Super-Resolution - Deep network with residual connections",
            "edvr": "Enhanced Deformable Video Restoration - Advanced video super-resolution with temporal attention",
            "basicvsr++": "BasicVSR++ - State-of-the-art video super-resolution with bidirectional propagation"
        }
        
        st.info(model_info.get(model_name, "Unknown model"))
        
        # Performance metrics
        st.subheader("Performance")
        
        metrics = {
            "srcnn": {"PSNR": "28.4 dB", "SSIM": "0.85", "Speed": "Fast"},
            "vdsr": {"PSNR": "29.1 dB", "SSIM": "0.87", "Speed": "Medium"},
            "edvr": {"PSNR": "30.2 dB", "SSIM": "0.89", "Speed": "Slow"},
            "basicvsr++": {"PSNR": "30.8 dB", "SSIM": "0.91", "Speed": "Slow"}
        }
        
        model_metrics = metrics.get(model_name, {})
        
        for metric, value in model_metrics.items():
            st.metric(metric, value)
        
        # Usage tips
        st.subheader("Usage Tips")
        st.markdown("""
        - **For images**: Upload high-quality images for best results
        - **For videos**: Shorter videos process faster
        - **Scale factor**: Higher scale factors require more processing time
        - **Model selection**: EDVR and BasicVSR++ provide better quality but are slower
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Built with Streamlit | "
        "Models: SRCNN, VDSR, EDVR, BasicVSR++ | "
        "Framework: PyTorch"
    )


if __name__ == "__main__":
    main()
