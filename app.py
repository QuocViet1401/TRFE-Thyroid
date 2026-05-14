import streamlit as st
import torch
import numpy as np
from PIL import Image
import os
from pathlib import Path

# Import model
from model.trfeplus import TRFEPLUS

st.set_page_config(page_title="Thyroid Ultrasound Analysis", layout="wide")

# Custom CSS for better styling
st.markdown("""
    <style>
    .abnormal {
        background-color: #ffcccc;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ff0000;
    }
    .normal {
        background-color: #ccffcc;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #00cc00;
    }
    .result-title {
        font-size: 32px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Thyroid Ultrasound Analysis")
st.write("AI detection system for thyroid nodules")

# Device
device = torch.device('cpu')

@st.cache_resource
def load_model():
    model = TRFEPLUS(in_ch=3, out_ch=1)
    
    # Trained model path (same path where train.py saves it)
    checkpoint_path = Path('run/trfeplus/fold0/trfeplus_best.pth')
    
    if checkpoint_path.exists():
        try:
            state = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state)
            st.success(f"Trained model loaded successfully!")
            return model.to(device).eval()
        except Exception as e:
            st.error(f"Error loading model: {e}")
    
    # Model not trained
    st.warning("UNTRAINED MODEL - Run TRAIN_ONCE.bat first to train the model")
    return model.to(device).eval()

def predict(image, model):
    """Predict segmentation on PIL Image"""
    # Convert PIL Image to numpy
    img_np = np.array(image, dtype=np.float32)
    
    # Resize to 224x224
    from PIL import Image as PILImage
    if isinstance(image, PILImage.Image):
        img_resized = image.resize((224, 224))
        img_np = np.array(img_resized, dtype=np.float32)
    else:
        from scipy.ndimage import zoom
        zoom_factor = 224 / img_np.shape[0]
        img_np = zoom(img_np, (zoom_factor, zoom_factor, 1), order=1)
    
    # Normalize
    img_np = img_np / 255.0  # Scale to [0, 1]
    img_np = (img_np - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    
    # Convert to tensor: (C, H, W)
    img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float()
    img_tensor = img_tensor.unsqueeze(0).to(device)  # Add batch dimension
    
    with torch.no_grad():
        nodule, thyroid, _ = model(img_tensor)
    
    # Get segmentation mask
    mask = (nodule[0, 0].sigmoid() > 0.5).cpu().numpy().astype(np.uint8) * 255
    conf = nodule[0, 0].sigmoid().cpu().numpy()
    
    return mask, conf

def analyze_abnormality(mask, conf):
    """Analyze if there are abnormalities (nodules) in the image"""
    nodule_pixels = np.sum(mask > 127)
    total_pixels = mask.shape[0] * mask.shape[1]
    nodule_percentage = (nodule_pixels / total_pixels) * 100
    
    # Average confidence in nodule region
    nodule_region = mask > 127
    if np.any(nodule_region):
        avg_confidence = np.mean(conf[nodule_region])
    else:
        avg_confidence = 0.0
    
    abnormality_threshold = 3.0  # 3% of image area
    confidence_threshold = 0.529   
    
    is_abnormal = False
    severity = "Normal"
    confidence_score = 0.0
    
    if nodule_percentage > abnormality_threshold and avg_confidence > confidence_threshold:
        is_abnormal = True
        if nodule_percentage > 5.0:
            severity = "Moderate"
        if nodule_percentage > 10.0:
            severity = "Severe"
        confidence_score = min(avg_confidence * 100, 100)
    
    return {
        'is_abnormal': is_abnormal,
        'nodule_percentage': nodule_percentage,
        'nodule_pixels': nodule_pixels,
        'avg_confidence': avg_confidence,
        'severity': severity,
        'confidence_score': confidence_score
    }

model = load_model()

# Main app
uploaded_file = st.file_uploader("Upload Thyroid Ultrasound Image", type=['png', 'jpg', 'jpeg', 'bmp'])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    
    # Perform analysis
    with st.spinner("Analyzing image..."):
        mask, pred = predict(image, model)
        analysis = analyze_abnormality(mask, pred)
    
    # Display results
    st.markdown("---")
    
    # Main result box
    if analysis['is_abnormal']:
        st.markdown(f"""
            <div class="abnormal">
            <div class="result-title" style="color: #ff0000;">ABNORMAL</div>
            <p style="font-size: 18px;"><strong>Thyroid Nodule Detected</strong></p>
            <p>Severity: <strong>{analysis['severity']}</strong></p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="normal">
            <div class="result-title" style="color: #00aa00;">NORMAL</div>
            <p style="font-size: 18px;"><strong>No Significant Abnormality</strong></p>
            <p>Thyroid appears normal</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Detailed analysis
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, width=300)
    
    with col2:
        st.subheader("Nodule Detection")
        st.image(mask, width=300)
    
    with col3:
        st.subheader("Confidence Map")
        st.image(pred, width=300)
    
    st.markdown("---")
    
    # Detailed statistics
    col_stats1, col_stats2, col_stats3 = st.columns(3)
    
    with col_stats1:
        st.metric("Nodule Area", f"{analysis['nodule_percentage']:.2f}%")
    
    with col_stats2:
        st.metric("Detected Pixels", f"{int(analysis['nodule_pixels'])}")
    
    with col_stats3:
        st.metric("Confidence", f"{analysis['confidence_score']:.1f}%")
    
    st.markdown("---")
    
    # Clinical recommendations
    st.subheader("Clinical Findings")
    
    if analysis['is_abnormal']:
        st.warning(f"""
        **Findings:**
        - Thyroid nodule detected
        - Nodule size: {analysis['nodule_percentage']:.2f}% of analyzed area
        - Detection confidence: {analysis['confidence_score']:.1f}%
        - Severity: {analysis['severity']}
        
        **Recommendations:**
        - Further clinical evaluation recommended
        - Consider ultrasound follow-up
        - Consult with endocrinologist if needed
        - Consider additional imaging (CT/MRI) if indicated
        """)
    else:
        st.success(f"""
        **Findings:**
        - No significant nodules detected
        - Thyroid parenchyma appears normal
        - Detection threshold not met
        
        **Recommendations:**
        - Continue routine monitoring
        - Normal follow-up schedule
        - Routine clinical examination as needed
        """)
    
    st.markdown("---")
    st.caption("This is an AI-assisted analysis. Always consult with a qualified physician for clinical decisions.")

else:
    st.write("Upload an image to get started")

