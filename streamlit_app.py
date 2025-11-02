import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import base64
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="MNIST Digit Recognition",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding-top: 1rem;
    }
    .stButton>button {
        width: 100%;
        height: 50px;
        font-size: 18px;
        font-weight: bold;
        border-radius: 10px;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        margin: 20px 0;
    }
    .digit-display {
        font-size: 120px;
        font-weight: 900;
        color: white;
        text-shadow: 0 0 20px rgba(255,255,255,0.5);
        margin: 20px 0;
    }
    .confidence-text {
        font-size: 24px;
        color: white;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    """Load the TensorFlow model (cached for performance)"""
    try:
        model = tf.keras.models.load_model('MNIST_epic_number_reader.model')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Initialize model
model = load_model()

# Title and description
st.title("🎨 MNIST Digit Recognition")
st.markdown("""
    <div style='text-align: center; padding: 10px 0;'>
        <p style='font-size: 18px; color: #666;'>
            Draw a digit from 0-9 on the canvas and watch AI predict it!
        </p>
        <p style='font-size: 14px; color: #888;'>
            Model Accuracy: 97.82% | Powered by TensorFlow & Streamlit
        </p>
    </div>
""", unsafe_allow_html=True)

# Main layout
col1, col2 = st.columns([1.5, 1])

with col1:
    st.markdown("### ✏️ Draw Your Digit")
    
    # Create canvas with Streamlit Components
    canvas_html = """
    <div style="text-align: center;">
        <canvas id="digitCanvas" width="280" height="280" style="border: 3px solid #667eea; background: white; cursor: crosshair; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.2);"></canvas>
        <br><br>
        <button onclick="clearCanvas()" style="padding: 15px 40px; font-size: 16px; background: #ef4444; color: white; border: none; border-radius: 10px; cursor: pointer; font-weight: bold; margin-right: 10px;">
            🗑️ Clear
        </button>
        <p style="color: #666; margin-top: 15px; font-size: 14px;">
            💡 <strong>Tip:</strong> Draw with your mouse or finger (on touch devices)
        </p>
    </div>

    <script>
        const canvas = document.getElementById('digitCanvas');
        const ctx = canvas.getContext('2d');
        let isDrawing = false;
        
        // Set drawing properties
        ctx.strokeStyle = 'black';
        ctx.lineWidth = 20;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        
        // Mouse events
        canvas.addEventListener('mousedown', (e) => {
            isDrawing = true;
            const rect = canvas.getBoundingClientRect();
            ctx.beginPath();
            ctx.moveTo(e.clientX - rect.left, e.clientY - rect.top);
        });
        
        canvas.addEventListener('mousemove', (e) => {
            if (!isDrawing) return;
            const rect = canvas.getBoundingClientRect();
            ctx.lineTo(e.clientX - rect.left, e.clientY - rect.top);
            ctx.stroke();
        });
        
        canvas.addEventListener('mouseup', () => { isDrawing = false; });
        canvas.addEventListener('mouseout', () => { isDrawing = false; });
        
        // Touch events
        canvas.addEventListener('touchstart', (e) => {
            e.preventDefault();
            isDrawing = true;
            const touch = e.touches[0];
            const rect = canvas.getBoundingClientRect();
            ctx.beginPath();
            ctx.moveTo(touch.clientX - rect.left, touch.clientY - rect.top);
        });
        
        canvas.addEventListener('touchmove', (e) => {
            e.preventDefault();
            if (!isDrawing) return;
            const touch = e.touches[0];
            const rect = canvas.getBoundingClientRect();
            ctx.lineTo(touch.clientX - rect.left, touch.clientY - rect.top);
            ctx.stroke();
        });
        
        canvas.addEventListener('touchend', () => { isDrawing = false; });
        
        function clearCanvas() {
            ctx.fillStyle = 'white';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.fillStyle = 'black';
        }
        
        // Store canvas data globally
        window.getCanvasData = function() {
            return canvas.toDataURL('image/png');
        };
    </script>
    """
    
    # Display canvas
    st.components.v1.html(canvas_html, height=400)
    
    # Buttons
    col_clear, col_predict = st.columns(2)
    with col_clear:
        if st.button("🗑️ Clear Canvas", use_container_width=True):
            st.rerun()
    
    with col_predict:
        predict_clicked = st.button("🔍 Predict", use_container_width=True)

with col2:
    st.markdown("### 🎯 Prediction")
    
    if predict_clicked:
        st.info("💡 **Note**: Please use the file uploader method below to upload your drawing. The canvas-to-SessionState feature is being improved.")
    
    # Alternative: File upload for canvas screenshot
    st.markdown("### 📤 Upload Drawing")
    st.markdown("**Method**: Draw on the canvas above, take a screenshot, save it, and upload below.")
    
    uploaded_file = st.file_uploader(
        "Upload your digit image",
        type=['png', 'jpg', 'jpeg'],
        help="Draw on the canvas above, capture it, save the image, and upload here"
    )
    
    if uploaded_file is not None:
        # Process the uploaded image
        image = Image.open(uploaded_file)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to grayscale
        image = image.convert('L')
        
        # Resize to 28x28 (MNIST input size)
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Normalize to 0-1 range (as per MNIST training data)
        img_array = img_array.astype('float32') / 255.0
        
        # Reshape for model input (batch_size, height, width)
        img_array = img_array.reshape(1, 28, 28)
        
        if model is not None:
            with st.spinner("Analyzing..."):
                # Make prediction
                predictions = model.predict(img_array, verbose=0)
                predicted_digit = np.argmax(predictions[0])
                confidence = float(predictions[0][predicted_digit])
                
                # Get top 3 predictions
                top_indices = np.argsort(predictions[0])[-3:][::-1]
                top_predictions = [
                    {'digit': int(idx), 'confidence': float(predictions[0][idx])}
                    for idx in top_indices
                ]
                
                # Display main prediction
                prediction_html = f"""
                <div class="prediction-box">
                    <div class="digit-display">{predicted_digit}</div>
                    <div class="confidence-text">
                        Confidence: {confidence*100:.2f}%
                    </div>
                </div>
                """
                st.markdown(prediction_html, unsafe_allow_html=True)
                
                # Confidence progress bar
                st.progress(confidence)
                
                # Top 3 predictions
                st.markdown("### 📊 Top 3 Predictions")
                for i, pred in enumerate(top_predictions):
                    label = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                    st.write(f"{label} **Digit {pred['digit']}**: {pred['confidence']*100:.1f}%")
                    st.progress(pred['confidence'])
                
                # Show what model sees (processed image)
                st.markdown("### 🖼️ Processed Image")
                processed_img = Image.fromarray((img_array[0] * 255).astype(np.uint8))
                st.image(processed_img.resize((280, 280)), caption="What the model sees (28x28)", use_container_width=True)
                
                # Show prediction probabilities
                with st.expander("📊 See all probabilities"):
                    for i, prob in enumerate(predictions[0]):
                        if prob > 0.001:  # Only show probabilities > 0.1%
                            bar_color = "#10b981" if i == predicted_digit else "#94a3b8"
                            st.write(f"**{i}**: {prob*100:.2f}%")
                            st.progress(prob)
        else:
            st.error("Model not loaded. Please check model files.")
    
    else:
        st.info("Draw a digit on the canvas and upload it to see predictions!")
        
        # Instructions
        st.markdown("""
        ### 💡 How to Use
        1. **Draw** a digit 0-9 on the canvas above
        2. **Take a screenshot** of the canvas
        3. **Save** as PNG or JPG
        4. **Upload** using the file uploader
        5. See instant predictions!
        """)
        
        # Tips
        with st.expander("🎯 Drawing Tips"):
            st.markdown("""
            - Draw in the **center** of the canvas
            - Make digits **clear and bold**
            - Try to match printed number style
            - Avoid extra lines or marks
            - Best results with **thick strokes**
            """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 20px 0; color: #888;'>
        <p>Built with ❤️ using TensorFlow & Streamlit</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 📚 About")
    st.markdown("""
    This MNIST digit recognition app uses a deep neural network 
    trained on 60,000 handwritten digits.
    
    **Model Stats**:
    - 5 Dense Layers (128 neurons each)
    - ReLU Activation
    - Adam Optimizer
    - 97.82% Accuracy
    
    **Dataset**: MNIST (Modified NIST)
    """)
    
    st.markdown("### 🔬 Technical Details")
    st.markdown("""
    **Input Format**:
    - 28×28 grayscale images
    - White background, black digit
    - Normalized to 0-1 range
    
    **Processing**:
    - Upload → 28×28 resize
    - Grayscale conversion
    - Pixel normalization
    - Direct model inference
    """)
    
    st.markdown("### 🔗 Deploy Your Own")
    st.markdown("""
    Host this free on Streamlit Cloud:
    1. Push to GitHub
    2. Go to share.streamlit.io
    3. Connect repository
    4. Deploy instantly!
    """)
    
    st.markdown("---")
    st.caption("Model trained on MNIST dataset | 97.82% test accuracy")
