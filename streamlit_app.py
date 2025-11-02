import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

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
    
    # Specify canvas parameters
    stroke_width = st.slider("Brush size:", 1, 30, 15, key="stroke_width")
    
    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 1.0)",  # Black fill
        stroke_width=stroke_width,
        stroke_color="rgba(0, 0, 0, 1.0)",  # Black stroke
        background_color="#ffffff",  # White background
        update_streamlit=True,
        height=280,
        width=280,
        drawing_mode="freedraw",  # Free drawing mode
        key="canvas",
    )
    
    # Buttons for canvas
    col_clear, col_predict = st.columns(2)
    with col_clear:
        if st.button("🗑️ Clear Canvas", use_container_width=True):
            st.rerun()
    
    with col_predict:
        predict_clicked = st.button("🔍 Predict", use_container_width=True)

with col2:
    st.markdown("### 🎯 Prediction")
    
    if predict_clicked and canvas_result.image_data is not None:
        # Convert canvas image to PIL Image
        img_data = canvas_result.image_data
        img = Image.fromarray(img_data.astype('uint8'), 'RGBA')
        
        # Convert to grayscale
        img = img.convert('L')
        
        # Resize to 28x28 (MNIST input size)
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Model expects black digit on white background
        # Canvas creates white background with black drawing, which is correct
        
        # Normalize to 0-1 range (as per MNIST training data)
        # MNIST uses normalize() which divides by 255
        img_array = img_array.astype('float32') / 255.0
        
        # Reshape for model input (batch_size, height, width)
        img_array = img_array.reshape(1, 28, 28)
        
        if model is not None:
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
            # Display the 28x28 image scaled up
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
    
    elif predict_clicked:
        st.warning("⚠️ Please draw a digit first!")
    else:
        st.info("Draw a digit and click Predict to see results here!")
        
        # Instructions
        st.markdown("""
        ### 💡 How to Use
        1. **Draw** a digit 0-9 on the canvas
        2. **Adjust** brush size if needed
        3. **Click Predict** to see results
        4. **Clear** to start over
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

# Sidebar with additional info
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
    - Canvas → 28×28 resize
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
