import streamlit as st
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
    /* Hide canvas toolbar completely */
    .stSelectbox, .stNumberInput {
        display: none !important;
    }
    /* Canvas border styling */
    canvas {
        border: 2px solid #667eea !important;
        border-radius: 10px !important;
    }
    </style>
""", unsafe_allow_html=True)

# Load model with lazy import
@st.cache_resource
def load_model():
    """Load the TensorFlow model (cached for performance)"""
    try:
        # Lazy import of tensorflow for faster startup
        import tensorflow as tf
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

# Initialize session state
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None

# Function to process canvas image and make prediction
def predict_digit(canvas_data):
    """Process canvas image and predict digit"""
    if canvas_data.image_data is None:
        return None
    
    # Convert canvas data to PIL Image
    img = Image.fromarray(canvas_data.image_data.astype('uint8'), 'RGBA')
    
    # Convert to grayscale
    img = img.convert('L')
    
    # Resize to 28x28 (MNIST input size)
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    
    # Convert to numpy array and normalize
    img_array = np.array(img).astype('float32') / 255.0
    
    # Invert colors (MNIST expects white digits on black background)
    img_array = 1 - img_array
    
    # Reshape for model input
    img_array = img_array.reshape(1, 28, 28)
    
    # Load model and make prediction
    model = load_model()
    if model is not None:
        # Lazy import of tensorflow for prediction
        import tensorflow as tf
        predictions = model(img_array, training=False).numpy()
        predicted_digit = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_digit])
        
        # Get top 3 predictions
        top_indices = np.argsort(predictions[0])[-3:][::-1]
        top_predictions = [
            {'digit': int(idx), 'confidence': float(predictions[0][idx])}
            for idx in top_indices
        ]
        
        return {
            'digit': predicted_digit,
            'confidence': confidence,
            'top_predictions': top_predictions,
            'all_predictions': predictions[0],
            'processed_image': img_array[0]
        }
    return None

# Main layout
col1, col2 = st.columns([1.5, 1])

with col1:
    st.markdown("### ✏️ Draw Your Digit")
    
    # Initialize clear canvas state
    if 'canvas_key' not in st.session_state:
        st.session_state.canvas_key = 0
    
    # Create drawable canvas
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0.0)",  # Transparent fill
        stroke_width=20,
        stroke_color="#000000",
        background_color="#FFFFFF",
        background_image=None,
        update_streamlit=True,
        height=280,
        width=280,
        drawing_mode="freedraw",
        point_display_radius=0,
        display_toolbar=False,  # Hide the toolbar with icons
        key=f"canvas_{st.session_state.canvas_key}",
    )
    
    # Buttons
    col_clear, col_predict = st.columns(2)
    with col_clear:
        if st.button("🗑️ Clear Canvas", use_container_width=True):
            st.session_state.canvas_key += 1  # Change key to clear canvas
            st.session_state.prediction_result = None  # Clear prediction
            st.rerun()
    
    with col_predict:
        if st.button("🔍 Predict", use_container_width=True):
            if canvas_result.image_data is not None:
                with st.spinner("Analyzing your drawing..."):
                    result = predict_digit(canvas_result)
                    st.session_state.prediction_result = result
                    st.rerun()
            else:
                st.warning("Please draw something on the canvas first!")

with col2:
    st.markdown("### 🎯 Prediction")
    
    # Display prediction results
    if st.session_state.prediction_result is not None:
        result = st.session_state.prediction_result
        
        # Display main prediction
        prediction_html = f"""
        <div class="prediction-box">
            <div class="digit-display">{result['digit']}</div>
            <div class="confidence-text">
                Confidence: {result['confidence']*100:.2f}%
            </div>
        </div>
        """
        st.markdown(prediction_html, unsafe_allow_html=True)
        
        # Confidence progress bar
        st.progress(result['confidence'])
        
        # Top 3 predictions
        st.markdown("### 📊 Top 3 Predictions")
        for i, pred in enumerate(result['top_predictions']):
            label = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
            st.write(f"{label} **Digit {pred['digit']}**: {pred['confidence']*100:.1f}%")
            st.progress(pred['confidence'])
        
        # Show what model sees (processed image)
        st.markdown("### 🖼️ Processed Image")
        processed_img = Image.fromarray((result['processed_image'] * 255).astype(np.uint8))
        st.image(processed_img.resize((140, 140)), caption="What the model sees (28x28)", use_container_width=False)
        
        # Show prediction probabilities
        with st.expander("📊 See all probabilities"):
            for i, prob in enumerate(result['all_predictions']):
                if prob > 0.001:  # Only show probabilities > 0.1%
                    st.write(f"**{i}**: {prob*100:.2f}%")
                    st.progress(prob)
    
    else:
        st.info("Draw a digit on the canvas and click **Predict** to see results!")
        
        # Instructions
        st.markdown("""
        ### 💡 How to Use
        1. **Draw** a digit 0-9 on the canvas
        2. **Click Predict** to get instant results
        3. **Clear Canvas** to start over
        4. See live predictions with confidence scores!
        """)
        
        # Tips
        with st.expander("🎯 Drawing Tips"):
            st.markdown("""
            - Draw in the **center** of the canvas
            - Make digits **clear and bold**
            - Try to match printed number style
            - Avoid extra lines or marks
            - Best results with **thick strokes**
            - Draw larger digits for better accuracy
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
    - Canvas → 28×28 resize
    - Grayscale conversion
    - Color inversion (white on black)
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
