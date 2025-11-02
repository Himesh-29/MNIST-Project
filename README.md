# 🎨 MNIST Digit Recognition - Interactive Canvas App

An interactive web application for recognizing handwritten digits using a deep learning model trained on the MNIST dataset. Built with **TensorFlow** and **Streamlit**.

![Model Accuracy](https://img.shields.io/badge/Accuracy-97.82%25-brightgreen)
![Python Version](https://img.shields.io/badge/Python-3.9-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)

## 🌟 Features

- **Interactive Canvas**: Draw digits directly on the webpage with your mouse or touch
- **Real-time Prediction**: Get instant predictions with confidence scores
- **Top 3 Predictions**: See the top 3 most likely digits with their probabilities
- **Beautiful UI**: Modern gradient design with smooth animations
- **Adjustable Brush**: Customize your drawing experience
- **Mobile Support**: Touch-friendly canvas works on all devices
- **100% Free**: Host forever on Streamlit Cloud - no credit card required!

## 🚀 Quick Start (Local)

### Prerequisites

- Python 3.9+
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Himesh-29/MNIST-Project.git
   cd MNIST-Project
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run streamlit_app.py
   ```

4. **Open in browser**
   The app will automatically open at `http://localhost:8501`

## 🎮 How to Use

1. **Draw a digit** on the canvas (0-9)
   - Adjust brush size using the slider
   - Draw with your mouse or touch
   - Draw in the center for best results

2. **Click "🔍 Predict"** button
   - See instant predictions!
   - Main prediction with confidence
   - Top 3 predictions with probabilities

3. **View results**
   - Large digit display
   - Confidence bar
   - All probabilities (expandable)

4. **Clear** and start over!

## 📊 Model Details

- **Architecture**: Sequential Neural Network with 5 dense layers
- **Training Dataset**: MNIST (60,000 training images, 10,000 test images)
- **Input Shape**: 28×28 grayscale images
- **Output**: 10 classes (digits 0-9)
- **Accuracy**: **97.82%** on test set
- **Normalization**: Pixel values 0-1 range (as per `tf.keras.utils.normalize`)
- **Activation**: ReLU for hidden layers, Softmax for output
- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy

## 📁 Project Structure

```
MNIST-Project/
│
├── streamlit_app.py              # Main Streamlit application
├── requirements.txt               # Python dependencies
├── .streamlit/
│   └── config.toml               # Streamlit theme configuration
│
├── MNIST_epic_number_reader.model/  # Saved TensorFlow model
│   ├── saved_model.pb
│   └── variables/
│       ├── variables.data-00000-of-00001
│       └── variables.index
│
├── MNIST project by Himesh Maniyar.ipynb  # Original training notebook
│
├── README.md                     # This file
└── .gitignore                    # Git ignore rules
```

## 🛠️ Technologies Used

- **Streamlit**: Interactive web app framework
- **landingai-streamlit-drawable-canvas**: Drawing canvas component (maintained fork)
- **TensorFlow/Keras**: Deep learning model
- **NumPy**: Numerical operations
- **Pillow**: Image processing
- **Python**: Core programming language

## 🔬 How It Works

1. **User draws** on the 280×280 canvas
2. **Image processed**: 
   - Converted to grayscale
   - Resized to 28×28 pixels (MNIST input size)
   - Normalized to 0-1 range (exactly like training data)
3. **Model predicts**: 
   - Input: (1, 28, 28) numpy array
   - Output: 10 probabilities (digits 0-9)
4. **Results displayed**: 
   - Main prediction with confidence
   - Top 3 predictions
   - Visual processed image

## 🎯 Drawing Tips

- ✅ Draw in the **center** of the canvas
- ✅ Make digits **clear and bold**
- ✅ Use **thick strokes** (adjust brush size)
- ✅ Try to match **printed number style**
- ❌ Avoid extra lines or marks
- ❌ Avoid very thin or shaky lines

## 🔧 Customization

### Adjust Theme

Edit `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#667eea"  # Change primary color
backgroundColor = "#f0f2f6"  # Change background
```

### Update Model

Re-train the model using the Jupyter notebook and save with:
```python
model.save('MNIST_epic_number_reader.model')
```

### Modify Features

The Streamlit app is easy to extend. Edit `streamlit_app.py` to add:
- Different brush styles
- Color options
- History tracking
- Export predictions
- More visualizations

## 📚 Learning Resources

- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [LandingAI Drawable Canvas](https://github.com/landing-ai/streamlit-drawable-canvas)
- [Streamlit Cloud Deployment](https://docs.streamlit.io/streamlit-community-cloud)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 👨‍💻 Author

**Himesh Maniyar**  
- GitHub: [@Himesh-29](https://github.com/Himesh-29)

## 🙏 Acknowledgments

- MNIST dataset creators
- TensorFlow team
- Streamlit team
- LandingAI (streamlit-drawable-canvas fork)
- All contributors and users
