# Traffic Sign Detection and Recognition Using CNN

## Overview
This project focuses on developing a deep learning-based system for detecting and recognizing traffic signs using Convolutional Neural Networks (CNN) and Logistic Regression models. The system leverages the German Traffic Sign Detection Benchmark (GTSDB) dataset to classify traffic signs based on their shape, size, and color, even in challenging conditions such as partial obscuring, blurring, or fading. A Tkinter-based GUI allows users to upload images or capture them in real-time for sign classification, with results displayed on-screen and optionally announced via text-to-speech.

## Features
- **Traffic Sign Classification**: Utilizes CNN and Logistic Regression models to classify traffic signs with high accuracy.
- **Dataset**: Uses the GTSDB dataset, containing diverse traffic sign images for robust training and testing.
- **Preprocessing**: Includes noise removal, image enhancement, and data normalization to improve model performance.
- **GUI**: Built with Tkinter for user-friendly interaction, supporting image uploads and real-time camera input.
- **Text-to-Speech**: Integrates `pyttsx3` for audio alerts of detected traffic signs.
- **Model Comparison**: Evaluates CNN (99.13% accuracy) and Logistic Regression (93.05% accuracy) models using confusion matrices and accuracy graphs.

## Technologies Used
- **Programming Language**: Python
- **Libraries**:
  - NumPy: For array and matrix operations
  - Pandas: For data manipulation and analysis
  - OpenCV: For image processing
  - TensorFlow & Keras: For building and training CNN models
  - Matplotlib: For plotting accuracy and loss graphs
  - Scikit-learn: For Logistic Regression and metrics
  - Pyttsx3: For text-to-speech conversion
  - Tkinter: For GUI development
- **Dataset**: German Traffic Sign Detection Benchmark (GTSDB)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/traffic-sign-detection-cnn.git
   ```
2. Navigate to the project directory:
   ```bash
   cd traffic-sign-detection-cnn
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Download the GTSDB dataset and place it in the `data/` folder (or update the dataset path in the code).

## Usage
1. Run the main script to launch the GUI:
   ```bash
   python main.py
   ```
2. Use the Tkinter interface to:
   - Upload a traffic sign image.
   - Capture an image in real-time using a connected camera.
3. The system will display the predicted traffic sign class and provide an audio alert (if enabled).
4. View model performance metrics (accuracy, loss, confusion matrix) in the generated plots.

## Project Structure
```
traffic-sign-detection-cnn/
├── Data/                     # GTSDB dataset (not included, must be downloaded), captures by the camera are also stored here
├── major.ipynb               # Contains the code for data preprocessing, model, and GUI implementation
├── logisticregression.ipynb  # Logistic Regression model definition and training
├── MajorProjectFinal.pdf     # Final Report
├── requirements.txt          # List of dependencies
└── README.md                 # Project documentation
```

## Results
- **CNN Model Accuracy**: 99.13% on the validation set.
- **Logistic Regression Model Accuracy**: 93.05% on the test set.
- **Performance Visualization**: Accuracy and loss vs. epochs plots, along with confusion matrices, are generated to compare model performance.

## Future Scope
- Real-time traffic sign recognition with faster processing for in-vehicle deployment.
- Integration of Auto Focus CNN to isolate traffic signs from complex backgrounds.
- Expansion to support additional traffic sign types and languages (e.g., Arabic signs).
- Implementation in specialized vehicles like bulldozers with driver-assist features.

## Contributors
- Parul Taley (0101CS191076)
- Sakshi Badole (0101CS191103)

## References
- Python Software Foundation: https://www.python.org
- TensorFlow: https://www.tensorflow.org
- Keras: https://www.keras.io
- OpenCV: https://opencv.org
- GTSDB Dataset: https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
