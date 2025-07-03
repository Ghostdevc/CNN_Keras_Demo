✅ CNN_Keras_Demo

# CNN Image Classification with Keras

This project demonstrates a basic convolutional neural network (CNN) built with Keras and TensorFlow for image classification. It is designed for educational purposes and shows how to structure and train a CNN model using a simple dataset.

## Project Overview

- Builds a convolutional neural network using Keras Sequential API
- Trains the model on image data (e.g., cats vs dogs or similar)
- Evaluates accuracy and loss on test data
- Visualizes training progress with plots

## Tech Stack

- Python
- Keras (TensorFlow backend)
- NumPy, Matplotlib
- scikit-learn (for evaluation)

## How to Run

1. Clone the repository:
git clone https://github.com/Ghostdevc/CNN_Keras_Demo.git
cd CNN_Keras_Demo

2. Install dependencies:
pip install -r requirements.txt

3. Prepare your dataset:
- Place your image dataset into appropriate folders (e.g., `train/` and `test/` directories with class subfolders).
- Alternatively, modify the data loading part in `main.py` to use a specific dataset (e.g., CIFAR-10, MNIST).

4. Run the training script:
python main.py


## Key Components

- Data preprocessing with `ImageDataGenerator`
- CNN architecture with Conv2D, MaxPooling2D, Flatten, and Dense layers
- Model training and validation
- Accuracy/loss plotting with Matplotlib

## Notes

- This project is ideal for learning CNN basics and model training workflow.
- Can be extended to use pre-trained models (transfer learning) or more complex architectures.

## License

MIT License
