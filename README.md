# 🌸 Iris Species Classifier (Artificial Neural Network)

An interactive web app that classifies Iris flowers into three species using a small feed-forward neural network, built with Python, scikit-learn and Streamlit.

### 🔗 [Live demo → basicirisannmodel.streamlit.app](https://basicirisannmodel.streamlit.app/)

---

## Overview

Move four sliders (sepal and petal measurements) and the model instantly predicts the species along with its confidence for each class. The app also shows how the network trained: accuracy curves, test accuracy and a confusion matrix.

The project grew out of a Jupyter notebook that compared a Perceptron with an ANN on the classic Iris dataset.

## Features

- Real-time prediction from four measurement sliders
- Probability chart for all three species
- Training and validation accuracy curve over 100 epochs
- Test accuracy and confusion matrix on a held-out set
- No dataset upload needed (Iris is loaded from scikit-learn)

## Model

| Item | Setting |
|---|---|
| Architecture | 4 inputs → Dense(16, ReLU) → Dense(8, ReLU) → 3 outputs (softmax) |
| Optimizer | Adam (learning rate 0.001) |
| Loss | Cross-entropy |
| Batch size / epochs | 8 / 100 |
| Data split | 80% train, 20% test (stratified, `random_state=42`) |
| Validation | Last 20% of the training set |
| Preprocessing | `StandardScaler` fitted on the training data only |

**Results:** about 96.7% test accuracy (30 test samples). The original notebook's Perceptron baseline scored about 86.7%.

## Dataset

[Iris dataset](https://archive.ics.uci.edu/dataset/53/iris): 150 samples, 4 features (sepal length, sepal width, petal length, petal width in cm), 3 classes (*Iris-setosa*, *Iris-versicolor*, *Iris-virginica*), 50 samples each.

## Run locally

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
streamlit run app.py
```

## Project structure

```
├── app.py              # Streamlit app (training + UI)
├── requirements.txt    # Python dependencies
└── README.md
```

## Tech stack

Python · Streamlit · scikit-learn · NumPy · pandas

## Notes

- The model trains when the app starts (a few seconds) and is cached afterwards.
- This app uses scikit-learn's `MLPClassifier` with the same architecture as the original Keras model, to keep deployment light and fast.
- Iris is a small, easy dataset, so results vary slightly between runs and should not be read as a benchmark.

## License

MIT. Feel free to use and adapt.
