import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(page_title="Iris Species Classifier", page_icon="🌸")
NAMES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]


@st.cache_resource(show_spinner="Training the neural network...")
def train(seed: int = 42):
    iris = load_iris()
    X, y = iris.data, iris.target
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler().fit(X_tr)
    X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)

    # Same split as notebook: last 20% of train is validation
    n_val = int(len(X_tr) * 0.2)
    X_fit, y_fit = X_tr[:-n_val], y_tr[:-n_val]
    X_val, y_val = X_tr[-n_val:], y_tr[-n_val:]

    # 4 -> 16 -> 8 -> 3, ReLU, Adam, batch size 8, 100 epochs
    model = MLPClassifier(hidden_layer_sizes=(16, 8), activation="relu",
                          solver="adam", learning_rate_init=0.001,
                          batch_size=8, random_state=seed)
    rng = np.random.default_rng(seed)
    hist = []
    for _ in range(100):
        idx = rng.permutation(len(X_fit))
        for s in range(0, len(idx), 8):
            b = idx[s:s + 8]
            model.partial_fit(X_fit[b], y_fit[b], classes=[0, 1, 2])
        hist.append((model.score(X_fit, y_fit), model.score(X_val, y_val)))

    pred = model.predict(X_te)
    return model, scaler, hist, accuracy_score(y_te, pred), confusion_matrix(y_te, pred)


model, scaler, hist, test_acc, cm = train()

st.title("🌸 Iris Species Classifier")
st.caption("Neural network 4 → 16 → 8 → 3 (ReLU, softmax output, Adam, batch size 8, 100 epochs)")

st.subheader("Flower measurements")
c1, c2 = st.columns(2)
sl = c1.slider("Sepal length (cm)", 4.3, 7.9, 5.8, 0.1)
sw = c2.slider("Sepal width (cm)", 2.0, 4.4, 3.0, 0.1)
pl = c1.slider("Petal length (cm)", 1.0, 6.9, 4.3, 0.1)
pw = c2.slider("Petal width (cm)", 0.1, 2.5, 1.3, 0.1)

probs = model.predict_proba(scaler.transform([[sl, sw, pl, pw]]))[0]
st.subheader(f"Prediction: {NAMES[int(np.argmax(probs))]}")
st.bar_chart(pd.DataFrame({"Probability": probs}, index=NAMES))

st.subheader("Model performance")
m1, m2 = st.columns(2)
m1.metric("Test accuracy", f"{test_acc:.1%}")
m2.metric("Final validation accuracy", f"{hist[-1][1]:.1%}")
st.line_chart(pd.DataFrame(hist, columns=["Train", "Validation"]).rename_axis("Epoch"))
st.write("Confusion matrix (rows = actual, columns = predicted)")
st.dataframe(pd.DataFrame(cm, index=NAMES, columns=NAMES))
