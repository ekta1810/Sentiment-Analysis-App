import streamlit as st
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.set_page_config(page_title="Sentiment Analysis App", page_icon="📊")

st.title("📊 Sentiment Analysis App")
st.write("Enter a review and check if it is Positive or Negative")

# ---------------- INPUT ----------------
review = st.text_area("Write your review here:")

# ---------------- PREDICTION ----------------
if st.button("Predict Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review")
    else:
        data = vectorizer.transform([review])

        prediction = model.predict(data)[0]

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(data)[0]
            neg = prob[0]
            pos = prob[1]
        else:
            neg, pos = 0, 0

        if prediction == 1:
            st.success("😊 Positive Review")
        else:
            st.error("😡 Negative Review")

        st.write("### Confidence Score")
        st.write(f"Positive: {pos:.2f}")
        st.write(f"Negative: {neg:.2f}")

# ---------------- CONFUSION MATRIX ----------------
st.markdown("---")
st.subheader("📊 Model Evaluation")

if st.button("Show Confusion Matrix"):

    try:
        cm = np.load("confusion_matrix.npy")

        fig, ax = plt.subplots()
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"]
        )

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")

        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error loading confusion matrix: {e}")