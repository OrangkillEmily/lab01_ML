
import sys
from pathlib import Path
import joblib

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

MODEL_DIR = ROOT_DIR / "models"

# Load Logistic Regression
lr = joblib.load(MODEL_DIR / "model_lr.pkl")

print("===== LOGISTIC REGRESSION =====")
print("Class:", type(lr).__name__)
print("Weights:", lr.weights)
print("Weights shape:", lr.weights.shape)
print("Bias:", lr.bias)

# Load Naive Bayes
nb = joblib.load(MODEL_DIR / "model_nb.pkl")

print("\n===== NAIVE BAYES =====")
print("Class:", type(nb).__name__)
print("Mean:", nb.mean)
print("Mean shape:", nb.mean.shape)
print("Variance:", nb.var)
print("Variance shape:", nb.var.shape)
print("Priors:", nb.priors)

# Load SVM
svm = joblib.load(MODEL_DIR / "model_svm.pkl")

print("\n===== SVM =====")
print("Class:", type(svm).__name__)
print("Weights:", svm.weights)
print("Weights shape:", svm.weights.shape)
print("Bias:", svm.bias)

# Load TF-IDF
vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")

print("\n===== TF-IDF =====")
print("Class:", type(vectorizer).__name__)
print("Vocabulary size:", len(vectorizer.vocabulary_))
print("Some vocabulary:", list(vectorizer.vocabulary_.items())[:20])
