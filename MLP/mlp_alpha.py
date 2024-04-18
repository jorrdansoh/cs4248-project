# NO CROSS VALIDATION

# Feature Engineering: Stopwords and Oversampling and Stemming
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import pandas as pd
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.neural_network import MLPClassifier
from nltk.corpus import stopwords
import nltk
from nltk import word_tokenize
import string
import numpy as np
from sklearn.preprocessing import MaxAbsScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, confusion_matrix, pair_confusion_matrix, cohen_kappa_score, matthews_corrcoef, multilabel_confusion_matrix, classification_report
import time
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.model_selection import GridSearchCV

# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')
# nltk.download('omw-1.4')
# nltk.download("averaged_perceptron_tagger")
# nltk.download("maxent_ne_chunker")
# nltk.download("words")

STOPWORDS = set(stopwords.words("english"))

X_train = pd.read_csv('fulltrain.csv', header=None).iloc[:, 1]
y_train = pd.read_csv('fulltrain.csv', header=None).iloc[:, 0]

X_test = pd.read_csv('balancedtest.csv', header=None).iloc[:, 1]
y_test = pd.read_csv('balancedtest.csv', header=None).iloc[:, 0]

def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)

    ps = PorterStemmer()
    tokens = [ps.stem(token) for token in tokens]

    return " ".join(tokens)


def features(X_train):
    stop_words_features = []
    for text in X_train:
        words = word_tokenize(text.lower())
        n = len(words)
        if n == 0:
            stop_words_features.append(0)
            continue
        num_stop_words = 0
        for word in words:
            if word in STOPWORDS:
                num_stop_words += 1

        stop_words_features.append(num_stop_words / n * 100)

    stop_words_features = np.array(stop_words_features).reshape(-1, 1)

    result = np.hstack((stop_words_features,))
    # max abs scaler
    result = MaxAbsScaler(copy=False).fit_transform(result)
    return result


start = time.time()

model = ImbPipeline(
    steps=[
        (
            "features",
            FeatureUnion(
                [
                    ("tfidf", TfidfVectorizer(preprocessor=preprocess_text)),
                    ("otherfeatures", FunctionTransformer(features, validate=False)),
                ],
                n_jobs=-1,
            ),
        ),
        ("resampler", RandomOverSampler(random_state=26)),
        ("classifier", MLPClassifier(random_state=26, max_iter=20, verbose=True)),
    ],
    verbose=True,
)
param_grid = {
    # "classifier__hidden_layer_sizes": [
        # (1024),
        # (512),
        #(256),
        # (128),
        # (1024, 512),
        # (512, 256),
        # (256, 128),
        # (1024, 512, 256),
        # (512, 256, 128),
        # (256, 128, 64),
    # ],
    # "classifier__max_iter": [50, 100, 150],
    # "classifier__activation": ["tanh", "relu", "logistic"], 
    # "classifier__solver": ["sgd", "adam"],
    "classifier__alpha": np.logspace(-6, 0, 10), # from 1e-6 to 1
    # "classifier__learning_rate": ["constant", "adaptive"],
}
grid_search = GridSearchCV(model, param_grid=param_grid, scoring="f1_macro", n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)
print("Best parameters set found on development set:")
print(grid_search.best_params_)
y_pred = grid_search.predict(X_test)
f1score_macro = f1_score(y_test, y_pred, average="macro")
f1score_micro = f1_score(y_test, y_pred, average="micro")
print("F1 Macro Score: ", f1score_macro)
print("F1 Micro Score: ", f1score_micro)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(cohen_kappa_score(y_test, y_pred))
print(matthews_corrcoef(y_test, y_pred))
print(multilabel_confusion_matrix(y_test, y_pred))
print(pair_confusion_matrix(y_test, y_pred))
print("Time taken: {}".format(time.time() - start))
