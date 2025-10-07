
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score


def load_and_clean(filepath):
    df = pd.read_csv(filepath)
    df = df.drop(columns='package_name')
    df = df.reset_index(drop=True)
    return df


def build_pipeline():
    pipe = Pipeline([
        ('count', CountVectorizer(stop_words='english')),
        ('nb', MultinomialNB())
    ])
    return pipe


def train_and_evaluate(df):
    X_train, X_test, y_train, y_test = train_test_split(
        df['review'], df['polarity'], test_size=0.2, random_state=2025
    )

    pipe = build_pipeline()

    # Hyperparameter grid
    param_grid = {
        'nb__alpha': [0.1, 0.5, 1.0, 2.0],
        'nb__fit_prior': [True, False]
    }

    grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)

    print("Best Parameters:", grid.best_params_)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    print("\n📊 Test Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return best_model


def predict_sentiment(model, text):
    """
    Takes a trained pipeline model and a string or list of strings.
    Returns predicted sentiment(s).
    """
    if isinstance(text, str):
        text = [text]
    preds = model.predict(text)
    return preds



def main():
    data_path = '/workspaces/play-store-sentiment-analysis/data/raw/playstore_reviews.csv'

    df = load_and_clean(data_path)
    
    train_and_evaluate(df)


if __name__ == "__main__":
    main()
