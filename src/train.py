import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from models.logistic_regression import MyLogisticRegression
from models.naive_bayes import MyMultinomialNB

def load_data(processed_dir):
    train_df = pd.read_csv(os.path.join(processed_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(processed_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(processed_dir, "test.csv"))
    return train_df, val_df, test_df

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='binary')
    return acc, precision, recall, f1

if __name__ == "__main__":
    processed_dir = "../data/processed"
    train_df, val_df, test_df = load_data(processed_dir)

    X_train_text, y_train = train_df['sentence'], train_df['label'].values
    X_val_text, y_val = val_df['sentence'], val_df['label'].values
    X_test_text, y_test = test_df['sentence'], test_df['label'].values

    # Vectorize the text data
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(X_train_text)
    X_val = vectorizer.transform(X_val_text)
    X_test = vectorizer.transform(X_test_text)

    # Train Logistic Regression
    lr_model = MyLogisticRegression(learning_rate=0.1, max_iter=2000, verbose=False)
    lr_model.fit(X_train, y_train)
    lr_acc, lr_precision, lr_recall, lr_f1 = evaluate_model(lr_model, X_val, y_val)

    print("Custom Logistic Regression Val Performance:")
    print(f"Accuracy: {lr_acc:.4f}, Precision: {lr_precision:.4f}, Recall: {lr_recall:.4f}, F1: {lr_f1:.4f}")

    # Train Naive Bayes
    nb_model = MyMultinomialNB(alpha=1.0)
    nb_model.fit(X_train, y_train)
    nb_acc, nb_precision, nb_recall, nb_f1 = evaluate_model(nb_model, X_val, y_val)

    print("\nCustom Naive Bayes Val Performance:")
    print(f"Accuracy: {nb_acc:.4f}, Precision: {nb_precision:.4f}, Recall: {nb_recall:.4f}, F1: {nb_f1:.4f}")

    # Choose final model based on validation F1
    if lr_f1 > nb_f1:
        final_model = lr_model
        model_name = "Custom Logistic Regression"
    else:
        final_model = nb_model
        model_name = "Custom Naive Bayes"

    test_acc, test_precision, test_recall, test_f1 = evaluate_model(final_model, X_test, y_test)
    print(f"\nFinal Model on Test Set ({model_name})")
    print(f"Accuracy: {test_acc:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")
