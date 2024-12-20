import numpy as np

class MyMultinomialNB:
    def __init__(self, alpha=1.0):
        # alpha: smoothing parameter (Laplace smoothing)
        self.alpha = alpha
        self.class_log_prior_ = None
        self.feature_log_prob_ = None
        self.classes_ = None
        self.is_fitted_ = False

    def fit(self, X, y):
        # Ensure X and y are numpy arrays
        if not isinstance(X, np.ndarray):
            X = X.toarray()  # If X is sparse, convert to dense for simplicity
        y = np.array(y)

        n_samples = X.shape[0]
        self.classes_ = np.unique(y)

        # Count how many samples per class
        class_count = np.array([np.sum(y == c) for c in self.classes_])

        # class log prior
        self.class_log_prior_ = np.log(class_count) - np.log(n_samples)

        # feature counts per class
        feature_count = np.array([np.sum(X[y == c], axis=0) for c in self.classes_])

        # Apply Laplace smoothing
        feature_count_smoothed = feature_count + self.alpha
        feature_log_prob = np.log(feature_count_smoothed) - np.log(feature_count_smoothed.sum(axis=1)[:, np.newaxis])

        # Ensure correct shapes and types
        self.feature_log_prob_ = np.asarray(feature_log_prob)
        self.class_log_prior_ = np.asarray(self.class_log_prior_)

        # Make sure feature_log_prob_ is 2D
        if self.feature_log_prob_.ndim == 1:
            self.feature_log_prob_ = self.feature_log_prob_[np.newaxis, :]

        # Ensure class_log_prior_ is 1D
        if self.class_log_prior_.ndim == 0:
            self.class_log_prior_ = self.class_log_prior_[np.newaxis]

        self.is_fitted_ = True
        return self

    def predict(self, X):
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction.")

        # If X is sparse, let's convert it to dense array for a simpler dimension handling
        if hasattr(X, "toarray"):
            X = X.toarray()

        # Compute joint log likelihood: (N,F) dot (F,C) = (N,C)
        # feature_log_prob_.T is (F,C)
        joint_log_likelihood = X.dot(self.feature_log_prob_.T)

        # Add class log prior
        # class_log_prior_ is (C,)
        # joint_log_likelihood is (N,C)
        joint_log_likelihood = joint_log_likelihood + self.class_log_prior_

        # Argmax over classes
        class_idx = np.argmax(joint_log_likelihood, axis=1)
        return self.classes_[class_idx]
