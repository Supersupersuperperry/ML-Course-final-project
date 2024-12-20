# Final Project Report: [Project Title]

**Course**: CS383 - Introduction to Machine Learning  
**Semester**: Fall 2024  
**Team Members**: [Supersupersuperperry]  
**Instructor**: [Yupei Sun]  

---

## Table of Contents
1. [Abstract](#abstract)
2. [Introduction](#introduction)
3. [Problem Statement](#problem-statement)
4. [Related Work](#related-work)
5. [Data Description](#data-description)
6. [Methodology](#methodology)
7. [Results](#results)
8. [Discussion](#discussion)
9. [Conclusion and Future Work](#conclusion-and-future-work)
10. [References](#references)

---

## Abstract
Provide a brief summary of your project, including the problem tackled, the methodology used, and the key findings. This section should be concise and no more than 150-200 words.

This project addresses the challenge of automating sentiment classification for user-generated text, specifically determining whether a given sentence is positive or negative. I focus on the "Sentiment Labelled Sentences" dataset from the UCI Machine Learning Repository, which contains 3,000 sentences with binary sentiment labels drawn from IMDb, Amazon, and Yelp reviews. My goal is to evaluate two well-known machine learning algorithms—Logistic Regression and Multinomial Naive Bayes—to identify which method yields more accurate and efficient sentiment predictions.

I preprocess the data and split it into training, validation, and test sets. I transform the raw text into numerical features using a bag-of-words approach through CountVectorizer. Both models are implemented from scratch, without relying on off-the-shelf model classes, to gain deeper insight into their underlying mechanics. I train each model, tune hyperparameters, and compare their performance on the validation set.

The results show that Multinomial Naive Bayes provides a higher F1 score on the validation set. When tested, it achieves an accuracy of about 83.29%, outperforming the Logistic Regression model. These findings suggest that Naive Bayes is a suitable and efficient approach for automated sentiment analysis tasks.

## Introduction
Introduce the problem or question your project addresses. Explain its significance and relevance to machine learning. Include a brief overview of your approach and the objectives of the project.

In the realm of machine learning, leveraging automated methods to interpret and classify the sentiment of textual data has become increasingly important. Sentiment analysis—determining whether a given piece of text is positive or negative—plays a pivotal role in various domains. For instance, my father’s small market research company regularly needs to assess customer feedback from surveys to guide product improvements. Manually evaluating each response for its sentiment is tedious and time-consuming. By automating this process, I can streamline their sentiment classification workflow, reducing the human labor required and allowing for more efficient decision-making.

To achieve this, I focus on comparing two well-established yet fundamentally different machine learning algorithms: Logistic Regression and Multinomial Naive Bayes. Both methods have a history of success in text classification tasks, but each offers distinct advantages and trade-offs. Through careful data preprocessing, feature extraction, model training, and evaluation, I aim to determine which algorithm provides more accurate, reliable, and computationally efficient sentiment classification.

By identifying the model that performs best on the "Sentiment Labelled Sentences" dataset, I will not only contribute to the broader understanding of text classification methodologies but also provide a practical, automated tool that can be applied to real-world tasks like those faced by my father’s company. Ultimately, my goal is to deliver insights that help facilitate quicker, more informed decisions based on the sentiments expressed in user-generated text.

## Problem Statement
Clearly define the problem you aimed to solve or the research question you sought to answer. Include any hypotheses you formulated and the scope of your project.

The central question of my project is to determine which algorithm—Logistic Regression or Multinomial Naive Bayes—is more effective in classifying the sentiment of sentences drawn from the "Sentiment Labelled Sentences" dataset. Specifically, I aim to identify which model produces higher classification accuracy and more favorable precision, recall, and F1 scores.

I hypothesize that Logistic Regression may achieve slightly higher accuracy due to its ability to capture more nuanced feature interactions, while Naive Bayes may offer faster training times and simpler computations, given its strong probabilistic foundation and conditional independence assumption.

The scope of this project is limited to binary sentiment analysis (positive versus negative). I focus on a well-defined dataset sourced from IMDb, Amazon, and Yelp reviews, ensuring that the evaluation is both manageable and relevant. By rigorously comparing these two commonly used classification models on this benchmark dataset, I seek to gain insights that may generalize to similar sentiment analysis tasks and ultimately inform the choice of techniques in practical, real-world applications.

## Related Work
Summarize prior research or existing methods related to your project. Include citations or links to relevant papers, tools, or datasets. Discuss how your work builds upon or differs from these efforts.

A lot of research has explored sentiment analysis using both traditional machine learning and more recent deep learning approaches. For instance, Kotzias et al. (2015) introduced methods for deriving individual-level sentiment labels from group-level annotations, illustrating how deep feature representations can enhance classification performance. Their findings emphasize the importance of robust text features in improving sentiment analysis tasks.

Earlier, Ng and Jordan (2002) contrasted discriminative and generative classifiers, including Logistic Regression and Naive Bayes. They demonstrated how these two approaches differ in their treatment of data distribution assumptions and parameter estimation, setting a conceptual baseline for comparing model families.

Beyond academic research, widely used tools such as scikit-learn and publicly available datasets, like those from the UCI Machine Learning Repository, provide accessible platforms for implementing and benchmarking sentiment analysis models.

My work builds directly upon these foundations by implementing both Logistic Regression and Naive Bayes from scratch, thus enabling a clear, controlled comparison. Rather than employing deep learning or complex ensembles, I focus on two well-established, interpretable methods. By doing so, I aim to provide insights that complement the richer literature on advanced techniques and help clarify the conditions under which each foundational method excels.

link to the mentioned works:
https://dl.acm.org/doi/10.1145/2783258.2783380
https://papers.nips.cc/paper_files/paper/2001/hash/7b7a53e239400a13bd6be6c91c4f6c4e-Abstract.html

## Data Description
Describe the dataset(s) you used, including:
- **Source(s)**: Where the data came from (e.g., Kaggle, UCI ML Repository, custom dataset).
- **Size and Format**: Number of rows, features, and data types.
- **Preprocessing**: Steps taken to clean or transform the data, including handling missing values or feature engineering.

I used the "Sentiment Labelled Sentences" dataset from the UCI Machine Learning Repository. This dataset consists of 3,000 sentences drawn equally from three sources: IMDb, Amazon, and Yelp reviews. Each source contributes 1,000 sentences, with a balanced split of 500 positive and 500 negative examples. The data is stored in plain text files, each containing one sentence per line followed by a tab-separated label (0 for negative, 1 for positive).

Size and Format:

Total sentences: 3,000
Classes: Binary (positive or negative sentiment)
Features: Raw text (unstructured), which I later transformed into a vectorized format using a bag-of-words representation.
Preprocessing Steps:
I first merged the three separate text files (IMDb, Amazon, Yelp) into a single dataset. I then applied basic text normalization procedures, such as converting all sentences to lowercase and removing leading or trailing whitespace. Since the dataset contains well-defined labels and does not present missing values, I did not need extensive data cleaning. After shuffling the combined dataset, I split it into training, validation, and test sets (70% training, 15% validation, and 15% testing). Finally, I used a CountVectorizer to transform the raw text into numerical feature vectors, enabling both the Logistic Regression and Naive Bayes models to process the data effectively.

## Methodology
Outline your approach, including:
1. The algorithms or models used (e.g., linear regression, neural networks, etc.).
2. Details of the training process (e.g., train-test splits, cross-validation).
3. Any hyperparameter tuning performed.
4. Tools and libraries employed (e.g., scikit-learn, PyTorch).

I implemented and compared two classic machine learning algorithms for text classification: Logistic Regression and Multinomial Naive Bayes. Rather than relying on pre-built model classes, I wrote both implementations from scratch to gain a thorough understanding of their underlying mathematical principles and training processes.

1. Models Used:

Logistic Regression: A discriminative model that uses a sigmoid function to estimate the probability of a given sentence being positive. I incorporated parameters such as learning rate, number of iterations, and optional L2 regularization to manage overfitting.
Multinomial Naive Bayes: A generative model that applies Bayes’ theorem under the naive assumption of feature independence. By counting feature occurrences and applying Laplace smoothing, I obtained a simple yet effective probabilistic classifier for text data.
2. Training Process:
I split the combined dataset into three subsets: 70% for training, 15% for validation, and 15% for testing. After vectorizing the text with CountVectorizer, I used the training set to fit both models. The validation set helped me compare model performance and select the final model configuration. Finally, the test set provided an unbiased evaluation of the chosen model’s performance.

3. Hyperparameter Tuning:
For Logistic Regression, I experimented with different learning rates and iteration counts to ensure convergence. I also considered adding L2 regularization to improve generalization. For Naive Bayes, I tested different values of the Laplace smoothing parameter (alpha). The choices were guided by validation performance, rather than exhaustive searches, given the scope and size of the dataset.

4. Tools and Libraries:

Python: Core language for data preprocessing, model implementation, and evaluation.
NumPy: Used for array manipulation, vectorized computations, and linear algebra operations.
pandas: Utilized to load and preprocess the dataset, making it convenient to handle CSV files and perform data filtering.
scikit-learn: Employed mainly for the CountVectorizer to transform raw text into feature vectors. Although I did not use scikit-learn’s model implementations, its feature extraction tools simplified the preprocessing stage.
By combining these tools with my custom model implementations, I established a flexible and transparent pipeline for training, evaluating, and comparing Logistic Regression and Naive Bayes classifiers for sentiment analysis.

## Results
Present the results of your experiments, including:
- Key metrics (e.g., accuracy, precision, recall, F1 score, etc.).
- Comparisons between models or baselines.
- Visualizations (e.g., plots, confusion matrices).

After training and tuning both models, I evaluated their performance on the validation and test sets. I focused on four primary metrics: Accuracy, Precision, Recall, and F1 Score. The validation set results guided me in selecting the final model, while the test set results provided an unbiased assessment of the chosen approach.

Validation Set Performance:

Model	Accuracy	Precision	Recall	F1 Score
Logistic Regression	0.7961	0.7981	0.7981	0.7981
Naive Bayes	0.8107	0.8351	0.7788	0.8060
Based on the validation results, the Naive Bayes model achieved a higher F1 score than Logistic Regression, indicating that it balanced precision and recall more effectively. As a result, I selected Naive Bayes as the final model.

Test Set Performance (Final Model: Naive Bayes):

Model	Accuracy	Precision	Recall	F1 Score
Naive Bayes	0.8329	0.8528	0.8077	0.8296
On the test set, the Naive Bayes model delivered strong performance, achieving an accuracy of approximately 83.29% and an F1 score of about 0.83. These results suggest that Naive Bayes is a practical and effective choice for sentiment analysis on this dataset.

## Discussion
Interpret your results:

- What worked well?
- What challenges or limitations did you encounter?
- How do the results address your problem statement?

The results demonstrate that the Multinomial Naive Bayes model performed slightly better than Logistic Regression in terms of overall F1 score and accuracy. This outcome supports the initial hypothesis that Naive Bayes, with its straightforward probabilistic framework and conditional independence assumption, can effectively handle this particular sentiment classification task. In practice, this means that for the given dataset and feature representation, Naive Bayes may provide a faster and more reliable approach, making it well-suited for scenarios where computational efficiency and interpretability are valued.

However, there were several challenges and limitations. First, I worked with a relatively simple bag-of-words representation of text, which does not capture word order, context, or semantic nuances. More advanced feature engineering or the integration of word embeddings might further improve model performance. Additionally, while I implemented both models from scratch to gain a deeper understanding, this approach is more time-consuming and prone to implementation errors than using well-tested libraries. Nevertheless, it provided valuable insights into the mechanics of each algorithm.

In terms of addressing the problem statement, the results confirm that it is possible to streamline sentiment classification by employing automated methods. By selecting a model that offers a good balance of accuracy, precision, recall, and speed, I have demonstrated that text-based sentiment analysis can be made more efficient and less resource-intensive. Consequently, these findings are directly applicable to the real-world motivation behind this project—helping to reduce the manual workload and enabling faster, more informed decision-making in market research scenarios.

## Conclusion and Future Work

Summarize the key findings and discuss potential extensions of your work. What would you do differently with more time or resources?

In this project, I successfully implemented and compared two foundational machine learning models—Logistic Regression and Multinomial Naive Bayes—for the task of sentiment classification. The results indicate that Naive Bayes slightly outperformed Logistic Regression in terms of F1 score and accuracy, suggesting that for the given dataset and approach, a simple probabilistic model can yield effective sentiment analysis outcomes.

The key takeaway is that sentiment classification can be both efficient and reliable without resorting to overly complex models. This is particularly beneficial for resource-constrained settings, such as small businesses in market research, where automating sentiment analysis can reduce manual effort and streamline decision-making processes.

With additional time and resources, I would consider exploring more sophisticated feature representations, such as word embeddings or transformer-based encodings, to capture semantic relationships and context more accurately. I would also experiment with larger datasets, domain-specific preprocessing strategies, and model ensembles. Finally, introducing methods for explainability and interpretability could provide deeper insights into why certain predictions are made, further enhancing the practical value of the sentiment analysis pipeline.

## References

Include any citations for datasets, tools, libraries, or papers used in your project. Use a consistent citation format.

1.	Kotzias, D., Denil, M., de Freitas, N., Smyth, P. "From Group to Individual Labels Using Deep Features." Proceedings of the 21st ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (2015).
2.	Ng, A. Y., Jordan, M. I. "On Discriminative vs. Generative Classifiers: A Comparison of Logistic Regression and Naive Bayes." Advances in Neural Information Processing Systems (2002).
