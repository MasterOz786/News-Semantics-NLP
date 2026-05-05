# Tasks 4 & 5 Report: N-Gram Language Models and Machine Learning Models

**Course:** NLP Assignment 2  
**Tasks Covered:** Task 4 and Task 5  
**Project Directory:** `2/`  
**Run Basis:** This report is based on the successful execution of `tasks_4_5.py` after debugging and testing in the project environment. All quantitative results reported here come from that successful run and the generated artifacts from the same run.

---

## Executive Summary

This report presents a complete analysis of two major fake news detection experiments:

- **Task 4:** class-specific unigram, bigram, and trigram language models for `fake` and `real` news, with trigram classification using **Kneser–Ney smoothing** and perplexity.
- **Task 5:** three mandatory machine learning classifiers:
  - Multinomial Naive Bayes from scratch
  - Logistic Regression with L1, L2, and ElasticNet regularization
  - Polynomial Features + Logistic Regression after PCA reduction

### Main findings

1. **Trigram language modeling performed strongly** on held-out classification, reaching:
   - Accuracy: **0.8800**
   - Precision: **0.8834**
   - Recall: **0.8800**
   - F1: **0.8803**

2. **Naive Bayes matched the best linear classifier** on this run when tuned properly:
   - Best alpha: **0.1**
   - Accuracy: **0.8800**
   - F1: **0.8800**

3. **L2 Logistic Regression was the best balanced discriminative model** among the linear variants:
   - Accuracy: **0.8800**
   - F1: **0.8800**

4. **ElasticNet underperformed L2** in this setup, suggesting that the chosen mixture penalty did not improve generalization on this feature space.

5. **Polynomial Logistic Regression on 2D PCA features performed much worse** than the text models:
   - Test Accuracy: **0.7633**
   - F1: **0.7632**

6. The weak polynomial result is explained by the fact that **2D PCA preserved only 4.4% of the TF-IDF variance**, which means most discriminative information was discarded before classification.

Overall, the results show that **well-tuned sparse linear models and smoothed n-gram language models are both competitive baselines for fake-news detection**, while aggressive dimensionality reduction before non-linear expansion causes a substantial loss of useful signal.

---

## Screenshot Placeholders for Submission

This section provides ready-to-fill placeholders for screenshots that can be inserted into the final submission. Store all images inside `2/screenshots/` and keep the filenames as suggested below so the report remains organized.

### Screenshot 1 — Successful end-to-end terminal execution

**What to capture:**
- the terminal after running the full script successfully
- include dataset size, label distribution, model results, and the final generated-files message

**Run from:** project root `2/`

**Command:**
- `python tasks_4_5.py`

**Alternative (if using the project virtual environment directly):**
- `venv311/bin/python tasks_4_5.py`

**Suggested filename:**
- `screenshots/01_full_terminal_run.png`

![Screenshot Placeholder: Full successful terminal run](screenshots/01_full_terminal_run.png)

---

### Screenshot 2 — Task 4 n-gram output and trigram LM metrics

**What to capture:**
- the terminal portion showing:
  - unigram, bigram, and trigram top n-grams
  - trigram LM accuracy and F1

**Source:**
- terminal output produced by `python tasks_4_5.py` from the `2/` directory

**Suggested filename:**
- `screenshots/02_task4_ngram_output.png`

![Screenshot Placeholder: Task 4 n-gram output](screenshots/02_task4_ngram_output.png)

---

### Screenshot 3 — Naive Bayes alpha sensitivity analysis

**What to capture:**
- the terminal portion showing:
  - TF-IDF shape
  - alpha sensitivity analysis for all values
  - best alpha
  - misclassification summary

**Source:**
- terminal output produced by `python tasks_4_5.py` from the `2/` directory

**Suggested filename:**
- `screenshots/03_naive_bayes_results.png`

![Screenshot Placeholder: Naive Bayes sensitivity analysis](screenshots/03_naive_bayes_results.png)

---

### Screenshot 4 — Logistic Regression variants output

**What to capture:**
- the terminal portion showing:
  - L1, L2, and ElasticNet results
  - top weighted features printed for each
  - the line indicating ROC plotting

**Source:**
- terminal output produced by `python tasks_4_5.py` from the `2/` directory

**Suggested filename:**
- `screenshots/04_logistic_regression_output.png`

![Screenshot Placeholder: Logistic Regression output](screenshots/04_logistic_regression_output.png)

---

### Screenshot 5 — ROC curves figure

**What to capture:**
- the generated plot file itself

**Source file:**
- `2/roc_curves_logistic_regression.png`

**Suggested filename to keep:**
- `roc_curves_logistic_regression.png`

![Screenshot Placeholder: ROC curves](screenshots/05_roc_curves_placeholder.png)

**Note:**
You can either:
- keep the original generated file `roc_curves_logistic_regression.png` and insert it directly in the final version, or
- take a screenshot of the image viewer and store it as `screenshots/05_roc_curves_placeholder.png`.

---

### Screenshot 6 — Polynomial degree analysis from terminal

**What to capture:**
- the terminal portion showing:
  - PCA explained variance
  - degree 1, 2, 3 train/test accuracy and F1
  - full degree-2 feature-space size

**Source:**
- terminal output produced by `python tasks_4_5.py` from the `2/` directory

**Suggested filename:**
- `screenshots/06_polynomial_terminal_output.png`

![Screenshot Placeholder: Polynomial degree analysis output](screenshots/06_polynomial_terminal_output.png)

---

### Screenshot 7 — Polynomial decision boundaries figure

**What to capture:**
- the generated decision boundary visualization

**Source file:**
- `2/polynomial_decision_boundaries.png`

**Suggested filename to keep:**
- `polynomial_decision_boundaries.png`

![Screenshot Placeholder: Polynomial decision boundaries](screenshots/07_polynomial_boundaries_placeholder.png)

**Note:**
You can either:
- embed `polynomial_decision_boundaries.png` directly, or
- store a screenshot copy in `screenshots/07_polynomial_boundaries_placeholder.png`.

---

### Screenshot 8 — Generated report and artifacts in file explorer

**What to capture:**
- the project folder showing:
  - `TASKS_4_5_REPORT.md`
  - `TASKS_4_5_REPORT_REVISED.md`
  - `roc_curves_logistic_regression.png`
  - `polynomial_decision_boundaries.png`
  - `tasks_4_5.py`

**Source:**
- Finder / file explorer view of the `2/` folder

**Suggested filename:**
- `screenshots/08_generated_artifacts.png`

![Screenshot Placeholder: Generated artifacts listing](screenshots/08_generated_artifacts.png)

---

## 1. Experimental Setup

### 1.1 Data split and pipeline

The script executed successfully with the following setup:

- Total sampled dataset size: **1,500** documents
- Label distribution:
  - `fake`: **771**
  - `real`: **729**
- Train/test split:
  - Training set: **1,200** documents
  - Test set: **300** documents
- For Task 4 perplexity classification, a subset of **100 held-out samples** was used.

### 1.2 Feature representation

The following feature representations were used:

- **Task 4:** custom tokenized text for n-gram language modeling
- **Task 5.1 and 5.2:** TF-IDF with `max_features = 5000`
- **Task 5.3:** PCA reduction from full TF-IDF to **2D**, followed by polynomial feature expansion of degree 1, 2, and 3

### 1.3 Why these design choices make sense

These choices are methodologically reasonable for an assignment setting:

- **TF-IDF** is a strong baseline for text classification because it downweights extremely common words and emphasizes discriminative terms.
- **Train/test split at 80/20** is standard and provides enough training signal while preserving unbiased evaluation.
- **Class-specific language models** are useful because each class has its own stylistic and lexical regularities.
- **Perplexity-based classification** is a natural way to compare which class language model better explains a test document.

---

## 2. Task 4 — N-Gram Language Models

## 2.1 Objective

The goal of Task 4 was to build separate language models for the `fake` and `real` classes using:

- unigram models
- bigram models
- trigram models

The trigram model was smoothed using **Kneser–Ney smoothing**, implemented from scratch, and used to classify held-out documents by **perplexity**.

---

## 2.2 Why language models are useful here

A class-specific language model captures the probability structure of how text is written in a given class.

For fake news detection, this is useful because fake and real news often differ in:

- topical phrasing
- lexical preferences
- stylistic repetition
- attribution patterns
- source conventions

If a held-out article has lower perplexity under the `fake` model than the `real` model, that means the fake-news model considers that article more probable, and vice versa.

---

## 2.3 Why Kneser–Ney smoothing instead of Laplace smoothing

### Laplace smoothing: why it is weak for trigrams

Laplace, or add-one smoothing, is simple but not ideal for higher-order language modeling.

Its main problems are:

1. It adds the same pseudo-count to every unseen event.
2. It over-allocates probability mass to impossible or extremely implausible n-grams.
3. It distorts the relative probabilities of observed n-grams.
4. It is especially poor in sparse settings like trigrams, where most possible sequences are unseen.

In text classification by perplexity, that matters because the quality of the probability distribution directly affects class assignment.

### Kneser–Ney: why it is better

Kneser–Ney smoothing is widely regarded as one of the strongest classical smoothing methods for n-gram language models because it does two important things:

1. **Absolute discounting:** it subtracts a fixed amount from observed n-gram counts.
2. **Intelligent backoff:** it redistributes the reserved probability mass to lower-order models using **continuation probability**, not raw unigram frequency.

That distinction is important. A word should not be considered likely in a new context just because it is frequent overall. It should be considered likely if it appears in **many different contexts**. This is exactly the rationale behind Kneser–Ney.

### Practical rationale for this assignment

This dataset is sparse at the trigram level. Many legitimate trigrams will never appear in training. Therefore:

- maximum-likelihood estimates are too brittle,
- Laplace is too blunt,
- Kneser–Ney offers a much better compromise between memorization and generalization.

That is why Kneser–Ney is the more defensible smoothing choice for trigram classification.

---

## 2.4 Top observed n-grams by class

The successful execution printed the most frequent n-grams for each class.

### Unigrams

#### Fake
1. `the` — 9510  
2. `.` — 8981  
3. `,` — 8723  
4. `to` — 5295  
5. `of` — 4317

#### Real
1. `the` — 10155  
2. `.` — 9925  
3. `,` — 8885  
4. `to` — 5658  
5. `of` — 4489

### Interpretation

At the unigram level, both classes are dominated by high-frequency function words and punctuation. That is expected. Unigrams capture broad lexical distribution but not local context. This is why unigram models usually have weaker discriminative power than higher-order models.

---

### Bigrams

#### Fake
1. `of the` — 926  
2. `in the` — 713  
3. `. the` — 676  
4. `, and` — 667  
5. `to the` — 491

#### Real
1. `. the` — 1102  
2. `u .` — 1077  
3. `of the` — 1041  
4. `. s` — 989  
5. `s .` — 986

### Interpretation

The real-news bigrams strongly reflect wire-style reporting and country abbreviations, particularly the tokenized pattern around **U.S.** This suggests that the real-news class contains a large number of formal journalistic articles with standardized punctuation and naming conventions.

The fake-news bigrams are more generic and less source-specific at the top. This may indicate broader stylistic variation or noisier content formatting within the fake class.

---

### Trigrams

#### Fake
1. `. s .` — 195  
2. `u . s` — 193  
3. `. it s` — 122  
4. `. featured image` — 112  
5. `. twitter .` — 108

#### Real
1. `u . s` — 989  
2. `. s .` — 982  
3. `the u .` — 324  
4. `. politicsnews </s>` — 229  
5. `the united states` — 191

### Interpretation

The trigram distributions are much more informative than the unigrams and bigrams.

#### Why the real-news trigrams make sense

The real-news class contains patterns like:

- `u . s`
- `the united states`
- `, he said`
- `washington reuters -`
- `, according to`

These are characteristic of formal news reporting, attribution-heavy journalism, and standardized newswire style. They are exactly the kinds of high-context phrases that a trigram model should capture well.

#### Why the fake-news trigrams are revealing

The fake-news class contains patterns like:

- `. featured image`
- `. twitter .`
- `twitter . com`
- `pic . twitter`

These suggest social-media references, reposting structures, and content templates that are less typical of institutional reporting. That gives the trigram model useful class-specific stylistic cues.

---

## 2.5 Trigram LM classification results

Held-out classification was performed on **100** test samples using perplexity under the `fake` and `real` trigram language models.

### Results

- **Accuracy:** 0.8800
- **Precision:** 0.8834
- **Recall:** 0.8800
- **F1-score:** 0.8803

### Interpretation

This is a strong result for a classical generative language-model baseline.

Why this performance is notable:

1. The language model does **not** use explicit discriminative feature engineering like logistic regression.
2. It makes decisions only from how naturally a document fits each class distribution.
3. Trigram context appears to capture enough stylistic structure to separate the classes effectively.

This shows that fake and real news differ not only in vocabulary, but also in **local phrase structure**.

---

## 2.6 Why the trigram LM worked well

The strong trigram result can be explained by several factors:

- fake and real news appear to use different source conventions,
- real news includes strong attribution and geographic reporting patterns,
- fake news includes more template-like and socially propagated phrasing,
- Kneser–Ney smoothing allowed the model to generalize to unseen trigrams without collapsing probability estimates.

In other words, the trigram LM succeeded because it modeled **style plus phrase context**, not just individual words.

---

## 3. Task 5.1 — Multinomial Naive Bayes from Scratch

## 3.1 Objective

The objective of Task 5.1 was to implement Multinomial Naive Bayes from scratch with:

- configurable Laplace smoothing
- log-space computation
- probability output
- support for sparse input
- alpha sensitivity analysis over:
  - 0.01
  - 0.1
  - 0.5
  - 1.0
  - 2.0
  - 5.0

---

## 3.2 Why Multinomial Naive Bayes is appropriate for text

Multinomial Naive Bayes is a classic baseline for text classification because:

1. documents are naturally represented as token counts or weighted term vectors,
2. the model is fast to train,
3. it works surprisingly well even when the independence assumption is false,
4. it performs especially well when high-dimensional sparse features are used.

It remains a standard benchmark in text categorization because it is simple, interpretable, and usually hard to beat by a large margin without more sophisticated modeling.

---

## 3.3 Why log-space computation is necessary

Naive Bayes multiplies many small probabilities together:

- one for the class prior
- many for term likelihoods

In long documents, direct multiplication quickly causes numerical underflow. Using log-space turns products into sums, which is both stable and standard practice.

This is not just an implementation detail. It is essential for correctness.

---

## 3.4 Alpha sensitivity analysis

The following results were obtained on the 300-sample test set.

| Alpha | Accuracy | F1 |
|------:|---------:|---:|
| 0.01 | 0.8767 | 0.8767 |
| 0.1  | 0.8800 | 0.8800 |
| 0.5  | 0.8733 | 0.8731 |
| 1.0  | 0.8700 | 0.8698 |
| 2.0  | 0.8600 | 0.8596 |
| 5.0  | 0.8667 | 0.8661 |

### Best configuration

- **Best alpha:** `0.1`
- **Accuracy:** `0.8800`
- **Precision:** `0.8800`
- **Recall:** `0.8800`
- **F1:** `0.8800`

### Interpretation

The best performance came from **light smoothing**, not heavy smoothing.

#### Why alpha = 0.1 worked best

A small amount of smoothing helps because it:

- prevents zero probabilities,
- still respects the observed class-specific term distributions,
- avoids over-flattening the model.

#### Why larger alphas performed worse

As alpha increases, the estimated term probabilities become more uniform. That hurts performance because the classifier stops trusting discriminative term differences strongly enough.

This trend is visible here:

- alpha `0.1` is best,
- alpha `1.0` is worse,
- alpha `2.0` is worse still.

That pattern strongly suggests that this dataset contains meaningful term-frequency differences that should not be oversmoothed.

---

## 3.5 Misclassification analysis

### Observed results

- Total test errors: **36 / 300**
- Misclassification rate: **0.1200**
- In the sampled error analysis summary from the successful run:
  - False positives: **16**
  - False negatives: **14**

### Interpretation

The error rate is fairly balanced across both classes, which is a positive sign because it suggests the model is not failing catastrophically on only one label.

### Likely reasons for Naive Bayes errors

The run output and the generated summary suggest three plausible error categories:

1. **Lexically ambiguous documents**  
   Some articles likely share vocabulary with both fake and real news. For example, political terms, health-related language, or emotionally loaded words can occur in both classes.

2. **Source-style overlap**  
   Real news discussing controversial topics can use the same nouns, names, and phrases that frequently appear in fake news, causing confusion.

3. **Broken independence assumption**  
   Naive Bayes assumes conditional independence between features given the class, but TF-IDF features in real text are highly correlated. Co-occurring terms like `reuters`, `washington`, `said`, and `according` jointly signal style and source. Naive Bayes treats them too independently.

### Rationale

This helps explain why Naive Bayes performed well but did not clearly dominate the better regularized linear model. It is strong as a count-based probabilistic baseline, but it cannot represent correlation structure directly.

---

## 4. Task 5.2 — Logistic Regression with L1, L2, and ElasticNet

## 4.1 Objective

This task trained Logistic Regression using TF-IDF features under three regularization settings:

- **L1**
- **L2**
- **ElasticNet**

with:

- `C = 1.0`
- sparse TF-IDF representation
- ROC curve generation for all variants

---

## 4.2 Why Logistic Regression is a strong text classifier

Logistic Regression is one of the strongest traditional baselines for sparse text classification because:

1. it directly models the conditional class probability,
2. it handles high-dimensional sparse vectors efficiently,
3. it benefits greatly from TF-IDF features,
4. it does not assume conditional independence,
5. regularization improves generalization and stabilizes coefficient estimates.

In many text problems, a well-regularized linear classifier is difficult to outperform without substantially more complex models.

---

## 4.3 Why LR handles correlated features better than Naive Bayes

This is one of the most important conceptual comparisons in the assignment.

### Naive Bayes

Naive Bayes assumes that features are conditionally independent given the class. In text, this is almost never true.

For example, these words are correlated:

- `reuters`, `washington`, `said`
- `twitter`, `video`, `featured`
- `covid`, `19`, `health`

When Naive Bayes sees correlated features, it effectively counts overlapping evidence multiple times.

### Logistic Regression

Logistic Regression does not make that independence assumption. Instead, it learns weights jointly to optimize class separation.

This matters because when two features are correlated:

- Naive Bayes tends to overstate confidence,
- Logistic Regression can distribute weight across them more appropriately,
- regularization discourages unstable or excessively large coefficients.

### Why L2 is especially suitable

L2 regularization is often the best fit for TF-IDF because:

- it shrinks all coefficients smoothly,
- it handles multicollinearity better than L1,
- it preserves distributed evidence across related features,
- it produces more stable decision boundaries.

That theoretical expectation is consistent with the observed results in this run.

---

## 4.4 Performance results

| Model | Accuracy | Precision | Recall | F1 |
|------|---------:|----------:|-------:|---:|
| LR (L1) | 0.8700 | 0.8770 | 0.8700 | 0.8691 |
| LR (L2) | 0.8800 | 0.8800 | 0.8800 | 0.8800 |
| LR (ElasticNet) | 0.8567 | 0.8577 | 0.8567 | 0.8564 |

### Interpretation

#### L2 was best overall

L2 tied the best overall accuracy and F1 in the entire run.

Why that makes sense:

- the feature space is sparse and high-dimensional,
- many predictive terms are correlated rather than independent,
- L2 preserves dense distributed evidence instead of zeroing too many terms.

#### L1 was slightly worse

L1 performed reasonably well but underperformed L2.

Why:

- L1 promotes sparsity,
- that is useful when many features are truly irrelevant,
- but in text classification, many weakly useful terms together can be important.

So L1 may have removed too much weak but collectively meaningful evidence.

#### ElasticNet was worst on this run

ElasticNet is often a compromise between L1 and L2, but here it underperformed both.

Likely reasons:

- the chosen `l1_ratio` may not have been optimal,
- the problem may simply favor smoother dense shrinkage over mixed sparsity,
- the current TF-IDF representation may already be sufficiently regularized by feature weighting and class balancing.

---

## 4.5 Top weighted features and what they mean

The successful run displayed the following top-weighted L2 features:

- `said`
- `reuters`
- `u`
- `video`
- `t`

The generated report from the same run also listed additional salient L2-weight features such as:

- `covid`
- `19`
- `on`
- `politicsnews`
- `news`
- `obama`
- `hillary`
- `trump`
- `worldnews`

### Interpretation

Even though the raw coefficient list in the generated script output is not perfectly organized by positive vs negative class description, the feature set itself is highly informative.

#### Features like `said`, `reuters`, `u`, `19`

These are strong indicators of:

- formal reporting style,
- wire-service structure,
- geographic attribution,
- topic-specific news coverage.

They are highly plausible markers of the **real-news** class.

#### Features like `video`, `news`, `obama`, `hillary`, `trump`

These suggest:

- politically framed articles,
- headline-style or content-aggregation behavior,
- template terms commonly seen in noisy or sensational sources,
- socially circulated and commentary-heavy text.

These are plausible markers of the **fake-news** side or at least of more weakly formalized content.

### Why this matters

A good text classifier should not only perform well but also expose interpretable signals. These feature weights show that the model is learning recognizable class-specific lexical patterns rather than arbitrary noise.

---

## 4.6 ROC curves and AUC rationale

The script generated a combined ROC figure for all three Logistic Regression variants:

- `roc_curves_logistic_regression.png`

### Why ROC curves matter

ROC curves show the trade-off between:

- **true positive rate**
- **false positive rate**

across classification thresholds.

This is useful because accuracy alone depends on a fixed threshold, while ROC analysis shows ranking quality across all thresholds.

### Interpretation for this experiment

The terminal log did not print exact AUC values, so this report does **not invent them**. However, based on the classification metrics:

- the **L2** curve is expected to be among the strongest,
- **L1** should be slightly below L2,
- **ElasticNet** should trail both.

This is consistent with the observed F1 and accuracy ordering.

---

## 5. Task 5.3 — Polynomial Features + Logistic Regression

## 5.1 Objective

The goal of this task was to:

1. reduce TF-IDF to **2 dimensions** using PCA,
2. apply polynomial expansions of degree **1, 2, and 3**,
3. train Logistic Regression in the transformed space,
4. visualize decision boundaries,
5. report train accuracy, test accuracy, and F1.

---

## 5.2 Results

### PCA compression

- Original TF-IDF dimension: **5000**
- Reduced dimension: **2**
- Explained variance: **4.4%**

### Degree-wise results

| Degree | Feature Count | Train Accuracy | Test Accuracy | F1 |
|------:|--------------:|---------------:|--------------:|---:|
| 1 | 2 | 0.7250 | 0.7633 | 0.7632 |
| 2 | 5 | 0.7292 | 0.7633 | 0.7632 |
| 3 | 9 | 0.7292 | 0.7633 | 0.7632 |

### Full-space degree-2 size

For degree-2 polynomial expansion on the original 5000-dimensional TF-IDF space:

- feature count ≈ **12,502,500**

---

## 5.3 Interpretation

This result is actually very instructive.

### Why performance dropped so much

The polynomial classifier is much worse than NB, LR, and the trigram LM. The main reason is not that polynomial features are inherently poor. The main reason is that the model was trained only after TF-IDF had been compressed to **2 PCA dimensions**.

But PCA preserved only **4.4%** of the total variance.

That means:

- over **95%** of the variation in the text representation was discarded,
- most discriminative lexical information was lost,
- polynomial terms could only interact over a highly impoverished 2D summary.

So the low performance is expected.

### Why degrees 2 and 3 did not help

Polynomial expansion only helps if the lower-dimensional representation still contains useful class structure.

Here it likely did not.

As a result:

- degree 2 added more complexity but no new useful signal,
- degree 3 added even more flexibility but still could not recover lost lexical information,
- all three models converged to almost identical performance.

### Important rationale

This result is not a failure of the method alone. It is a demonstration of a classic trade-off:

- **interpretability and visualization** improve in 2D,
- **classification fidelity** decreases when high-dimensional text is compressed too aggressively.

That is exactly the kind of insight this task is designed to reveal.

---

## 5.4 Why the 12.5 million feature count matters

A full degree-2 expansion over 5000 TF-IDF features would create approximately **12.5 million** features.

That is important because it explains why PCA reduction was used in the first place.

### Why full polynomial expansion is problematic

- memory cost becomes very high,
- training slows dramatically,
- overfitting risk increases,
- interpretability becomes poor,
- sparse text data becomes extremely high-dimensional.

So the dimensionality calculation is not just arithmetic. It provides the core rationale for why a visual 2D approximation was used instead of full polynomial expansion.

---

## 5.5 Alternative non-linear approach

A better non-linear alternative would be a **kernel-based classifier**, such as:

- kernel logistic regression, or
- SVM with RBF kernel

### Why this is better conceptually

A kernel method can model non-linear decision boundaries **implicitly** without explicitly constructing millions of interaction features.

This avoids:

- combinatorial feature explosion,
- manual polynomial engineering at large scale,
- unnecessary memory costs.

For course rationale, this is a more principled non-linear alternative than brute-force polynomial expansion of full TF-IDF.

---

## 6. Cross-Model Comparison

| Method | Accuracy | Precision | Recall | F1 |
|------|---------:|----------:|-------:|---:|
| Trigram LM (Kneser–Ney) | 0.8800 | 0.8834 | 0.8800 | 0.8803 |
| Naive Bayes (alpha = 0.1) | 0.8800 | 0.8800 | 0.8800 | 0.8800 |
| Logistic Regression (L1) | 0.8700 | 0.8770 | 0.8700 | 0.8691 |
| Logistic Regression (L2) | 0.8800 | 0.8800 | 0.8800 | 0.8800 |
| Logistic Regression (ElasticNet) | 0.8567 | 0.8577 | 0.8567 | 0.8564 |
| Polynomial LR (degree 2, PCA-2D) | 0.7633 | — | — | 0.7632 |

### Key conclusions from the comparison

1. **Three methods clustered at the top:**
   - Trigram LM
   - Naive Bayes
   - L2 Logistic Regression

2. **The best F1 came from the trigram language model** by a very small margin.

3. **Naive Bayes remained highly competitive**, showing that simple probabilistic baselines are still strong for text.

4. **L2 Logistic Regression matched Naive Bayes**, which supports the theory that regularized discriminative linear models are excellent choices for TF-IDF classification.

5. **ElasticNet did not improve over L2**, meaning the mixed penalty was unnecessary in this setup.

6. **Polynomial LR lagged badly**, primarily because the PCA reduction removed most of the text signal.

---

## 7. Final Discussion

## 7.1 What these results say about fake news detection

This experiment suggests that fake news detection can be modeled successfully in at least two different ways:

- **generative sequence modeling** through trigram language models,
- **discriminative lexical classification** through TF-IDF + Logistic Regression / Naive Bayes.

That is important because it shows the problem is not only about which words occur, but also about:

- how phrases are formed,
- how information is attributed,
- how source conventions differ,
- how strongly local context contributes to class identity.

---

## 7.2 Why simple models remain strong baselines

A major takeaway is that classical methods are still powerful:

- Multinomial Naive Bayes is fast, interpretable, and strong.
- Logistic Regression with L2 is stable and robust.
- Trigram LM with Kneser–Ney captures phrase-level style effectively.

For an assignment or production baseline, these models are excellent choices because they provide:

- fast training,
- strong interpretability,
- reproducible behavior,
- clear probabilistic reasoning.

---

## 7.3 Recommended model choice

If one model had to be recommended from this run, the best practical choice is:

### **Logistic Regression with L2 regularization**

Why:

- tied-best discriminative performance,
- stable under correlated sparse features,
- interpretable coefficients,
- standard and reliable optimization behavior,
- easy deployment with TF-IDF features.

### Secondary recommendation

The **trigram LM with Kneser–Ney** is an excellent complementary model because it captures sequential phrase structure that TF-IDF does not.

An ensemble of:

- L2 Logistic Regression
- Naive Bayes
- Trigram LM

could plausibly perform even better by combining lexical, probabilistic, and sequential evidence.

---

## 8. Limitations and honesty notes

This report is based strictly on the successful run output and generated artifacts from that run.

Accordingly:

- Exact **AUC values** were not printed to terminal, so they are not fabricated here.
- Exact **unique n-gram counts** were not exposed in the tested console output, so they are not invented here.
- The assignment wording asks for top **most probable** n-grams, but the current implementation surfaces top **observed-count** n-grams in the output.
- The “manual examination” of 30 misclassified samples is summarized at the aggregate level rather than documented sample-by-sample.

These limitations do not invalidate the results, but they should be acknowledged explicitly for academic correctness.

---

## 9. Generated Artifacts

The successful run produced the following files:

1. `TASKS_4_5_REPORT.md`
2. `roc_curves_logistic_regression.png`
3. `polynomial_decision_boundaries.png`
4. `tasks_4_5.py`

This revised report has been written as:

5. `TASKS_4_5_REPORT_REVISED.md`

---

## 10. References

1. **Kneser, R., & Ney, H.** (1995). *Improved backing-off for M-gram language modeling*. In **Proceedings of the 1995 IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP)**, 181–184. IEEE. https://doi.org/10.1109/ICASSP.1995.479394

2. **Chen, S. F., & Goodman, J.** (1999). *An empirical study of smoothing techniques for language modeling*. **Computer Speech & Language, 13**(4), 359–394. https://doi.org/10.1006/csla.1999.0128

3. **Jurafsky, D., & Martin, J. H.** (2026 draft). *Speech and Language Processing* — chapter on Kneser–Ney smoothing. Stanford University draft. https://web.stanford.edu/~jurafsky/slp3/

4. **Manning, C. D., & Schütze, H.** (1999). *Foundations of Statistical Natural Language Processing*. MIT Press.

5. **McCallum, A., & Nigam, K.** (1998). *A comparison of event models for Naive Bayes text classification*. In **AAAI-98 Workshop on Learning for Text Categorization**, 41–48.

6. **Hastie, T., Tibshirani, R., & Friedman, J.** (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer. https://doi.org/10.1007/978-0-387-84858-7

7. **Fawcett, T.** (2006). *An introduction to ROC analysis*. **Pattern Recognition Letters, 27**(8), 861–874. https://doi.org/10.1016/j.patrec.2005.10.010

8. **Sparck Jones, K.** (1972). *A statistical interpretation of term specificity and its application in retrieval*. **Journal of Documentation, 28**(1), 11–21. https://doi.org/10.1108/eb026526

9. **Robertson, S.** (2004). *Understanding inverse document frequency: On theoretical arguments for IDF*. **Journal of Documentation, 60**(5), 503–520.

10. **scikit-learn developers.** *LogisticRegression documentation*. Scikit-learn stable documentation. https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression

11. **scikit-learn developers.** *roc_auc_score documentation*. Scikit-learn stable documentation. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html

---

## 11. Final Conclusion

The tested implementation demonstrates that fake-news detection can be addressed effectively with both **smoothed language models** and **regularized linear classifiers**.

The most important outcome is that performance is not driven by a single modeling philosophy:

- **Kneser–Ney trigram language modeling** captures phrase-level stylistic and source patterns.
- **Naive Bayes** provides a strong probabilistic lexical baseline.
- **L2 Logistic Regression** offers the best balance of interpretability, robustness, and accuracy for TF-IDF features.
- **Polynomial expansion after 2D PCA** is useful pedagogically for visualization, but not competitive for real classification in this setup.

If this work were extended, the strongest next steps would be:

1. report exact unique n-gram counts and probability-ranked n-grams,
2. add exact ROC-AUC numbers to the report,
3. document the 30-error manual analysis explicitly,
4. test ensemble methods combining LM and LR signals,
5. evaluate stronger non-linear methods without collapsing TF-IDF to only 2 dimensions.

In summary, the results strongly support **L2 Logistic Regression** as the best practical baseline and **Kneser–Ney trigram modeling** as the most interesting classical generative alternative.
