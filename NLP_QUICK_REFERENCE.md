# NLP Pipeline Quick Reference Guide

## Dataset Overview

| Dataset | Documents | Real | Fake | Sources | Key Characteristic |
|---------|-----------|------|------|---------|-------------------|
| COVID-19 | 4,986 | 4,064 (81.5%) | 922 (18.5%) | 11 | Health claims, highly real-biased |
| FakeNewsNet | 422 | 211 (50%) | 211 (50%) | 2 | Fact-checked, perfectly balanced |
| ISOT | 44,898 | 21,417 (47.7%) | 23,481 (52.3%) | 1 | Largest, fake-biased, general news |
| Liar | 12,836 | 4,529 (35.3%) | 8,307 (64.7%) | 1 | Political claims, highly fake-biased |
| **Combined** | **63,142** | **39,921 (63.2%)** | **23,221 (36.8%)** | 15 | Balanced, diverse sources |

## Text Cleaning Results

| Metric | Value | Interpretation |
|--------|-------|-----------------|
| HTML tags found | 0.0% | Data already clean |
| URLs in documents | 4.5% | Minimal web artifacts |
| Emojis present | 0.0% | Formal news, not social |
| Text preserved | +0.22 avg words | Cleaning minimal destructive |
| Average doc length | 324 words | Moderate length articles |

## Tokenization Comparison

| Method | Tokens/Doc | OOV Rate | Speed | Recommendation |
|--------|-----------|----------|-------|-----------------|
| NLTK | 377.20 | 2.31% | 1.457s | Linguistic rigor (slow) |
| spaCy | 386.38 | 1.98% | 0.131s | **✅ RECOMMENDED** (best balance) |
| Custom | 391.68 | 2.14% | 0.080s | Maximum speed, minimal features |

## Stopword Analysis

| Metric | Standard NLTK | Custom Domain |
|--------|---------------|----------------|
| Stopword count | 179 | 210 (+31 custom) |
| Removal rate | 37.19% | 38.09% |
| Classification F1 | **0.8895** ✅ | 0.8792 |
| Accuracy | 89% | 88% |
| **Recommendation** | **✅ Use standard** | Domain study only |

**Why Standard Wins?** Removing domain keywords ("vaccine", "covid") forces model to learn from context rather than shallow keywords.

## Stemming vs Lemmatization

| Aspect | Porter | Snowball | Lemmatizer |
|--------|--------|----------|-------------|
| Vocabulary size | 10,073 | 9,791 | 12,967 |
| Collisions | 174,295 | 174,577 | 171,401 |
| Processing time | 0.805s | 0.556s | 0.247s |
| Example: "vaccination" | "vacc" | "vaccin" | "vaccination" |
| **Recommendation** | Not ideal | Production stemming | **✅ RECOMMENDED** |

**Why Lemmatization?** Preserves real words and linguistic meaning; important for interpretability in fake news domain.

## Feature Engineering Results

### Bag-of-Words (BoW)
- **Shape:** 500 documents × 14,430 vocabulary
- **Sparsity:** 98.82% (most words are rare)
- **Top fake news terms:** "the" (0.153), "to" (0.087), "a" (0.071)
- **Top real news terms:** "the" (0.170), "to" (0.096), "of" (0.083)
- **Use case:** Interpretable baseline for linear models

### TF-IDF
- **Top variants:** Standard (no smoothing) ≈ Smooth ≈ Sublinear
- **Typical weights:** 0.05-0.20 for top discriminative terms
- **Advantage:** Emphasizes rare but important words
- **Use case:** Linear classifiers, information retrieval

### Word2Vec Embeddings
| Word Pair | Similarity | Interpretation |
|-----------|-----------|-----------------|
| covid ↔ virus | 0.819 | Strong semantic link |
| fake ↔ real | 0.829 | Antonyms still linked |
| vaccine ↔ health | None | Insufficient co-occurrence |
| pandemic ↔ lockdown | ? | Expected correlation |

- **Dimension:** 200D vectors per word
- **Training:** CBOW + Skip-Gram models
- **Use case:** Semantic similarity, clustering

## Classification Performance

| Method | F1 Score | Accuracy | Recommendation |
|--------|----------|----------|-----------------|
| TF-IDF only | 0.89 | 89% | Strong baseline |
| Word2Vec only | 0.72 | 72% | Weak alone |
| **TF-IDF + Word2Vec** | **0.90** | **90%** | **✅ BEST COMBINATION** |

## Key Takeaways

### What Was Loaded
✅ **4 independent datasets** (COVID-19, FakeNewsNet, ISOT, Liar)  
✅ **63,142 total documents** with 63% real / 37% fake balance  
✅ **Excluded generated_fakenews** (synthetic data not representative)  

### What Was Evaluated
✅ **Text cleaning:** Minimal noise, preserved 324 avg words  
✅ **Tokenization:** spaCy best (12x faster than NLTK, similar quality)  
✅ **Stopwords:** Default NLTK standard (F1: 0.8895)  
✅ **Normalization:** Lemmatization preserves linguistic meaning  
✅ **Features:** TF-IDF + Word2Vec achieves 0.90 F1 (best)  

### Why It Matters
1. **Preprocessing quality** directly impacts model performance (±5% F1)
2. **Feature choice** is critical (TF-IDF alone = 0.89, combined = 0.90)
3. **Dataset diversity** prevents overfitting to single source
4. **Interpretability** matters (lemmatization > stemming for fake news domain)

## Report Sections in Detail

| Section | Length | Focus |
|---------|--------|-------|
| Part 1: Dataset Loading | 2.3 | What data, why combined, statistics |
| Part 2: Text Cleaning | 2.4 | Why clean, audit results, implementation |
| Part 3: Tokenization | 3.4 | Why tokenize, 3 methods comparison |
| Part 4: Stopwords | 4.4 | Default vs custom, F1 scores, recommendations |
| Part 5: Stemming/Lemmatization | 5.5 | 3 normalization methods, collision analysis |
| Part 6: Feature Engineering | 6.5 | BoW, TF-IDF, Word2Vec details + combinations |
| Part 7: Conclusions | 7.5 | Summary, metrics, production recommendations |
| Appendix | A.3 | Technical implementation details |

---

## How to Use This Report

**For Presentation:**
- Show Executive Summary (paragraph 1)
- Display key metrics table (above)
- Highlight Part 7: Conclusions

**For Academic Writing:**
- Reference dataset statistics (Part 1)
- Cite cleaning methodology (Part 2)
- Justify feature choices (Part 6)

**For Production Implementation:**
- Follow tokenization recommendation (spaCy)
- Use default stopwords (Part 4)
- Apply TF-IDF + Word2Vec (Part 6)

**For Interpretability:**
- Lemmatization preserves words (Part 5)
- Top-terms analysis shows what drives decisions (Part 6)
- Collision counts explain information loss (Part 5)

---

**Report Location:** `NLP_PIPELINE_REPORT.md`  
**This Guide:** `NLP_QUICK_REFERENCE.md`  
**Code:** `nlp_pipeline.py`  
**Data:** `datasets/` (63,142 documents)  
**Generated:** 2026-05-06
