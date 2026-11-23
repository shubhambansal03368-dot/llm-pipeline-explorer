# LLM Pipeline Explorer

An interactive educational simulation designed to help students, developers, and machine learning enthusiasts understand how Large Language Models transform human language into numerical representations through vocabulary creation, tokenization, token IDs, word embeddings, and 3D vector-space geometry.

## 🚀 Features

- **01 — Build Your Vocabulary:**
  - Multiline natural language input with live statistics (word count, token occurrences, unique tokens, characters).
  - Deterministic 0-indexed vocabulary generator preserving first-occurrence order.
  - Three-column vocabulary table with instant search and column sorting (ID, Token, Count).
  - Stage 3 Token ID sequence viewer displaying tensor array input stream and interactive token-ID badges.
  - Educational comparison between simplified word-level tokenization and production subword Byte-Pair Encoding (BPE).

- **02 — Word Embedding Explorer:**
  - **Spelling-Based Embeddings:** 16-dimensional morphological feature vectors (character n-grams, length, vowel ratio, prefix/suffix hash).
  - **Meaning-Based Embeddings:** 16-dimensional continuous semantic feature vectors (calibrated conceptual categories and analogical properties).
  - Dense numerical vectors table with micro-heatmaps showing component activations.

- **03 — Interactive 3D Vector Space (Three.js):**
  - Continuous 3D vector-space manifold computed via Principal Component Analysis (PCA).
  - Smooth coordinate interpolation (lerp animation) when toggling between Spelling and Meaning modes.
  - OrbitControls (rotate, zoom, pan), selection halo, dashed connecting lines to Top-3 nearest neighbors.
  - 2D screen-projected typography label overlays and hover inspection tooltips.
  - Integrated 2D Canvas fallback if WebGL / Three.js CDN is unavailable.

- **Word Details Panel & Pairwise Comparison:**
  - 3D PCA coordinates $(X, Y, Z)$ and dense 16-D vector bar chart.
  - Top 5 Nearest Neighbors sorted by Cosine Similarity.
  - Pairwise vector comparison tool calculating Cosine Similarity, Dot Product, and Euclidean Distance.

- **04 — "What Just Happened?" Explanation Panel:**
  - Educational breakdown of spelling morphology vs semantic directional alignment.
  - Canonical examples (King & Queen, Car & Vehicle, Apple & Banana).
  - Interactive vector analogy playground ($\vec{\text{king}} - \vec{\text{man}} + \vec{\text{woman}} \approx \vec{\text{queen}}$).
  - 6 Core Pillars of LLM representation.

## 🛠️ Technology Stack

- **HTML5:** Semantic document structure
- **CSS3:** Custom minimal light-theme styling (No frameworks)
- **Vanilla JavaScript (ES6 Modules):** All application logic, tokenization, PCA, and vector calculations
- **Three.js (CDN):** Interactive 3D vector space visualization

## 💻 Running Locally

Simply open `index.html` in any modern web browser, or serve it via a local static server:

```bash
# Using Python
python3 -m http.server 8080

# Then open in browser:
http://localhost:8080/
```
