/**
 * LLM Pipeline Explorer - Embeddings Module
 * Pure Vanilla JavaScript (ES6 Module)
 *
 * Implements:
 * 1. Vector mathematics (Dot product, Norm, Cosine Similarity, Euclidean Distance, Vector Arithmetic)
 * 2. Principal Component Analysis (PCA) for 3D coordinate projection
 * 3. Option A: Deterministic 16-D Spelling-Based (Morphological) Embeddings
 * 4. Option B: Calibrated 16-D Meaning-Based (Semantic) Embeddings with Word2Vec-style analogies
 */

// -------------------------------------------------------------
// 1. Vector Mathematical Utilities
// -------------------------------------------------------------

export function dotProduct(a, b) {
  let sum = 0;
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

export function vectorNorm(a) {
  let sumSq = 0;
  for (let i = 0; i < a.length; i++) {
    sumSq += a[i] * a[i];
  }
  return Math.sqrt(sumSq);
}

export function normalizeVector(v) {
  const norm = vectorNorm(v);
  if (norm === 0) return [...v];
  return v.map((val) => val / norm);
}

export function cosineSimilarity(a, b) {
  const normA = vectorNorm(a);
  const normB = vectorNorm(b);
  if (normA === 0 || normB === 0) return 0;
  const dot = dotProduct(a, b);
  const cos = dot / (normA * normB);
  return Math.max(-1, Math.min(1, cos));
}

export function euclideanDistance(a, b) {
  let sumSq = 0;
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) {
    const diff = a[i] - b[i];
    sumSq += diff * diff;
  }
  return Math.sqrt(sumSq);
}

export function vectorArithmetic(vecA, vecB, vecC) {
  const len = Math.min(vecA.length, vecB.length, vecC.length);
  const result = new Array(len);
  for (let i = 0; i < len; i++) {
    result[i] = vecA[i] - vecB[i] + vecC[i];
  }
  return normalizeVector(result);
}

// -------------------------------------------------------------
// 2. Principal Component Analysis (PCA) for 3D Projection
// -------------------------------------------------------------

export function projectTo3DWithPCA(vectors, targetRadius = 7.5) {
  const n = vectors.length;
  if (n === 0) return [];
  if (n === 1) return [[0, 0, 0]];
  if (n === 2) {
    return [
      [-targetRadius * 0.6, 0, 0],
      [targetRadius * 0.6, 0, 0],
    ];
  }

  const d = vectors[0].length;

  // 1. Calculate Mean
  const mean = new Array(d).fill(0);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < d; j++) {
      mean[j] += vectors[i][j];
    }
  }
  for (let j = 0; j < d; j++) {
    mean[j] /= n;
  }

  // 2. Center Data
  const centered = vectors.map((v) => v.map((val, j) => val - mean[j]));

  // 3. Covariance Matrix (D x D)
  const cov = Array.from({ length: d }, () => new Array(d).fill(0));
  for (let j1 = 0; j1 < d; j1++) {
    for (let j2 = j1; j2 < d; j2++) {
      let sum = 0;
      for (let i = 0; i < n; i++) {
        sum += centered[i][j1] * centered[i][j2];
      }
      const val = sum / Math.max(1, n - 1);
      cov[j1][j2] = val;
      cov[j2][j1] = val;
    }
  }

  // 4. Power Iteration for top 3 eigenvectors
  const eigenvectors = [];
  const numComponents = Math.min(3, d);

  for (let comp = 0; comp < numComponents; comp++) {
    let v = new Array(d).fill(0).map((_, idx) => Math.sin(comp + 1 + idx * 0.5));
    let norm = vectorNorm(v);
    if (norm > 0) v = v.map((x) => x / norm);

    for (let iter = 0; iter < 35; iter++) {
      const w = new Array(d).fill(0);
      for (let r = 0; r < d; r++) {
        let rowSum = 0;
        for (let c = 0; c < d; c++) {
          rowSum += cov[r][c] * v[c];
        }
        w[r] = rowSum;
      }

      // Gram-Schmidt orthogonalization
      for (const prevEigen of eigenvectors) {
        const proj = dotProduct(w, prevEigen);
        for (let j = 0; j < d; j++) {
          w[j] -= proj * prevEigen[j];
        }
      }

      const wNorm = vectorNorm(w);
      if (wNorm < 1e-10) break;
      v = w.map((x) => x / wNorm);
    }

    eigenvectors.push(v);
  }

  while (eigenvectors.length < 3) {
    const dummy = new Array(d).fill(0);
    dummy[eigenvectors.length % d] = 1;
    eigenvectors.push(dummy);
  }

  // 5. Project centered points onto top 3 eigenvectors
  const rawCoords = centered.map((row) => [
    dotProduct(row, eigenvectors[0]),
    dotProduct(row, eigenvectors[1]),
    dotProduct(row, eigenvectors[2]),
  ]);

  // 6. Scale neatly to 3D bounding sphere
  let maxAbs = 0;
  for (const [x, y, z] of rawCoords) {
    maxAbs = Math.max(maxAbs, Math.abs(x), Math.abs(y), Math.abs(z));
  }
  const scale = maxAbs > 1e-6 ? targetRadius / maxAbs : 1.0;

  return rawCoords.map(([x, y, z], idx) => {
    const jitterX = Math.sin(idx * 7.13) * 0.05;
    const jitterY = Math.cos(idx * 5.27) * 0.05;
    const jitterZ = Math.sin(idx * 3.41) * 0.05;

    return [
      Number((x * scale + jitterX).toFixed(3)),
      Number((y * scale + jitterY).toFixed(3)),
      Number((z * scale + jitterZ).toFixed(3)),
    ];
  });
}

// -------------------------------------------------------------
// 3. Option A: Spelling-Based Embeddings
// -------------------------------------------------------------

const COMMON_PREFIXES = [
  'un', 're', 'in', 'im', 'dis', 'pre', 'pro', 'auto', 'arti', 'tech',
  'trans', 'neu', 'mod', 'alg', 'lan', 'sub', 'inter', 'over', 'co'
];

const COMMON_SUFFIXES = [
  'ing', 'tion', 'sion', 'ment', 'ness', 'able', 'ible', 'ed', 'er',
  'est', 'al', 'ic', 'ly', 's', 'es', 'ize', 'ise', 'ity', 'ive'
];

export function generateSpellingVector(word) {
  const clean = word.toLowerCase().trim();
  const len = clean.length;
  if (len === 0) {
    return {
      vector: new Array(16).fill(0),
      featuresSummary: { length: 0 },
      category: 'Spelling (Empty)',
    };
  }

  const normLength = Math.min(len / 14, 1.0);

  const charAtNorm = (idx) => {
    if (idx >= len || idx < 0) return 0;
    const code = clean.charCodeAt(idx);
    if (code >= 97 && code <= 122) return (code - 97) / 25;
    if (code >= 48 && code <= 57) return (code - 48) / 9;
    return 0.5;
  };

  const firstChar = charAtNorm(0);
  const secondChar = len > 1 ? charAtNorm(1) : firstChar;
  const penultChar = len > 2 ? charAtNorm(len - 2) : secondChar;
  const lastChar = charAtNorm(len - 1);

  const vowels = (clean.match(/[aeiou]/g) || []).length;
  const vowelRatio = vowels / len;
  const consonantRatio = Math.max(0, 1 - vowelRatio);

  let prefixMatch = '';
  let prefixScore = 0.0;
  for (let i = 0; i < COMMON_PREFIXES.length; i++) {
    if (clean.startsWith(COMMON_PREFIXES[i])) {
      prefixMatch = COMMON_PREFIXES[i];
      prefixScore = (i + 1) / COMMON_PREFIXES.length;
      break;
    }
  }

  let suffixMatch = '';
  let suffixScore = 0.0;
  for (let i = 0; i < COMMON_SUFFIXES.length; i++) {
    if (clean.endsWith(COMMON_SUFFIXES[i])) {
      suffixMatch = COMMON_SUFFIXES[i];
      suffixScore = (i + 1) / COMMON_SUFFIXES.length;
      break;
    }
  }

  const bigramBuckets = [0, 0, 0];
  for (let i = 0; i < len - 1; i++) {
    const bigramHash = (clean.charCodeAt(i) * 31 + clean.charCodeAt(i + 1)) % 3;
    bigramBuckets[bigramHash] += 1;
  }
  const bigramTotal = Math.max(1, len - 1);
  const bigram0 = bigramBuckets[0] / bigramTotal;
  const bigram1 = bigramBuckets[1] / bigramTotal;
  const bigram2 = bigramBuckets[2] / bigramTotal;

  let q1 = 0, q2 = 0, q3 = 0, q4 = 0;
  for (let i = 0; i < len; i++) {
    const c = clean.charCodeAt(i);
    if (c >= 97 && c <= 102) q1++;
    else if (c >= 103 && c <= 108) q2++;
    else if (c >= 109 && c <= 114) q3++;
    else if (c >= 115 && c <= 122) q4++;
  }
  const normQ1 = q1 / len;
  const normQ2 = q2 / len;
  const normQ3 = q3 / len;
  const normQ4 = q4 / len;

  const rawVector = [
    normLength,
    firstChar,
    secondChar,
    penultChar,
    lastChar,
    vowelRatio,
    consonantRatio,
    prefixScore,
    suffixScore,
    bigram0,
    bigram1,
    bigram2,
    normQ1,
    normQ2,
    normQ3,
    normQ4,
  ];

  const vector = normalizeVector(rawVector);
  const initial = clean.charAt(0).toUpperCase() || '?';
  const category = suffixMatch
    ? `Suffix: -${suffixMatch}`
    : prefixMatch
    ? `Prefix: ${prefixMatch}-`
    : `Letter: ${initial}`;

  return {
    vector,
    category,
    featuresSummary: {
      'Word Length': `${len} chars`,
      'First Letter': clean.charAt(0),
      'Last Letter': clean.charAt(len - 1),
      'Vowel Ratio': `${Math.round(vowelRatio * 100)}%`,
      'Detected Prefix': prefixMatch || 'None',
      'Detected Suffix': suffixMatch || 'None',
      'Feature Dimensions': '16-D Morphological Vector',
    },
  };
}

export function buildSpellingEmbeddings(tokens) {
  if (tokens.length === 0) return [];

  const intermediate = tokens.map((tokenItem) => {
    const { vector, featuresSummary, category } = generateSpellingVector(tokenItem.token);
    return {
      id: tokenItem.id,
      word: tokenItem.token,
      count: tokenItem.count,
      vector,
      featuresSummary,
      category,
    };
  });

  const vectors = intermediate.map((item) => item.vector);
  const coords3D = projectTo3DWithPCA(vectors, 7.5);

  return intermediate.map((item, idx) => ({
    id: item.id,
    word: item.word,
    count: item.count,
    vector: item.vector,
    coordinates3D: coords3D[idx],
    category: item.category,
    featuresSummary: item.featuresSummary,
    mode: 'spelling',
  }));
}

// -------------------------------------------------------------
// 4. Option B: Calibrated Semantic Embeddings
// -------------------------------------------------------------

export const SEMANTIC_CATEGORIES = {
  ai_tech: 'AI & Technology',
  royalty_people: 'Royalty & People',
  vehicles: 'Vehicles & Transit',
  food_fruit: 'Food & Nutrition',
  animals: 'Animals & Fauna',
  nature_space: 'Nature & Cosmos',
  actions: 'Actions & Transformation',
  abstract_concepts: 'Abstract Concepts',
  common: 'General Lexicon',
};

export const SEMANTIC_CORPUS = {
  // --- AI & Technology ---
  artificial: { vector: [0.1, 0.1, 0.0, 0.95, 0.8, 0.2, 0.0, 0.0, 0.1, 0.5, 0.2, 0.3, 0.4, 0.7, 0.2, 0.9], category: 'ai_tech' },
  intelligence: { vector: [0.6, 0.3, 0.0, 0.9, 0.98, 0.1, 0.0, 0.0, 0.1, 0.6, 0.5, 0.4, 0.6, 0.8, 0.4, 0.95], category: 'ai_tech' },
  ai: { vector: [0.4, 0.2, 0.0, 0.98, 0.92, 0.1, 0.0, 0.0, 0.1, 0.6, 0.4, 0.4, 0.5, 0.8, 0.3, 0.95], category: 'ai_tech' },
  technology: { vector: [0.1, 0.2, 0.0, 0.92, 0.85, 0.4, 0.0, 0.0, 0.1, 0.6, 0.3, 0.5, 0.3, 0.85, 0.3, 0.8], category: 'ai_tech' },
  model: { vector: [0.1, 0.1, 0.0, 0.85, 0.8, 0.1, 0.0, 0.0, 0.1, 0.5, 0.2, 0.2, 0.6, 0.9, 0.2, 0.85], category: 'ai_tech' },
  models: { vector: [0.1, 0.1, 0.0, 0.85, 0.8, 0.1, 0.0, 0.0, 0.1, 0.55, 0.2, 0.2, 0.6, 0.9, 0.2, 0.85], category: 'ai_tech' },
  neural: { vector: [0.4, 0.1, 0.0, 0.92, 0.88, 0.1, 0.0, 0.1, 0.1, 0.4, 0.2, 0.3, 0.3, 0.88, 0.2, 0.85], category: 'ai_tech' },
  network: { vector: [0.1, 0.2, 0.0, 0.88, 0.75, 0.3, 0.0, 0.0, 0.1, 0.6, 0.2, 0.4, 0.5, 0.98, 0.2, 0.75], category: 'ai_tech' },
  networks: { vector: [0.1, 0.2, 0.0, 0.88, 0.75, 0.3, 0.0, 0.0, 0.1, 0.65, 0.2, 0.4, 0.5, 0.98, 0.2, 0.75], category: 'ai_tech' },
  algorithm: { vector: [0.0, 0.1, 0.0, 0.92, 0.95, 0.1, 0.0, 0.0, 0.0, 0.4, 0.2, 0.4, 0.5, 0.92, 0.3, 0.9], category: 'ai_tech' },
  algorithms: { vector: [0.0, 0.1, 0.0, 0.92, 0.95, 0.1, 0.0, 0.0, 0.0, 0.45, 0.2, 0.4, 0.5, 0.92, 0.3, 0.9], category: 'ai_tech' },
  computer: { vector: [0.1, 0.1, 0.0, 0.95, 0.75, 0.2, 0.0, 0.0, 0.0, 0.5, 0.2, 0.5, 0.4, 0.8, 0.2, 0.3], category: 'ai_tech' },
  computers: { vector: [0.1, 0.1, 0.0, 0.95, 0.75, 0.2, 0.0, 0.0, 0.0, 0.55, 0.2, 0.5, 0.4, 0.8, 0.2, 0.3], category: 'ai_tech' },
  data: { vector: [0.0, 0.1, 0.0, 0.9, 0.8, 0.1, 0.0, 0.0, 0.1, 0.6, 0.1, 0.3, 0.7, 0.85, 0.1, 0.85], category: 'ai_tech' },
  pipeline: { vector: [0.1, 0.1, 0.0, 0.82, 0.7, 0.5, 0.0, 0.0, 0.0, 0.6, 0.2, 0.5, 0.4, 0.95, 0.4, 0.7], category: 'ai_tech' },
  software: { vector: [0.0, 0.1, 0.0, 0.95, 0.8, 0.1, 0.0, 0.0, 0.0, 0.5, 0.2, 0.4, 0.6, 0.85, 0.2, 0.85], category: 'ai_tech' },
  hardware: { vector: [0.0, 0.1, 0.0, 0.92, 0.6, 0.2, 0.0, 0.0, 0.0, 0.5, 0.1, 0.3, 0.2, 0.75, 0.1, 0.2], category: 'ai_tech' },
  code: { vector: [0.0, 0.1, 0.0, 0.92, 0.82, 0.1, 0.0, 0.0, 0.0, 0.4, 0.2, 0.3, 0.88, 0.85, 0.3, 0.8], category: 'ai_tech' },
  system: { vector: [0.1, 0.3, 0.0, 0.8, 0.75, 0.3, 0.0, 0.0, 0.1, 0.7, 0.2, 0.3, 0.3, 0.95, 0.2, 0.85], category: 'ai_tech' },
  systems: { vector: [0.1, 0.3, 0.0, 0.8, 0.75, 0.3, 0.0, 0.0, 0.1, 0.75, 0.2, 0.3, 0.3, 0.95, 0.2, 0.85], category: 'ai_tech' },
  machine: { vector: [0.1, 0.1, 0.0, 0.85, 0.65, 0.4, 0.0, 0.0, 0.0, 0.6, 0.1, 0.4, 0.2, 0.75, 0.2, 0.3], category: 'ai_tech' },
  vector: { vector: [0.0, 0.1, 0.0, 0.85, 0.9, 0.2, 0.0, 0.0, 0.1, 0.4, 0.2, 0.3, 0.6, 0.85, 0.1, 0.95], category: 'ai_tech' },
  vectors: { vector: [0.0, 0.1, 0.0, 0.85, 0.9, 0.2, 0.0, 0.0, 0.1, 0.5, 0.2, 0.3, 0.6, 0.85, 0.1, 0.95], category: 'ai_tech' },
  token: { vector: [0.0, 0.1, 0.0, 0.85, 0.75, 0.1, 0.0, 0.0, 0.0, 0.3, 0.1, 0.2, 0.92, 0.75, 0.1, 0.8], category: 'ai_tech' },
  tokens: { vector: [0.0, 0.1, 0.0, 0.85, 0.75, 0.1, 0.0, 0.0, 0.0, 0.4, 0.1, 0.2, 0.92, 0.75, 0.1, 0.8], category: 'ai_tech' },
  tokenization: { vector: [0.0, 0.1, 0.0, 0.88, 0.8, 0.1, 0.0, 0.0, 0.0, 0.4, 0.2, 0.3, 0.95, 0.85, 0.5, 0.9], category: 'ai_tech' },
  vocabulary: { vector: [0.1, 0.1, 0.0, 0.75, 0.85, 0.1, 0.0, 0.0, 0.0, 0.6, 0.2, 0.2, 0.95, 0.85, 0.2, 0.9], category: 'ai_tech' },
  embedding: { vector: [0.0, 0.1, 0.0, 0.9, 0.92, 0.1, 0.0, 0.0, 0.1, 0.5, 0.3, 0.3, 0.85, 0.9, 0.3, 0.95], category: 'ai_tech' },
  embeddings: { vector: [0.0, 0.1, 0.0, 0.9, 0.92, 0.1, 0.0, 0.0, 0.1, 0.55, 0.3, 0.3, 0.85, 0.9, 0.3, 0.95], category: 'ai_tech' },

  // --- Royalty & People ---
  king: { vector: [0.95, 0.95, -0.35, 0.05, 0.5, 0.2, 0.05, 0.0, 0.1, 0.7, 0.6, 0.3, 0.4, 0.6, 0.3, 0.2], category: 'royalty_people' },
  queen: { vector: [0.95, 0.95, 0.35, 0.05, 0.5, 0.2, 0.05, 0.0, 0.1, 0.7, 0.6, 0.3, 0.4, 0.6, 0.3, 0.2], category: 'royalty_people' },
  prince: { vector: [0.9, 0.85, -0.35, 0.05, 0.45, 0.3, 0.05, 0.0, 0.1, 0.6, 0.55, 0.4, 0.3, 0.5, 0.3, 0.2], category: 'royalty_people' },
  princess: { vector: [0.9, 0.85, 0.35, 0.05, 0.45, 0.3, 0.05, 0.0, 0.1, 0.6, 0.55, 0.4, 0.3, 0.5, 0.3, 0.2], category: 'royalty_people' },
  man: { vector: [0.95, 0.2, -0.35, 0.1, 0.4, 0.2, 0.0, 0.0, 0.1, 0.5, 0.2, 0.3, 0.4, 0.3, 0.3, 0.2], category: 'royalty_people' },
  men: { vector: [0.95, 0.2, -0.35, 0.1, 0.4, 0.2, 0.0, 0.0, 0.1, 0.6, 0.2, 0.3, 0.4, 0.3, 0.3, 0.2], category: 'royalty_people' },
  woman: { vector: [0.95, 0.2, 0.35, 0.1, 0.4, 0.2, 0.0, 0.0, 0.1, 0.5, 0.25, 0.3, 0.4, 0.3, 0.3, 0.2], category: 'royalty_people' },
  women: { vector: [0.95, 0.2, 0.35, 0.1, 0.4, 0.2, 0.0, 0.0, 0.1, 0.6, 0.25, 0.3, 0.4, 0.3, 0.3, 0.2], category: 'royalty_people' },
  human: { vector: [0.98, 0.3, 0.0, 0.2, 0.6, 0.2, 0.0, 0.0, 0.3, 0.5, 0.3, 0.3, 0.5, 0.4, 0.3, 0.4], category: 'royalty_people' },
  humans: { vector: [0.98, 0.3, 0.0, 0.2, 0.6, 0.2, 0.0, 0.0, 0.3, 0.6, 0.3, 0.3, 0.5, 0.4, 0.3, 0.4], category: 'royalty_people' },
  leader: { vector: [0.95, 0.8, 0.0, 0.2, 0.6, 0.3, 0.0, 0.0, 0.1, 0.6, 0.4, 0.4, 0.5, 0.6, 0.4, 0.4], category: 'royalty_people' },
  monarch: { vector: [0.95, 0.98, 0.0, 0.05, 0.5, 0.2, 0.05, 0.0, 0.1, 0.7, 0.6, 0.3, 0.4, 0.7, 0.3, 0.3], category: 'royalty_people' },
  kingdom: { vector: [0.2, 0.9, 0.0, 0.05, 0.3, 0.2, 0.0, 0.0, 0.4, 0.85, 0.5, 0.2, 0.2, 0.8, 0.1, 0.5], category: 'royalty_people' },
  palace: { vector: [0.1, 0.85, 0.0, 0.05, 0.2, 0.1, 0.0, 0.0, 0.3, 0.8, 0.5, 0.1, 0.1, 0.8, 0.1, 0.3], category: 'royalty_people' },
  throne: { vector: [0.1, 0.9, 0.0, 0.05, 0.3, 0.1, 0.0, 0.0, 0.2, 0.6, 0.5, 0.1, 0.1, 0.6, 0.1, 0.3], category: 'royalty_people' },

  // --- Vehicles & Transport ---
  car: { vector: [0.0, 0.1, 0.0, 0.6, 0.2, 0.95, 0.0, 0.0, 0.0, 0.5, 0.2, 0.85, 0.1, 0.6, 0.2, 0.1], category: 'vehicles' },
  cars: { vector: [0.0, 0.1, 0.0, 0.6, 0.2, 0.95, 0.0, 0.0, 0.0, 0.55, 0.2, 0.85, 0.1, 0.6, 0.2, 0.1], category: 'vehicles' },
  vehicle: { vector: [0.0, 0.1, 0.0, 0.62, 0.25, 0.98, 0.0, 0.0, 0.0, 0.6, 0.2, 0.85, 0.1, 0.65, 0.2, 0.2], category: 'vehicles' },
  vehicles: { vector: [0.0, 0.1, 0.0, 0.62, 0.25, 0.98, 0.0, 0.0, 0.0, 0.65, 0.2, 0.85, 0.1, 0.65, 0.2, 0.2], category: 'vehicles' },
  automobile: { vector: [0.0, 0.1, 0.0, 0.65, 0.25, 0.96, 0.0, 0.0, 0.0, 0.55, 0.2, 0.85, 0.1, 0.65, 0.2, 0.15], category: 'vehicles' },
  truck: { vector: [0.0, 0.1, 0.0, 0.5, 0.15, 0.92, 0.0, 0.0, 0.0, 0.75, 0.1, 0.7, 0.1, 0.6, 0.2, 0.1], category: 'vehicles' },
  train: { vector: [0.0, 0.1, 0.0, 0.55, 0.2, 0.92, 0.0, 0.0, 0.1, 0.85, 0.2, 0.8, 0.1, 0.8, 0.2, 0.1], category: 'vehicles' },
  plane: { vector: [0.0, 0.1, 0.0, 0.7, 0.3, 0.96, 0.0, 0.0, 0.3, 0.8, 0.3, 0.98, 0.1, 0.7, 0.2, 0.1], category: 'vehicles' },
  airplane: { vector: [0.0, 0.1, 0.0, 0.72, 0.3, 0.96, 0.0, 0.0, 0.3, 0.8, 0.3, 0.98, 0.1, 0.7, 0.2, 0.1], category: 'vehicles' },
  road: { vector: [0.0, 0.1, 0.0, 0.3, 0.1, 0.75, 0.0, 0.0, 0.4, 0.75, 0.1, 0.4, 0.1, 0.7, 0.1, 0.2], category: 'vehicles' },
  drive: { vector: [0.3, 0.1, 0.0, 0.3, 0.2, 0.9, 0.0, 0.0, 0.1, 0.4, 0.2, 0.8, 0.1, 0.4, 0.9, 0.3], category: 'vehicles' },
  drove: { vector: [0.3, 0.1, 0.0, 0.3, 0.2, 0.9, 0.0, 0.0, 0.1, 0.4, 0.2, 0.8, 0.1, 0.4, 0.85, 0.3], category: 'vehicles' },
  driver: { vector: [0.9, 0.1, 0.0, 0.3, 0.3, 0.9, 0.0, 0.0, 0.1, 0.5, 0.3, 0.6, 0.1, 0.5, 0.7, 0.2], category: 'vehicles' },

  // --- Food & Fruit ---
  apple: { vector: [0.1, 0.0, 0.0, 0.0, 0.05, 0.0, 0.98, 0.0, 0.4, 0.15, 0.3, 0.0, 0.0, 0.1, 0.05, 0.1], category: 'food_fruit' },
  apples: { vector: [0.1, 0.0, 0.0, 0.0, 0.05, 0.0, 0.98, 0.0, 0.4, 0.2, 0.3, 0.0, 0.0, 0.1, 0.05, 0.1], category: 'food_fruit' },
  banana: { vector: [0.1, 0.0, 0.0, 0.0, 0.05, 0.0, 0.98, 0.0, 0.4, 0.15, 0.3, 0.0, 0.0, 0.1, 0.05, 0.1], category: 'food_fruit' },
  bananas: { vector: [0.1, 0.0, 0.0, 0.0, 0.05, 0.0, 0.98, 0.0, 0.4, 0.2, 0.3, 0.0, 0.0, 0.1, 0.05, 0.1], category: 'food_fruit' },
  orange: { vector: [0.1, 0.0, 0.0, 0.0, 0.05, 0.0, 0.96, 0.0, 0.4, 0.15, 0.3, 0.0, 0.0, 0.1, 0.05, 0.1], category: 'food_fruit' },
  fruit: { vector: [0.12, 0.0, 0.0, 0.0, 0.05, 0.0, 0.99, 0.0, 0.45, 0.2, 0.35, 0.0, 0.0, 0.1, 0.05, 0.25], category: 'food_fruit' },
  fruits: { vector: [0.12, 0.0, 0.0, 0.0, 0.05, 0.0, 0.99, 0.0, 0.45, 0.25, 0.35, 0.0, 0.0, 0.1, 0.05, 0.25], category: 'food_fruit' },
  bread: { vector: [0.1, 0.0, 0.0, 0.0, 0.05, 0.0, 0.92, 0.0, 0.3, 0.25, 0.3, 0.0, 0.0, 0.1, 0.05, 0.1], category: 'food_fruit' },
  food: { vector: [0.15, 0.0, 0.0, 0.0, 0.05, 0.0, 0.99, 0.0, 0.4, 0.3, 0.4, 0.0, 0.0, 0.15, 0.05, 0.25], category: 'food_fruit' },
  eat: { vector: [0.7, 0.0, 0.0, 0.0, 0.1, 0.0, 0.85, 0.1, 0.2, 0.3, 0.4, 0.2, 0.0, 0.1, 0.9, 0.2], category: 'food_fruit' },
  ate: { vector: [0.7, 0.0, 0.0, 0.0, 0.1, 0.0, 0.85, 0.1, 0.2, 0.3, 0.4, 0.2, 0.0, 0.1, 0.85, 0.2], category: 'food_fruit' },
  lunch: { vector: [0.3, 0.1, 0.0, 0.0, 0.1, 0.0, 0.88, 0.0, 0.2, 0.3, 0.5, 0.1, 0.1, 0.3, 0.2, 0.3], category: 'food_fruit' },

  // --- Animals & Fauna ---
  cat: { vector: [0.92, 0.0, 0.0, 0.05, 0.3, 0.2, 0.0, 0.95, 0.5, 0.3, 0.6, 0.6, 0.1, 0.2, 0.3, 0.1], category: 'animals' },
  cats: { vector: [0.92, 0.0, 0.0, 0.05, 0.3, 0.2, 0.0, 0.95, 0.5, 0.35, 0.6, 0.6, 0.1, 0.2, 0.3, 0.1], category: 'animals' },
  dog: { vector: [0.92, 0.0, 0.0, 0.05, 0.35, 0.2, 0.0, 0.95, 0.5, 0.35, 0.65, 0.65, 0.1, 0.2, 0.3, 0.1], category: 'animals' },
  dogs: { vector: [0.92, 0.0, 0.0, 0.05, 0.35, 0.2, 0.0, 0.95, 0.5, 0.4, 0.65, 0.65, 0.1, 0.2, 0.3, 0.1], category: 'animals' },
  pet: { vector: [0.9, 0.0, 0.0, 0.05, 0.3, 0.1, 0.0, 0.92, 0.4, 0.3, 0.7, 0.5, 0.1, 0.2, 0.2, 0.2], category: 'animals' },
  pets: { vector: [0.9, 0.0, 0.0, 0.05, 0.3, 0.1, 0.0, 0.92, 0.4, 0.35, 0.7, 0.5, 0.1, 0.2, 0.2, 0.2], category: 'animals' },
  lion: { vector: [0.95, 0.4, -0.2, 0.05, 0.3, 0.3, 0.0, 0.98, 0.6, 0.55, 0.5, 0.75, 0.1, 0.3, 0.4, 0.1], category: 'animals' },
  tiger: { vector: [0.95, 0.3, -0.1, 0.05, 0.3, 0.3, 0.0, 0.98, 0.6, 0.55, 0.5, 0.78, 0.1, 0.3, 0.4, 0.1], category: 'animals' },
  animal: { vector: [0.95, 0.1, 0.0, 0.05, 0.3, 0.2, 0.0, 0.99, 0.6, 0.45, 0.4, 0.6, 0.1, 0.3, 0.3, 0.3], category: 'animals' },
  animals: { vector: [0.95, 0.1, 0.0, 0.05, 0.3, 0.2, 0.0, 0.99, 0.6, 0.55, 0.4, 0.6, 0.1, 0.3, 0.3, 0.3], category: 'animals' },

  // --- Nature & Cosmos ---
  sun: { vector: [0.0, 0.3, 0.0, 0.05, 0.2, 0.1, 0.0, 0.0, 0.98, 0.9, 0.7, 0.3, 0.0, 0.4, 0.1, 0.2], category: 'nature_space' },
  moon: { vector: [0.0, 0.2, 0.1, 0.05, 0.2, 0.1, 0.0, 0.0, 0.96, 0.8, 0.5, 0.2, 0.0, 0.4, 0.1, 0.2], category: 'nature_space' },
  star: { vector: [0.0, 0.3, 0.0, 0.1, 0.3, 0.2, 0.0, 0.0, 0.99, 0.95, 0.7, 0.4, 0.0, 0.5, 0.1, 0.3], category: 'nature_space' },
  stars: { vector: [0.0, 0.3, 0.0, 0.1, 0.3, 0.2, 0.0, 0.0, 0.99, 0.98, 0.7, 0.4, 0.0, 0.5, 0.1, 0.3], category: 'nature_space' },
  earth: { vector: [0.4, 0.2, 0.0, 0.1, 0.4, 0.2, 0.1, 0.2, 0.95, 0.88, 0.6, 0.3, 0.1, 0.6, 0.1, 0.3], category: 'nature_space' },
  planet: { vector: [0.0, 0.2, 0.0, 0.1, 0.4, 0.2, 0.0, 0.0, 0.98, 0.9, 0.5, 0.3, 0.0, 0.6, 0.1, 0.3], category: 'nature_space' },
  space: { vector: [0.0, 0.1, 0.0, 0.6, 0.75, 0.3, 0.0, 0.0, 0.95, 0.98, 0.2, 0.3, 0.3, 0.7, 0.1, 0.9], category: 'nature_space' },
  forest: { vector: [0.8, 0.0, 0.0, 0.05, 0.1, 0.0, 0.1, 0.2, 0.95, 0.7, 0.5, 0.0, 0.0, 0.5, 0.1, 0.2], category: 'nature_space' },

  // --- Actions & Concepts ---
  learn: { vector: [0.8, 0.1, 0.0, 0.6, 0.95, 0.1, 0.0, 0.0, 0.1, 0.4, 0.6, 0.3, 0.5, 0.6, 0.92, 0.8], category: 'actions' },
  learning: { vector: [0.8, 0.1, 0.0, 0.6, 0.95, 0.1, 0.0, 0.0, 0.1, 0.45, 0.6, 0.3, 0.5, 0.65, 0.92, 0.85], category: 'actions' },
  transform: { vector: [0.2, 0.2, 0.0, 0.6, 0.7, 0.3, 0.0, 0.0, 0.2, 0.5, 0.5, 0.6, 0.3, 0.7, 0.95, 0.8], category: 'actions' },
  transforming: { vector: [0.2, 0.2, 0.0, 0.6, 0.7, 0.3, 0.0, 0.0, 0.2, 0.55, 0.5, 0.65, 0.3, 0.7, 0.98, 0.8], category: 'actions' },
  process: { vector: [0.1, 0.1, 0.0, 0.75, 0.75, 0.3, 0.0, 0.0, 0.1, 0.5, 0.3, 0.5, 0.4, 0.85, 0.9, 0.85], category: 'actions' },
  pattern: { vector: [0.0, 0.1, 0.0, 0.7, 0.85, 0.1, 0.0, 0.0, 0.3, 0.4, 0.3, 0.2, 0.5, 0.9, 0.2, 0.9], category: 'abstract_concepts' },
  patterns: { vector: [0.0, 0.1, 0.0, 0.7, 0.85, 0.1, 0.0, 0.0, 0.3, 0.5, 0.3, 0.2, 0.5, 0.9, 0.2, 0.9], category: 'abstract_concepts' },
  language: { vector: [0.7, 0.2, 0.0, 0.7, 0.85, 0.1, 0.0, 0.0, 0.1, 0.6, 0.4, 0.3, 0.98, 0.75, 0.3, 0.85], category: 'abstract_concepts' },
  text: { vector: [0.0, 0.1, 0.0, 0.75, 0.7, 0.1, 0.0, 0.0, 0.0, 0.4, 0.2, 0.2, 0.95, 0.65, 0.1, 0.65], category: 'abstract_concepts' },
};

export function getSemanticVector(word) {
  const clean = word.toLowerCase().trim();

  if (SEMANTIC_CORPUS[clean]) {
    const entry = SEMANTIC_CORPUS[clean];
    const catLabel = SEMANTIC_CATEGORIES[entry.category] || 'General Lexicon';
    return {
      vector: normalizeVector(entry.vector),
      category: catLabel,
      source: 'Pretrained Semantic Corpus',
      featuresSummary: {
        'Corpus Match': 'Exact Pretrained Vector',
        'Semantic Cluster': catLabel,
        'Living / Animate': (entry.vector[0] * 100).toFixed(0) + '%',
        'Technology / AI': (entry.vector[3] * 100).toFixed(0) + '%',
        'Science / Intellect': (entry.vector[4] * 100).toFixed(0) + '%',
        'Transit / Motion': (entry.vector[5] * 100).toFixed(0) + '%',
        'Edible / Food': (entry.vector[6] * 100).toFixed(0) + '%',
      },
    };
  }

  const candidateStems = [
    clean.replace(/s$/, ''),
    clean.replace(/es$/, ''),
    clean.replace(/ing$/, ''),
    clean.replace(/ed$/, ''),
    clean.replace(/er$/, ''),
    clean.replace(/tion$/, ''),
    clean.replace(/ly$/, ''),
    clean.replace(/ment$/, ''),
  ];

  for (const stem of candidateStems) {
    if (stem.length > 2 && SEMANTIC_CORPUS[stem]) {
      const baseEntry = SEMANTIC_CORPUS[stem];
      const catLabel = SEMANTIC_CATEGORIES[baseEntry.category] || 'General Lexicon';
      const derived = baseEntry.vector.map((val, idx) => {
        if (idx === 14) return Math.min(1.0, val + 0.1);
        return val;
      });

      return {
        vector: normalizeVector(derived),
        category: catLabel,
        source: `Derived from root "${stem}"`,
        featuresSummary: {
          'Corpus Match': `Morphological Stem (${stem})`,
          'Semantic Cluster': catLabel,
          'Living / Animate': (baseEntry.vector[0] * 100).toFixed(0) + '%',
          'Technology / AI': (baseEntry.vector[3] * 100).toFixed(0) + '%',
          'Derived Inflection': clean,
        },
      };
    }
  }

  // Lexical semantic approximation fallback
  const hashVal = clean.split('').reduce((acc, char, i) => acc + char.charCodeAt(0) * (i + 1), 0);
  const synthetic = new Array(16).fill(0);
  for (let i = 0; i < 16; i++) {
    synthetic[i] = 0.15 + 0.1 * Math.sin(hashVal * 0.13 + i * 1.6);
  }

  if (/^(algo|auto|cyber|digi|info|micro|nano|neuro|techno|tele)/.test(clean)) {
    synthetic[3] = 0.85;
    synthetic[4] = 0.75;
    synthetic[13] = 0.8;
  } else if (/(tion|sion|ence|ance|ology|ism|ment)$/.test(clean)) {
    synthetic[15] = 0.9;
    synthetic[4] = 0.6;
  } else if (/(able|ive|ous|ful|less)$/.test(clean)) {
    synthetic[10] = 0.6;
    synthetic[15] = 0.8;
  }

  return {
    vector: normalizeVector(synthetic),
    category: 'Contextual / General Vocabulary',
    source: 'Lexical Semantic Approximation',
    featuresSummary: {
      'Corpus Match': 'Lexical Fallback',
      'Semantic Cluster': 'General Lexicon',
      'Word Length': `${clean.length} characters`,
      'Deterministic Vector': '16-D Synthesized',
    },
  };
}

export function buildSemanticEmbeddings(tokens) {
  if (tokens.length === 0) return [];

  const intermediate = tokens.map((tokenItem) => {
    const { vector, category, featuresSummary } = getSemanticVector(tokenItem.token);
    return {
      id: tokenItem.id,
      word: tokenItem.token,
      count: tokenItem.count,
      vector,
      category,
      featuresSummary,
    };
  });

  const vectors = intermediate.map((item) => item.vector);
  const coords3D = projectTo3DWithPCA(vectors, 7.5);

  return intermediate.map((item, idx) => ({
    id: item.id,
    word: item.word,
    count: item.count,
    vector: item.vector,
    coordinates3D: coords3D[idx],
    category: item.category,
    featuresSummary: item.featuresSummary,
    mode: 'semantic',
  }));
}

export function findNearestNeighbors(targetWord, embeddings, k = 5) {
  const target = embeddings.find((e) => e.word.toLowerCase() === targetWord.toLowerCase());
  if (!target) return [];

  const neighbors = embeddings
    .filter((e) => e.word.toLowerCase() !== targetWord.toLowerCase())
    .map((e) => {
      const sim = cosineSimilarity(target.vector, e.vector);
      const dist = euclideanDistance(target.vector, e.vector);
      return {
        word: e.word,
        id: e.id,
        similarity: sim,
        distance: dist,
        coordinates3D: e.coordinates3D,
      };
    });

  neighbors.sort((a, b) => b.similarity - a.similarity);
  return neighbors.slice(0, k);
}

export function compareWords(embA, embB) {
  return {
    wordA: embA.word,
    wordB: embB.word,
    cosineSimilarity: cosineSimilarity(embA.vector, embB.vector),
    euclideanDistance: euclideanDistance(embA.vector, embB.vector),
    dotProduct: dotProduct(embA.vector, embB.vector),
  };
}
