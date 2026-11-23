/**
 * LLM Pipeline Explorer - Main Application Controller
 * Pure Vanilla JavaScript (ES6 Module)
 *
 * Coordinates state, event listeners, UI rendering,
 * and user interactions between tokenizer, vocabulary,
 * embeddings, and 3D vector-space visualization.
 */

import { getLiveInputStats } from './tokenizer.js';
import { buildVocabulary, filterAndSortVocabulary } from './vocabulary.js';
import {
  buildSpellingEmbeddings,
  buildSemanticEmbeddings,
  findNearestNeighbors,
  compareWords,
  vectorArithmetic,
  cosineSimilarity,
} from './embeddings.js';
import { EmbeddingVisualizer } from './visualization.js';

// -------------------------------------------------------------
// Presets Data
// -------------------------------------------------------------
const PRESETS = [
  {
    id: 'ai_intro',
    title: 'AI & Language Models',
    text: 'Artificial intelligence is transforming technology. Language models learn patterns from text. Neural networks process data with machine learning algorithms. Deep learning enables computer systems to learn language patterns.',
  },
  {
    id: 'semantic_clusters',
    title: 'Semantic Analogies & Clusters',
    text: 'The king and queen ruled the kingdom from the royal throne. A car and a vehicle drove along the road. An apple and a banana were served as fruit for lunch. The monarch admired the royal palace while a driver steered the automobile.',
  },
  {
    id: 'morphology_spelling',
    title: 'Morphology & Spelling Variations',
    text: 'The model is modeling new models. Learners learn learning algorithms while processing processed data. Transforming transformers transforms computational computation.',
  },
  {
    id: 'nature_cosmos',
    title: 'Cosmos & Living Nature',
    text: 'The sun and moon shine above the earth. Stars and planets drift through deep space. A lion and a tiger roam the forest while a dog and a cat rest as loyal pets.',
  },
];

// -------------------------------------------------------------
// Application State
// -------------------------------------------------------------
const state = {
  text: PRESETS[0].text,
  vocabularyResult: null,
  embeddingMode: 'spelling', // 'spelling' | 'semantic'
  embeddings: [],
  selectedWord: null,
  currentStage: 1,

  // Table filters
  vocabSearch: '',
  vocabSortBy: 'id',
  vocabSortOrder: 'asc',

  embeddingSearch: '',

  // Comparison
  compareTargetWord: '',

  // Analogy sandbox
  analogyA: 'king',
  analogyB: 'man',
  analogyC: 'woman',
};

let visualizer = null;

// -------------------------------------------------------------
// Initialization
// -------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
  initVisualizer();
  bindEvents();

  // Load initial demo state
  loadText(state.text);
});

function initVisualizer() {
  const container = document.getElementById('viz-3d-container');
  if (!container) return;

  visualizer = new EmbeddingVisualizer(container, {
    onSelectWord: (word) => {
      selectWord(word);
    },
  });
}

// -------------------------------------------------------------
// Core Actions & State Transitions
// -------------------------------------------------------------
function loadText(inputText) {
  state.text = inputText;

  // Sync textareas
  const vocabInput = document.getElementById('vocab-input');
  const embedInput = document.getElementById('embedding-input');
  if (vocabInput) vocabInput.value = inputText;
  if (embedInput) embedInput.value = inputText;

  updateLiveStats(inputText);
  executeBuildVocabulary(inputText);
}

function updateLiveStats(text) {
  const stats = getLiveInputStats(text);

  const elWords = document.getElementById('stat-live-words');
  const elOccur = document.getElementById('stat-token-occur');
  const elUnique = document.getElementById('stat-unique-words');
  const elChars = document.getElementById('stat-char-count');

  if (elWords) elWords.textContent = stats.rawWordCount;
  if (elOccur) elOccur.textContent = stats.totalTokenCount;
  if (elUnique) elUnique.textContent = stats.uniqueTokenCount;
  if (elChars) elChars.textContent = stats.characterCount;
}

function executeBuildVocabulary(inputText) {
  state.text = inputText;
  state.vocabularyResult = buildVocabulary(inputText);

  // Set default selected word
  if (state.vocabularyResult.tokens.length > 0) {
    if (!state.selectedWord || !state.vocabularyResult.tokenToId.has(state.selectedWord)) {
      state.selectedWord = state.vocabularyResult.tokens[0].token;
    }
  } else {
    state.selectedWord = null;
  }

  // Update embeddings
  updateEmbeddings();

  // Render UI sections
  renderPipelineProgress();
  renderTokenSequence();
  renderVocabularyTable();
  renderWordDetails();
  renderExplanationSection();
}

function updateEmbeddings() {
  if (!state.vocabularyResult || state.vocabularyResult.tokens.length === 0) {
    state.embeddings = [];
    if (visualizer) visualizer.setEmbeddings([], state.embeddingMode);
    renderEmbeddingTable();
    return;
  }

  const tokens = state.vocabularyResult.tokens;
  if (state.embeddingMode === 'spelling') {
    state.embeddings = buildSpellingEmbeddings(tokens);
  } else {
    state.embeddings = buildSemanticEmbeddings(tokens);
  }

  if (visualizer) {
    visualizer.setEmbeddings(state.embeddings, state.embeddingMode);
    visualizer.setSelectedWord(state.selectedWord);
  }

  renderEmbeddingTable();
  renderWordDetails();
  renderExplanationSection();
}

function selectWord(word) {
  if (!word) return;
  state.selectedWord = word.toLowerCase();

  if (visualizer) {
    visualizer.setSelectedWord(state.selectedWord);
  }

  renderTokenSequence();
  renderVocabularyTable();
  renderEmbeddingTable();
  renderWordDetails();
}

function setEmbeddingMode(newMode) {
  if (state.embeddingMode === newMode) return;
  state.embeddingMode = newMode;

  // Update radio cards in UI
  const cardSpelling = document.getElementById('card-mode-spelling');
  const cardSemantic = document.getElementById('card-mode-semantic');
  const radioSpelling = document.getElementById('radio-mode-spelling');
  const radioSemantic = document.getElementById('radio-mode-semantic');
  const badgeMethod = document.getElementById('badge-embedding-method');
  const badgeVizMode = document.getElementById('badge-viz-mode');

  if (newMode === 'spelling') {
    cardSpelling?.classList.add('selected');
    cardSemantic?.classList.remove('selected');
    if (radioSpelling) radioSpelling.checked = true;
    if (badgeMethod) badgeMethod.textContent = 'Method: Character N-Gram & Morphological Projection';
    if (badgeVizMode) {
      badgeVizMode.textContent = 'Spelling Coordinates';
      badgeVizMode.className = 'mode-tag spelling';
    }
  } else {
    cardSemantic?.classList.add('selected');
    cardSpelling?.classList.remove('selected');
    if (radioSemantic) radioSemantic.checked = true;
    if (badgeMethod) badgeMethod.textContent = 'Method: Pretrained 16-D Semantic Feature Space';
    if (badgeVizMode) {
      badgeVizMode.textContent = 'Semantic Coordinates';
      badgeVizMode.className = 'mode-tag semantic';
    }
  }

  updateEmbeddings();
}

// -------------------------------------------------------------
// UI Rendering Functions
// -------------------------------------------------------------

function renderPipelineProgress() {
  const wordsCount = state.vocabularyResult ? state.vocabularyResult.totalWordCount : 0;
  const uniqueCount = state.vocabularyResult ? state.vocabularyResult.uniqueWordCount : 0;

  const sub1 = document.getElementById('step-sub-1');
  const sub2 = document.getElementById('step-sub-2');
  const sub4 = document.getElementById('step-sub-4');

  if (sub1) sub1.textContent = `${wordsCount} words`;
  if (sub2) sub2.textContent = `${uniqueCount} unique`;
  if (sub4) sub4.textContent = state.embeddingMode === 'spelling' ? 'Spelling 16-D' : 'Semantic 16-D';
}

function renderTokenSequence() {
  const container = document.getElementById('token-sequence-viewer');
  if (!container || !state.vocabularyResult) return;

  const { rawTokens, tokenIds } = state.vocabularyResult;
  if (rawTokens.length === 0) {
    container.classList.add('hidden');
    return;
  }

  container.classList.remove('hidden');

  const countBadge = document.getElementById('token-stream-count');
  if (countBadge) countBadge.textContent = `Length: ${rawTokens.length} tokens`;

  // Render raw tensor array [0, 1, 2, ...]
  const arrayContainer = document.getElementById('token-ids-array');
  if (arrayContainer) {
    let arrayHtml = '<span class="bracket">[</span> ';
    tokenIds.forEach((id, idx) => {
      const word = rawTokens[idx];
      const isSelected = state.selectedWord === word;
      arrayHtml += `<button class="token-id-chip ${isSelected ? 'selected' : ''}" data-word="${word}">${id}</button>`;
      if (idx < tokenIds.length - 1) {
        arrayHtml += '<span class="comma">, </span>';
      }
    });
    arrayHtml += ' <span class="bracket">]</span>';
    arrayContainer.innerHTML = arrayHtml;

    arrayContainer.querySelectorAll('.token-id-chip').forEach((chip) => {
      chip.addEventListener('click', (e) => {
        const w = e.target.getAttribute('data-word');
        if (w) selectWord(w);
      });
    });
  }

  // Render paired word + ID badges
  const badgesContainer = document.getElementById('token-badges-list');
  if (badgesContainer) {
    let badgesHtml = '';
    rawTokens.forEach((token, idx) => {
      const id = tokenIds[idx];
      const isSelected = state.selectedWord === token;
      badgesHtml += `
        <button class="token-badge-pair ${isSelected ? 'selected' : ''}" data-word="${token}">
          <span class="badge-token">${token}</span>
          <span class="badge-id">${id}</span>
        </button>
      `;
    });
    badgesContainer.innerHTML = badgesHtml;

    badgesContainer.querySelectorAll('.token-badge-pair').forEach((btn) => {
      btn.addEventListener('click', (e) => {
        const w = e.currentTarget.getAttribute('data-word');
        if (w) selectWord(w);
      });
    });
  }
}

function renderVocabularyTable() {
  const tbody = document.getElementById('vocab-table-body');
  const sizeBadge = document.getElementById('vocab-size-badge');
  const totalCountBadge = document.getElementById('vocab-total-badge');
  if (!tbody || !state.vocabularyResult) return;

  const { tokens } = state.vocabularyResult;
  if (sizeBadge) sizeBadge.textContent = `${tokens.length} Unique Tokens`;
  if (totalCountBadge) totalCountBadge.textContent = `${tokens.length} unique tokens`;

  const filtered = filterAndSortVocabulary(
    tokens,
    state.vocabSearch,
    state.vocabSortBy,
    state.vocabSortOrder
  );

  if (filtered.length === 0) {
    tbody.innerHTML = `<tr><td colspan="3" class="table-empty">No matching tokens found.</td></tr>`;
    return;
  }

  let html = '';
  filtered.forEach((item) => {
    const isSelected = state.selectedWord === item.token;
    html += `
      <tr class="${isSelected ? 'selected-row' : ''}" data-word="${item.token}">
        <td class="col-id"><span class="id-pill ${isSelected ? 'selected' : ''}">${item.id}</span></td>
        <td class="col-token">
          <strong>${item.token}</strong>
          ${isSelected ? '<span class="selected-check">✓</span>' : ''}
        </td>
        <td class="col-count"><span class="count-pill">${item.count}</span></td>
      </tr>
    `;
  });

  tbody.innerHTML = html;

  tbody.querySelectorAll('tr').forEach((row) => {
    row.addEventListener('click', (e) => {
      const w = e.currentTarget.getAttribute('data-word');
      if (w) selectWord(w);
    });
  });
}

function renderEmbeddingTable() {
  const tbody = document.getElementById('embedding-table-body');
  const countBadge = document.getElementById('embedding-count-badge');
  if (!tbody) return;

  if (countBadge) {
    countBadge.textContent = `${state.embeddings.length} Vectors (16-D)`;
  }

  let filtered = state.embeddings;
  if (state.embeddingSearch && state.embeddingSearch.trim()) {
    const q = state.embeddingSearch.trim().toLowerCase();
    filtered = filtered.filter(
      (e) =>
        e.word.includes(q) ||
        e.id.toString() === q ||
        e.category.toLowerCase().includes(q)
    );
  }

  if (filtered.length === 0) {
    tbody.innerHTML = `<tr><td colspan="5" class="table-empty">No matching vectors found.</td></tr>`;
    return;
  }

  let html = '';
  filtered.forEach((item) => {
    const isSelected = state.selectedWord === item.word;

    // Vector heatmap blocks
    const heatmap = item.vector
      .map((val, dimIdx) => {
        const intensity = Math.min(1, Math.abs(val));
        const color =
          val >= 0
            ? `rgba(37, 99, 235, ${0.15 + intensity * 0.85})`
            : `rgba(225, 29, 72, ${0.15 + intensity * 0.85})`;
        return `<div class="vec-cell" style="background-color: ${color};" title="Dim ${dimIdx}: ${val.toFixed(3)}"></div>`;
      })
      .join('');

    const previewNums = item.vector
      .slice(0, 4)
      .map((v) => v.toFixed(2))
      .join(', ');

    html += `
      <tr class="${isSelected ? 'selected-row' : ''}" data-word="${item.word}">
        <td class="col-id font-mono"><strong>${item.id}</strong></td>
        <td class="col-token"><strong>${item.word}</strong></td>
        <td><span class="cat-pill">${item.category}</span></td>
        <td>
          <div class="vec-preview-row">
            <div class="vec-heatmap">${heatmap}</div>
            <span class="vec-nums">[${previewNums}, ...]</span>
          </div>
        </td>
        <td class="text-right">
          <button class="inspect-btn ${isSelected ? 'active' : ''}" data-word="${item.word}">
            ${isSelected ? 'Selected' : 'Inspect'}
          </button>
        </td>
      </tr>
    `;
  });

  tbody.innerHTML = html;

  tbody.querySelectorAll('tr').forEach((row) => {
    row.addEventListener('click', (e) => {
      const w = e.currentTarget.getAttribute('data-word');
      if (w) selectWord(w);
    });
  });

  tbody.querySelectorAll('.inspect-btn').forEach((btn) => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const w = e.currentTarget.getAttribute('data-word');
      if (w) selectWord(w);
    });
  });
}

function renderWordDetails() {
  const panel = document.getElementById('word-details-content');
  const emptyState = document.getElementById('word-details-empty');
  if (!panel || !emptyState) return;

  if (!state.selectedWord) {
    panel.classList.add('hidden');
    emptyState.classList.remove('hidden');
    return;
  }

  const selectedEmb = state.embeddings.find(
    (e) => e.word.toLowerCase() === state.selectedWord.toLowerCase()
  );

  if (!selectedEmb) {
    panel.classList.add('hidden');
    emptyState.classList.remove('hidden');
    return;
  }

  emptyState.classList.add('hidden');
  panel.classList.remove('hidden');

  // Header info
  document.getElementById('detail-word-title').textContent = `"${selectedEmb.word}"`;
  document.getElementById('detail-word-id').textContent = `ID: ${selectedEmb.id}`;
  document.getElementById('detail-word-cat').textContent = selectedEmb.category;
  document.getElementById('detail-word-count').textContent = `Appears ${selectedEmb.count} ${
    selectedEmb.count === 1 ? 'time' : 'times'
  } in corpus`;

  // 3D Coordinates
  document.getElementById('coord-x').textContent = selectedEmb.coordinates3D[0];
  document.getElementById('coord-y').textContent = selectedEmb.coordinates3D[1];
  document.getElementById('coord-z').textContent = selectedEmb.coordinates3D[2];

  // 16D vector visual bars
  const vecBarsContainer = document.getElementById('detail-vector-bars');
  if (vecBarsContainer) {
    let barsHtml = '';
    selectedEmb.vector.forEach((val, idx) => {
      const absVal = Math.abs(val);
      const heightPct = Math.max(12, Math.round(absVal * 100));
      const isPos = val >= 0;
      barsHtml += `
        <div class="vec-bar-col" title="Dim ${idx}: ${val.toFixed(4)}">
          <div class="vec-bar-track">
            <div class="vec-bar-fill ${isPos ? 'pos' : 'neg'}" style="height: ${heightPct}%;"></div>
          </div>
          <span class="vec-bar-label">d${idx}</span>
        </div>
      `;
    });
    vecBarsContainer.innerHTML = barsHtml;
  }

  // Feature breakdown
  const featuresContainer = document.getElementById('detail-features-grid');
  if (featuresContainer && selectedEmb.featuresSummary) {
    let featHtml = '';
    for (const [key, val] of Object.entries(selectedEmb.featuresSummary)) {
      featHtml += `
        <div class="feature-item">
          <div class="feature-key">${key}</div>
          <div class="feature-val">${val}</div>
        </div>
      `;
    }
    featuresContainer.innerHTML = featHtml;
  }

  // Nearest neighbors
  const neighborsContainer = document.getElementById('detail-neighbors-list');
  if (neighborsContainer) {
    const neighbors = findNearestNeighbors(selectedEmb.word, state.embeddings, 5);
    if (neighbors.length === 0) {
      neighborsContainer.innerHTML = '<p class="text-muted italic">No other words in vocabulary.</p>';
    } else {
      let nHtml = '';
      neighbors.forEach((n) => {
        const simPct = Math.max(0, Math.min(100, Math.round(n.similarity * 100)));
        nHtml += `
          <div class="neighbor-row" data-word="${n.word}">
            <div class="neighbor-name">
              <span class="neighbor-dot"></span>
              <strong>${n.word}</strong>
            </div>
            <div class="neighbor-metric">
              <div class="neighbor-meter"><div class="neighbor-meter-fill" style="width: ${simPct}%;"></div></div>
              <span class="neighbor-val">${n.similarity.toFixed(3)}</span>
            </div>
          </div>
        `;
      });
      neighborsContainer.innerHTML = nHtml;

      neighborsContainer.querySelectorAll('.neighbor-row').forEach((row) => {
        row.addEventListener('click', (e) => {
          const w = e.currentTarget.getAttribute('data-word');
          if (w) selectWord(w);
        });
      });
    }
  }

  // Populate comparison dropdown
  const compSelect = document.getElementById('compare-word-select');
  if (compSelect) {
    let optionsHtml = '<option value="">Select word to compare...</option>';
    state.embeddings
      .filter((e) => e.word.toLowerCase() !== selectedEmb.word.toLowerCase())
      .forEach((e) => {
        const isSel = state.compareTargetWord.toLowerCase() === e.word.toLowerCase();
        optionsHtml += `<option value="${e.word}" ${isSel ? 'selected' : ''}>${e.word} (ID: ${e.id})</option>`;
      });
    compSelect.innerHTML = optionsHtml;
  }

  renderComparisonResult();
}

function renderComparisonResult() {
  const resultBox = document.getElementById('comparison-result-box');
  if (!resultBox) return;

  if (!state.selectedWord || !state.compareTargetWord) {
    resultBox.classList.add('hidden');
    return;
  }

  const embA = state.embeddings.find((e) => e.word.toLowerCase() === state.selectedWord.toLowerCase());
  const embB = state.embeddings.find((e) => e.word.toLowerCase() === state.compareTargetWord.toLowerCase());

  if (!embA || !embB) {
    resultBox.classList.add('hidden');
    return;
  }

  resultBox.classList.remove('hidden');
  const comp = compareWords(embA, embB);

  document.getElementById('comp-pair-title').textContent = `"${embA.word}" ↔ "${embB.word}"`;
  document.getElementById('comp-cosine-val').textContent = `Cosine: ${comp.cosineSimilarity.toFixed(4)}`;
  document.getElementById('comp-dot-val').textContent = comp.dotProduct.toFixed(4);
  document.getElementById('comp-dist-val').textContent = comp.euclideanDistance.toFixed(4);

  let explanation = '';
  if (comp.cosineSimilarity > 0.7) {
    explanation = 'High directional alignment: Vectors point in nearly the same conceptual direction.';
  } else if (comp.cosineSimilarity > 0.3) {
    explanation = 'Moderate correlation: Some shared features exist between these two words.';
  } else {
    explanation = 'Orthogonal / distant: Vectors occupy distinctly separated regions of the vector space.';
  }
  document.getElementById('comp-explanation').textContent = explanation;
}

function renderExplanationSection() {
  const modeTitle = document.getElementById('explain-mode-title');
  const modeDesc = document.getElementById('explain-mode-desc');
  const explainBox = document.getElementById('explain-mode-card');

  const exKingQueen = document.getElementById('ex-king-queen');
  const exCarVehicle = document.getElementById('ex-car-vehicle');
  const exAppleBanana = document.getElementById('ex-apple-banana');

  if (state.embeddingMode === 'spelling') {
    if (modeTitle) modeTitle.textContent = 'Active Mode: Spelling-Based Embeddings';
    if (modeDesc) {
      modeDesc.textContent =
        'The model represents words using numerical features derived from character patterns, word lengths, and morphological affixes. Words with shared roots or letters cluster near each other.';
    }
    if (explainBox) {
      explainBox.className = 'explain-card mode-spelling';
    }
    if (exKingQueen) exKingQueen.textContent = 'Distant: Disjoint spelling patterns (k-i-n-g vs q-u-e-e-n).';
    if (exCarVehicle) exCarVehicle.textContent = 'Distant: Different initial letters, lengths, and character sets.';
    if (exAppleBanana) exAppleBanana.textContent = 'Separated: Unique prefixes and vowel sequences.';
  } else {
    if (modeTitle) modeTitle.textContent = 'Active Mode: Meaning-Based (Semantic) Embeddings';
    if (modeDesc) {
      modeDesc.textContent =
        'The model represents words using numerical features that capture semantic relationships and conceptual categories. Words with related meanings cluster closely regardless of spelling.';
    }
    if (explainBox) {
      explainBox.className = 'explain-card mode-semantic';
    }
    if (exKingQueen) exKingQueen.textContent = 'Clustered: Share royal authority & leadership axes; separated only by gender.';
    if (exCarVehicle) exCarVehicle.textContent = 'Clustered: High affinity along transit & vehicle axes.';
    if (exAppleBanana) exAppleBanana.textContent = 'Clustered: Grouped in the food, nutrition, and fruit cluster.';
  }

  renderAnalogyResult();
}

function renderAnalogyResult() {
  const resultContainer = document.getElementById('analogy-result-container');
  if (!resultContainer) return;

  const getEmb = (w) => state.embeddings.find((e) => e.word.toLowerCase() === w.toLowerCase());
  const embA = getEmb(state.analogyA);
  const embB = getEmb(state.analogyB);
  const embC = getEmb(state.analogyC);

  if (!embA || !embB || !embC) {
    resultContainer.innerHTML = `
      <div class="analogy-placeholder">
        Enter words present in the active vocabulary to test vector arithmetic (e.g. king - man + woman).
      </div>
    `;
    return;
  }

  const resultVec = vectorArithmetic(embA.vector, embB.vector, embC.vector);
  const rankings = state.embeddings
    .filter((e) => ![state.analogyA, state.analogyB, state.analogyC].includes(e.word.toLowerCase()))
    .map((e) => ({
      word: e.word,
      similarity: cosineSimilarity(resultVec, e.vector),
    }))
    .sort((a, b) => b.similarity - a.similarity)
    .slice(0, 3);

  if (rankings.length === 0) {
    resultContainer.innerHTML = '<div class="analogy-placeholder">No other words available to evaluate.</div>';
    return;
  }

  let html = `
    <div class="analogy-winner-row">
      <span class="winner-label">Closest Vector in Active Vocabulary:</span>
      <div class="winner-chips">
  `;

  rankings.forEach((r, idx) => {
    html += `
      <button class="analogy-chip ${idx === 0 ? 'top' : ''}" data-word="${r.word}">
        <strong>${r.word}</strong> (${r.similarity.toFixed(3)})
      </button>
    `;
  });

  html += '</div></div>';
  resultContainer.innerHTML = html;

  resultContainer.querySelectorAll('.analogy-chip').forEach((btn) => {
    btn.addEventListener('click', (e) => {
      const w = e.currentTarget.getAttribute('data-word');
      if (w) selectWord(w);
    });
  });
}

// -------------------------------------------------------------
// Event Listeners Binding
// -------------------------------------------------------------
function bindEvents() {
  // Preset buttons in header
  document.querySelectorAll('.preset-btn').forEach((btn) => {
    btn.addEventListener('click', (e) => {
      const presetId = e.currentTarget.getAttribute('data-preset');
      const found = PRESETS.find((p) => p.id === presetId);
      if (found) loadText(found.text);
    });
  });

  // Global reset button
  const resetBtn = document.getElementById('global-reset-btn');
  if (resetBtn) {
    resetBtn.addEventListener('click', () => {
      loadText(PRESETS[0].text);
    });
  }

  // Live input stats on typing
  const vocabInput = document.getElementById('vocab-input');
  if (vocabInput) {
    vocabInput.addEventListener('input', (e) => {
      updateLiveStats(e.target.value);
    });
  }

  // Build Vocabulary button
  const buildBtn = document.getElementById('build-vocab-btn');
  if (buildBtn) {
    buildBtn.addEventListener('click', () => {
      const text = vocabInput ? vocabInput.value : '';
      if (text.trim()) executeBuildVocabulary(text);
    });
  }

  // Clear text button
  const clearBtn = document.getElementById('clear-vocab-btn');
  if (clearBtn) {
    clearBtn.addEventListener('click', () => {
      if (vocabInput) vocabInput.value = '';
      updateLiveStats('');
    });
  }

  // Mode cards
  const cardSpelling = document.getElementById('card-mode-spelling');
  const cardSemantic = document.getElementById('card-mode-semantic');
  if (cardSpelling) {
    cardSpelling.addEventListener('click', () => setEmbeddingMode('spelling'));
  }
  if (cardSemantic) {
    cardSemantic.addEventListener('click', () => setEmbeddingMode('semantic'));
  }

  // Generate embeddings button
  const genEmbedBtn = document.getElementById('gen-embeddings-btn');
  const embedInput = document.getElementById('embedding-input');
  if (genEmbedBtn) {
    genEmbedBtn.addEventListener('click', () => {
      const text = embedInput ? embedInput.value : '';
      if (text.trim()) executeBuildVocabulary(text);
    });
  }

  // Sync with vocabulary text
  const syncBtn = document.getElementById('sync-vocab-btn');
  if (syncBtn && vocabInput && embedInput) {
    syncBtn.addEventListener('click', () => {
      embedInput.value = vocabInput.value;
      executeBuildVocabulary(vocabInput.value);
    });
  }

  // Vocabulary search
  const vocabSearchInput = document.getElementById('vocab-search-input');
  if (vocabSearchInput) {
    vocabSearchInput.addEventListener('input', (e) => {
      state.vocabSearch = e.target.value;
      renderVocabularyTable();
    });
  }

  // Vocabulary table column sorting
  document.querySelectorAll('.sortable-th').forEach((th) => {
    th.addEventListener('click', (e) => {
      const field = e.currentTarget.getAttribute('data-sort');
      if (state.vocabSortBy === field) {
        state.vocabSortOrder = state.vocabSortOrder === 'asc' ? 'desc' : 'asc';
      } else {
        state.vocabSortBy = field;
        state.vocabSortOrder = field === 'count' ? 'desc' : 'asc';
      }
      renderVocabularyTable();
    });
  });

  // Embedding table search
  const embedSearchInput = document.getElementById('embedding-search-input');
  if (embedSearchInput) {
    embedSearchInput.addEventListener('input', (e) => {
      state.embeddingSearch = e.target.value;
      renderEmbeddingTable();
    });
  }

  // Copy vector button
  const copyVecBtn = document.getElementById('copy-vector-btn');
  if (copyVecBtn) {
    copyVecBtn.addEventListener('click', () => {
      if (!state.selectedWord) return;
      const emb = state.embeddings.find((e) => e.word.toLowerCase() === state.selectedWord.toLowerCase());
      if (emb) {
        navigator.clipboard.writeText(JSON.stringify(emb.vector, null, 2));
        copyVecBtn.textContent = 'Copied!';
        setTimeout(() => {
          copyVecBtn.textContent = 'Copy Vector';
        }, 1500);
      }
    });
  }

  // Pairwise compare dropdown
  const compSelect = document.getElementById('compare-word-select');
  if (compSelect) {
    compSelect.addEventListener('change', (e) => {
      state.compareTargetWord = e.target.value;
      renderComparisonResult();
    });
  }

  // Vector analogy inputs
  const inputA = document.getElementById('analogy-input-a');
  const inputB = document.getElementById('analogy-input-b');
  const inputC = document.getElementById('analogy-input-c');

  if (inputA) {
    inputA.addEventListener('input', (e) => {
      state.analogyA = e.target.value.trim().toLowerCase();
      renderAnalogyResult();
    });
  }
  if (inputB) {
    inputB.addEventListener('input', (e) => {
      state.analogyB = e.target.value.trim().toLowerCase();
      renderAnalogyResult();
    });
  }
  if (inputC) {
    inputC.addEventListener('input', (e) => {
      state.analogyC = e.target.value.trim().toLowerCase();
      renderAnalogyResult();
    });
  }

  // 3D Visualization Toolbar Buttons
  const btnReset = document.getElementById('btn-viz-reset');
  const btnZoomIn = document.getElementById('btn-viz-zoomin');
  const btnZoomOut = document.getElementById('btn-viz-zoomout');
  const btnLabels = document.getElementById('btn-viz-labels');
  const btnLines = document.getElementById('btn-viz-lines');
  const btnRotate = document.getElementById('btn-viz-rotate');
  const btnGrid = document.getElementById('btn-viz-grid');
  const btnFullscreen = document.getElementById('btn-viz-fullscreen');

  if (btnReset && visualizer) {
    btnReset.addEventListener('click', () => visualizer.resetCamera());
  }
  if (btnZoomIn && visualizer) {
    btnZoomIn.addEventListener('click', () => visualizer.zoomIn());
  }
  if (btnZoomOut && visualizer) {
    btnZoomOut.addEventListener('click', () => visualizer.zoomOut());
  }
  if (btnLabels && visualizer) {
    btnLabels.addEventListener('click', () => {
      const active = visualizer.toggleLabels();
      btnLabels.classList.toggle('active', active);
    });
  }
  if (btnLines && visualizer) {
    btnLines.addEventListener('click', () => {
      const active = visualizer.toggleLines();
      btnLines.classList.toggle('active', active);
    });
  }
  if (btnRotate && visualizer) {
    btnRotate.addEventListener('click', () => {
      const active = visualizer.toggleAutoRotate();
      btnRotate.classList.toggle('active', active);
    });
  }
  if (btnGrid && visualizer) {
    btnGrid.addEventListener('click', () => {
      const active = visualizer.toggleGrid();
      btnGrid.classList.toggle('active', active);
    });
  }
  if (btnFullscreen) {
    btnFullscreen.addEventListener('click', () => {
      const c = document.getElementById('viz-3d-container');
      if (!c) return;
      if (!document.fullscreenElement) {
        c.requestFullscreen().catch((err) => console.error(err));
      } else {
        document.exitFullscreen().catch((err) => console.error(err));
      }
    });
  }
}
