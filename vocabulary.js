/**
 * LLM Pipeline Explorer - Vocabulary Module
 * Pure Vanilla JavaScript (ES6 Module)
 *
 * Builds a deterministic, 0-indexed word-level vocabulary from tokens,
 * preserves first-occurrence order, computes word frequencies,
 * and maintains the numerical sequence of token IDs.
 */

import { extractTokens, countSentences } from './tokenizer.js';

/**
 * Builds vocabulary and numerical token ID sequence from raw text
 * @param {string} text - Raw input text
 * @returns {Object} Vocabulary result containing tokens list, maps, and stats
 */
export function buildVocabulary(text) {
  const rawTokens = extractTokens(text);
  const tokenMap = new Map();
  const tokenToId = new Map();
  const idToToken = new Map();
  const tokenIds = [];

  let nextId = 0;

  for (let idx = 0; idx < rawTokens.length; idx++) {
    const word = rawTokens[idx];

    if (!tokenMap.has(word)) {
      const assignedId = nextId++;
      const item = {
        id: assignedId,
        token: word,
        count: 1,
        firstIndex: idx,
        length: word.length,
        positions: [idx],
      };
      tokenMap.set(word, item);
      tokenToId.set(word, assignedId);
      idToToken.set(assignedId, word);
      tokenIds.push(assignedId);
    } else {
      const existing = tokenMap.get(word);
      existing.count += 1;
      existing.positions.push(idx);
      tokenIds.push(existing.id);
    }
  }

  // Preserve first-occurrence order (ordered by assigned ID 0, 1, 2, ...)
  const tokens = Array.from(tokenMap.values()).sort((a, b) => a.id - b.id);

  return {
    tokens,
    tokenToId,
    idToToken,
    totalWordCount: rawTokens.length,
    uniqueWordCount: tokens.length,
    characterCount: text.length,
    sentenceCount: countSentences(text),
    rawTokens,
    tokenIds,
  };
}

/**
 * Filters and sorts vocabulary items for display
 * @param {Array} tokens - List of token items
 * @param {string} query - Search string
 * @param {'id' | 'token' | 'count'} sortBy - Column key to sort by
 * @param {'asc' | 'desc'} sortOrder - Sort direction
 * @returns {Array} Filtered and sorted token items
 */
export function filterAndSortVocabulary(tokens, query = '', sortBy = 'id', sortOrder = 'asc') {
  let filtered = [...tokens];

  if (query && query.trim()) {
    const q = query.trim().toLowerCase();
    filtered = filtered.filter(
      (item) => item.token.toLowerCase().includes(q) || item.id.toString() === q
    );
  }

  filtered.sort((a, b) => {
    let comparison = 0;
    if (sortBy === 'id') {
      comparison = a.id - b.id;
    } else if (sortBy === 'count') {
      comparison = a.count - b.count;
    } else if (sortBy === 'token') {
      comparison = a.token.localeCompare(b.token);
    }

    return sortOrder === 'asc' ? comparison : -comparison;
  });

  return filtered;
}
