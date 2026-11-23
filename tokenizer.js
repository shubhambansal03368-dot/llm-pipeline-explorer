/**
 * LLM Pipeline Explorer - Tokenizer Module
 * Pure Vanilla JavaScript (ES6 Module)
 *
 * Implements clean word tokenization, case normalization,
 * punctuation stripping, sentence segmentation, and live statistics.
 */

/**
 * Extracts normalized word tokens from raw text
 * @param {string} text - Raw input string
 * @param {Object} options - Options for case and punctuation
 * @returns {string[]} Array of clean token strings
 */
export function extractTokens(text, options = { toLowerCase: true }) {
  if (!text || typeof text !== 'string') return [];

  let processed = text;
  if (options.toLowerCase) {
    processed = processed.toLowerCase();
  }

  // Match words, including alphanumeric tokens and contractions (e.g., don't, we'll)
  // Strips standalone punctuation: periods, commas, colons, semicolons, brackets, quotes
  const matches = processed.match(/[a-zA-Z0-9]+(?:'[a-zA-Z0-9]+)?/g);
  return matches || [];
}

/**
 * Counts sentences in raw text based on sentence-ending punctuation
 * @param {string} text
 * @returns {number}
 */
export function countSentences(text) {
  if (!text || !text.trim()) return 0;
  const segments = text.trim().split(/[.!?]+/).filter((s) => s.trim().length > 0);
  return segments.length;
}

/**
 * Computes real-time statistics for text input area
 * @param {string} text
 * @returns {Object} Live counts
 */
export function getLiveInputStats(text) {
  const charCount = text ? text.length : 0;
  const rawWords = text && text.trim() ? text.trim().split(/\s+/).filter(Boolean) : [];
  const tokens = extractTokens(text);
  const uniqueTokens = new Set(tokens);
  const sentenceCount = countSentences(text);

  return {
    characterCount: charCount,
    rawWordCount: rawWords.length,
    totalTokenCount: tokens.length,
    uniqueTokenCount: uniqueTokens.size,
    sentenceCount,
  };
}
