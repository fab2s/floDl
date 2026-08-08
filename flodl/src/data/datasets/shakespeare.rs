//! Shakespeare character-level language modeling dataset.
//!
//! Tokenizes text into character indices and creates input/target pairs
//! for next-character prediction (shifted by one position).
//!
//! # Example
//!
//! ```ignore
//! let text = std::fs::read_to_string("input.txt")?;
//! let data = Shakespeare::parse(&text, 128)?;
//! // data.data:    [N, 128] Int64 -- input sequences
//! // data.targets: [N, 128] Int64 -- targets (shifted by 1)
//! // data.vocab_size: ~65 unique characters
//! ```

use crate::data::BatchDataSet;
use crate::tensor::{Device, Result, Tensor, TensorError};

/// Parsed Shakespeare character-level dataset.
pub struct Shakespeare {
    /// Input sequences as `[N, seq_len]` Int64 (character indices).
    pub data: Tensor,
    /// Target sequences as `[N, seq_len]` Int64 (shifted by 1).
    pub targets: Tensor,
    /// Number of unique characters in the vocabulary.
    pub vocab_size: usize,
    /// Character-to-index mapping (sorted by char value).
    pub char_to_idx: Vec<(char, usize)>,
    /// Index-to-character mapping.
    pub idx_to_char: Vec<char>,
}

impl Shakespeare {
    /// Parse raw text into character-level sequences.
    ///
    /// Creates non-overlapping windows of `seq_len` characters.
    /// Input is `text[i..i+seq_len]`, target is `text[i+1..i+seq_len+1]`.
    pub fn parse(text: &str, seq_len: usize) -> Result<Self> {
        if text.len() < seq_len + 1 {
            return Err(TensorError::new(&format!(
                "Shakespeare: text length {} too short for seq_len {}",
                text.len(),
                seq_len
            )));
        }

        // Build vocabulary from sorted unique characters
        let chars: Vec<char> = text.chars().collect();
        let mut vocab: Vec<char> = chars.clone();
        vocab.sort();
        vocab.dedup();
        let vocab_size = vocab.len();

        // Build lookup table (char -> index)
        let mut char_to_idx = Vec::with_capacity(vocab_size);
        let mut lookup = [0usize; 256]; // ASCII fast path
        for (idx, &ch) in vocab.iter().enumerate() {
            char_to_idx.push((ch, idx));
            if (ch as u32) < 256 {
                lookup[ch as usize] = idx;
            }
        }

        // Encode entire text to indices
        let encoded: Vec<i64> = chars
            .iter()
            .map(|&ch| {
                if (ch as u32) < 256 {
                    lookup[ch as usize] as i64
                } else {
                    // Fallback for non-ASCII (shouldn't happen in Shakespeare)
                    char_to_idx
                        .iter()
                        .find(|(c, _)| *c == ch)
                        .map(|(_, i)| *i as i64)
                        .unwrap_or(0)
                }
            })
            .collect();

        // Create non-overlapping sequences
        let n_sequences = (encoded.len() - 1) / seq_len;
        if n_sequences == 0 {
            return Err(TensorError::new(
                "Shakespeare: not enough text for even one sequence",
            ));
        }

        let mut input_data = Vec::with_capacity(n_sequences * seq_len);
        let mut target_data = Vec::with_capacity(n_sequences * seq_len);

        for i in 0..n_sequences {
            let start = i * seq_len;
            input_data.extend_from_slice(&encoded[start..start + seq_len]);
            target_data.extend_from_slice(&encoded[start + 1..start + seq_len + 1]);
        }

        let data = Tensor::from_i64(
            &input_data,
            &[n_sequences as i64, seq_len as i64],
            Device::CPU,
        )?;
        let targets = Tensor::from_i64(
            &target_data,
            &[n_sequences as i64, seq_len as i64],
            Device::CPU,
        )?;

        Ok(Shakespeare {
            data,
            targets,
            vocab_size,
            char_to_idx,
            idx_to_char: vocab,
        })
    }

    /// Number of sequences.
    pub fn len(&self) -> usize {
        self.data.shape()[0] as usize
    }

    /// True if the dataset is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Decode a sequence of indices back to a string.
    pub fn decode(&self, indices: &[i64]) -> String {
        indices
            .iter()
            .map(|&i| self.idx_to_char.get(i as usize).copied().unwrap_or('?'))
            .collect()
    }
}

impl BatchDataSet for Shakespeare {
    fn len(&self) -> usize {
        self.data.shape()[0] as usize
    }

    fn get_batch(&self, indices: &[usize]) -> Result<Vec<Tensor>> {
        let idx: Vec<i64> = indices.iter().map(|&i| (i % self.len()) as i64).collect();
        let idx_tensor = Tensor::from_i64(&idx, &[idx.len() as i64], Device::CPU)?;
        let data = self.data.index_select(0, &idx_tensor)?;
        let targets = self.targets.index_select(0, &idx_tensor)?;
        Ok(vec![data, targets])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_simple_text() {
        let text = "abcdefghijklmnop"; // 16 chars
        let data = Shakespeare::parse(text, 4).unwrap();

        // 15 usable chars (need +1 for target), 15/4 = 3 sequences
        assert_eq!(data.data.shape(), &[3, 4]);
        assert_eq!(data.targets.shape(), &[3, 4]);
        assert!(data.vocab_size <= 16);
    }

    #[test]
    fn target_is_shifted_by_one() {
        let text = "abcdefghij"; // 10 chars
        let data = Shakespeare::parse(text, 3).unwrap();

        // Sequence 0: input="abc", target="bcd"
        let _input_0 = data
            .data
            .select(0, 0)
            .unwrap()
            .select(0, 0)
            .unwrap()
            .to_i64_vec()
            .unwrap()[0];
        let target_0 = data
            .targets
            .select(0, 0)
            .unwrap()
            .select(0, 0)
            .unwrap()
            .to_i64_vec()
            .unwrap()[0];

        // target[0] should equal input[1] (both are 'b')
        let input_1 = data
            .data
            .select(0, 0)
            .unwrap()
            .select(0, 1)
            .unwrap()
            .to_i64_vec()
            .unwrap()[0];
        assert_eq!(target_0, input_1);
    }

    #[test]
    fn vocab_is_sorted() {
        let text = "zyxwvutsrqponmlkjihgfedcba";
        let data = Shakespeare::parse(text, 3).unwrap();

        // idx_to_char should be sorted
        for i in 1..data.idx_to_char.len() {
            assert!(data.idx_to_char[i] > data.idx_to_char[i - 1]);
        }
    }

    #[test]
    fn decode_roundtrip() {
        let text = "hello world";
        let data = Shakespeare::parse(text, 4).unwrap();

        // Encode then decode first sequence
        let seq: Vec<i64> = (0..4)
            .map(|j| {
                data.data
                    .select(0, 0)
                    .unwrap()
                    .select(0, j)
                    .unwrap()
                    .to_i64_vec()
                    .unwrap()[0]
            })
            .collect();
        let decoded = data.decode(&seq);
        assert_eq!(decoded, "hell");
    }

    #[test]
    fn text_too_short() {
        assert!(Shakespeare::parse("ab", 5).is_err());
    }
}
