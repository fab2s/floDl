//! Wrappers over the HuggingFace `tokenizers` crate for flodl pipelines.
//!
//! The main type is [`HfTokenizer`] — a thin façade over
//! [`tokenizers::Tokenizer`] that loads from a local `tokenizer.json` file
//! or (with the `hub` feature) from a HuggingFace Hub repo and produces
//! flodl [`Variable`]s ready to feed into a transformer graph.
//!
//! `HfTokenizer` is model-agnostic: the same wrapper serves BERT, GPT2,
//! LLaMA, etc. — the loaded `tokenizer.json` carries the model-specific
//! pre-tokenizer and post-processor. For BERT in particular, the raw
//! `[B, S]` attention mask this wrapper emits still needs to be converted
//! to the additive form via
//! [`crate::models::bert::build_extended_attention_mask`] before it can be
//! fed to `BertModel`'s graph.
//!
//! ## Example
//!
//! ```ignore
//! use flodl::DType;
//! use flodl_hf::tokenizer::HfTokenizer;
//! use flodl_hf::models::bert::{BertModel, build_extended_attention_mask};
//!
//! let tok = HfTokenizer::from_pretrained("google-bert/bert-base-uncased")?;
//! let enc = tok.encode(&["hello world"])?;
//!
//! let graph = BertModel::from_pretrained("google-bert/bert-base-uncased")?;
//! let mask_f32 = enc.attention_mask.data().to_dtype(DType::Float32)?;
//! let mask = build_extended_attention_mask(&mask_f32)?;
//! // ... feed enc.input_ids / enc.position_ids / enc.token_type_ids / mask
//! //     into graph.forward_multi(...)
//! ```

use std::path::Path;

use tokenizers::{EncodeInput, PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer};

use flodl::{Device, Result, Tensor, TensorError, Variable};

/// Wrapper around a HuggingFace [`tokenizers::Tokenizer`] that emits flodl
/// tensors.
pub struct HfTokenizer {
    inner: Tokenizer,
}

/// Per-batch tokenization output. All tensors are `i64 [batch, seq]`.
///
/// `attention_mask` is the raw 0/1 form the tokenizer produces; for BERT,
/// convert to the additive form with
/// [`crate::models::bert::build_extended_attention_mask`] before
/// `forward_multi`.
///
/// `position_ids` is `0..seq` broadcast across the batch — matches the
/// defaults `BertEmbeddings` expects and is pinned in the parity fixture.
/// RoBERTa-family models compute position ids internally from `input_ids`
/// using `pad_token_id`, so this field is ignored there.
///
/// `sequence_ids` is the per-token segment tag from the HF tokenizer:
/// `0` = first sequence (e.g. question in a QA pair), `1` = second
/// sequence (e.g. context), `-1` = special token or padding (the
/// tokenizer returned `None`). This is the canonical signal used by HF
/// QA pipelines to locate the context span — it's model-agnostic,
/// whereas `token_type_ids` is a *model input* whose semantics vary
/// (BERT sets segment B to 1; RoBERTa keeps everything at 0).
#[derive(Debug)]
pub struct EncodedBatch {
    pub input_ids: Variable,
    pub attention_mask: Variable,
    pub token_type_ids: Variable,
    pub position_ids: Variable,
    pub sequence_ids: Variable,
}

impl HfTokenizer {
    /// Load a tokenizer from a local `tokenizer.json` file.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let tok = Tokenizer::from_file(path.as_ref())
            .map_err(|e| TensorError::new(&format!("tokenizer load: {e}")))?;
        Ok(Self::from_inner(tok))
    }

    /// Wrap a pre-built [`Tokenizer`]. Installs batch-longest padding with a
    /// `[PAD]`-derived `pad_id` if the tokenizer doesn't already carry a
    /// padding configuration. Truncation is left to callers — most BERT
    /// inputs fit under the 512-token limit, and silently truncating is
    /// worse than a loud shape error for out-of-range texts.
    pub fn from_inner(mut inner: Tokenizer) -> Self {
        if inner.get_padding().is_none() {
            let pad_id = inner.token_to_id("[PAD]").unwrap_or(0);
            inner.with_padding(Some(PaddingParams {
                strategy: PaddingStrategy::BatchLongest,
                direction: PaddingDirection::Right,
                pad_to_multiple_of: None,
                pad_id,
                pad_type_id: 0,
                pad_token: "[PAD]".to_string(),
            }));
        }
        Self { inner }
    }

    /// Borrow the underlying [`tokenizers::Tokenizer`] for advanced
    /// configuration (custom padding, truncation, normalization, etc.).
    pub fn inner(&self) -> &Tokenizer {
        &self.inner
    }

    /// Serialize the tokenizer to a `tokenizer.json` file at `path`.
    ///
    /// Pair with [`Self::from_pretrained`] / [`Self::from_file`] to
    /// persist a fine-tuned model's tokenizer next to its `.fdl`
    /// checkpoint so `fdl flodl-hf export --checkpoint` picks it up
    /// via the auto-tokenizer-copy whitelist. The output is the same
    /// JSON form HuggingFace's fast-tokenizer ecosystem (`tokenizers`,
    /// HF Python's `AutoTokenizer.from_pretrained`) reads back.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        self.inner.save(path.as_ref(), true).map_err(|e| {
            TensorError::new(&format!(
                "tokenizer save to {}: {e}",
                path.as_ref().display(),
            ))
        })
    }

    /// Encode a batch of texts into [`EncodedBatch`] on CPU.
    pub fn encode(&self, texts: &[&str]) -> Result<EncodedBatch> {
        self.encode_on_device(texts, Device::CPU)
    }

    /// Encode a batch of texts into [`EncodedBatch`] on `device`.
    pub fn encode_on_device(&self, texts: &[&str], device: Device) -> Result<EncodedBatch> {
        if texts.is_empty() {
            return Err(TensorError::new("tokenize: empty batch"));
        }
        let inputs: Vec<EncodeInput> = texts.iter().map(|s| (*s).into()).collect();
        let encodings = self
            .inner
            .encode_batch(inputs, true)
            .map_err(|e| TensorError::new(&format!("tokenize: {e}")))?;

        let batch = encodings.len() as i64;
        let seq = encodings[0].get_ids().len() as i64;

        let cap = (batch * seq) as usize;
        let mut input_ids = Vec::<i64>::with_capacity(cap);
        let mut attention_mask = Vec::<i64>::with_capacity(cap);
        let mut token_type_ids = Vec::<i64>::with_capacity(cap);
        let mut sequence_ids = Vec::<i64>::with_capacity(cap);

        for enc in &encodings {
            // BatchLongest padding guarantees all encodings share seq length.
            debug_assert_eq!(enc.get_ids().len() as i64, seq);
            input_ids.extend(enc.get_ids().iter().map(|&x| x as i64));
            attention_mask.extend(enc.get_attention_mask().iter().map(|&x| x as i64));
            token_type_ids.extend(enc.get_type_ids().iter().map(|&x| x as i64));
            sequence_ids.extend(
                enc.get_sequence_ids()
                    .iter()
                    .map(|opt| opt.map(|v| v as i64).unwrap_or(-1)),
            );
        }

        let mut position_ids = Vec::<i64>::with_capacity(cap);
        for _ in 0..batch {
            position_ids.extend(0i64..seq);
        }

        let shape = [batch, seq];
        Ok(EncodedBatch {
            input_ids: Variable::new(Tensor::from_i64(&input_ids, &shape, device)?, false),
            attention_mask: Variable::new(
                Tensor::from_i64(&attention_mask, &shape, device)?,
                false,
            ),
            token_type_ids: Variable::new(
                Tensor::from_i64(&token_type_ids, &shape, device)?,
                false,
            ),
            position_ids: Variable::new(Tensor::from_i64(&position_ids, &shape, device)?, false),
            sequence_ids: Variable::new(Tensor::from_i64(&sequence_ids, &shape, device)?, false),
        })
    }

    /// Encode a batch of `(text_a, text_b)` pairs on CPU.
    ///
    /// The resulting `token_type_ids` mark segment B (e.g. the QA
    /// context, or the second sentence in an NLI pair) with `1`, as HF
    /// tokenizers do. Question-answering and pair-classification
    /// pipelines consume this directly.
    pub fn encode_pairs(&self, pairs: &[(&str, &str)]) -> Result<EncodedBatch> {
        self.encode_pairs_on_device(pairs, Device::CPU)
    }

    /// Device-aware variant of [`encode_pairs`](Self::encode_pairs).
    pub fn encode_pairs_on_device(
        &self,
        pairs: &[(&str, &str)],
        device: Device,
    ) -> Result<EncodedBatch> {
        if pairs.is_empty() {
            return Err(TensorError::new("tokenize pairs: empty batch"));
        }
        let inputs: Vec<EncodeInput> = pairs
            .iter()
            .map(|(a, b)| EncodeInput::Dual((*a).into(), (*b).into()))
            .collect();
        let encodings = self
            .inner
            .encode_batch(inputs, true)
            .map_err(|e| TensorError::new(&format!("tokenize pairs: {e}")))?;

        let batch = encodings.len() as i64;
        let seq = encodings[0].get_ids().len() as i64;
        let cap = (batch * seq) as usize;
        let mut input_ids = Vec::<i64>::with_capacity(cap);
        let mut attention_mask = Vec::<i64>::with_capacity(cap);
        let mut token_type_ids = Vec::<i64>::with_capacity(cap);
        let mut sequence_ids = Vec::<i64>::with_capacity(cap);

        for enc in &encodings {
            debug_assert_eq!(enc.get_ids().len() as i64, seq);
            input_ids.extend(enc.get_ids().iter().map(|&x| x as i64));
            attention_mask.extend(enc.get_attention_mask().iter().map(|&x| x as i64));
            token_type_ids.extend(enc.get_type_ids().iter().map(|&x| x as i64));
            sequence_ids.extend(
                enc.get_sequence_ids()
                    .iter()
                    .map(|opt| opt.map(|v| v as i64).unwrap_or(-1)),
            );
        }
        let mut position_ids = Vec::<i64>::with_capacity(cap);
        for _ in 0..batch {
            position_ids.extend(0i64..seq);
        }

        let shape = [batch, seq];
        Ok(EncodedBatch {
            input_ids: Variable::new(Tensor::from_i64(&input_ids, &shape, device)?, false),
            attention_mask: Variable::new(
                Tensor::from_i64(&attention_mask, &shape, device)?,
                false,
            ),
            token_type_ids: Variable::new(
                Tensor::from_i64(&token_type_ids, &shape, device)?,
                false,
            ),
            position_ids: Variable::new(Tensor::from_i64(&position_ids, &shape, device)?, false),
            sequence_ids: Variable::new(Tensor::from_i64(&sequence_ids, &shape, device)?, false),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_batch_errors() {
        // Build a minimal tokenizer from the simplest possible BPE model so
        // the test doesn't need network. The texts[] check fires before any
        // actual encoding, so the model's contents don't matter.
        use tokenizers::models::bpe::BPE;
        let bpe = BPE::default();
        let tok = Tokenizer::new(bpe);
        let hf = HfTokenizer::from_inner(tok);

        let err = hf.encode(&[]).unwrap_err();
        assert!(format!("{err}").contains("empty batch"));
    }

    #[test]
    fn save_and_reload_round_trips() {
        // Use a WordLevel tokenizer with a fixed vocabulary so the
        // re-loaded copy can encode the same string and produce the
        // same ids. BPE::default() has no merges/vocab, so any
        // non-empty string would be UNK or rejected.
        use std::collections::HashMap;
        use std::process;
        use tokenizers::models::wordlevel::WordLevel;
        use tokenizers::pre_tokenizers::whitespace::Whitespace;

        let mut vocab: HashMap<String, u32> = HashMap::new();
        for (i, w) in ["[UNK]", "hello", "world", "rust"].iter().enumerate() {
            vocab.insert((*w).to_string(), i as u32);
        }
        let wl = WordLevel::builder()
            .vocab(vocab.into_iter().collect())
            .unk_token("[UNK]".into())
            .build()
            .expect("WordLevel build");
        let mut inner = Tokenizer::new(wl);
        inner.with_pre_tokenizer(Some(Whitespace {}));
        let hf = HfTokenizer::from_inner(inner);

        // Same temp-dir convention as flodl-hf/src/export.rs
        // (`unique_tempdir`): temp_dir + pid + tag, no dev-dep.
        let dir = std::env::temp_dir().join(format!("flodl_hf_tokenizer_save_{}", process::id()));
        std::fs::create_dir_all(&dir).expect("create_dir_all");
        let path = dir.join("tokenizer.json");
        hf.save(&path).expect("save");
        assert!(path.is_file(), "tokenizer.json was not written");

        let reloaded = HfTokenizer::from_file(&path).expect("from_file");
        let original_ids = hf.encode(&["hello world"]).unwrap();
        let reloaded_ids = reloaded.encode(&["hello world"]).unwrap();
        assert_eq!(
            original_ids.input_ids.data().to_i64_vec().unwrap(),
            reloaded_ids.input_ids.data().to_i64_vec().unwrap(),
            "round-trip changed input_ids",
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
