//! Shared task-head types and helpers.
//!
//! [`Answer`] and [`TokenPrediction`] are the public output types for
//! question-answering and token-classification heads across every
//! model family (BERT, RoBERTa, DistilBERT, …). Centralising them here
//! means callers can match on one type regardless of which model
//! produced the prediction, and the three families don't each define
//! their own copy.
//!
//! The module-internal helpers (`default_labels`, `check_num_labels`,
//! `logits_to_sorted_labels`, `extract_best_span`) carry the behaviour
//! that's identical across families — label fallback naming, positive-
//! `num_labels` guard, softmax-and-sort for sequence classification,
//! and the QA span-search algorithm.

use flodl::{cross_entropy_loss, Buffer, Graph, HasGraph, Module, Parameter, Result, TensorError, Variable};

/// One extracted answer span from a
/// `*ForQuestionAnswering::answer` / `::answer_batch` call.
#[derive(Debug, Clone)]
pub struct Answer {
    /// Decoded answer text (special tokens stripped).
    pub text: String,
    /// Start token index in the input sequence.
    pub start: usize,
    /// End token index in the input sequence (inclusive).
    pub end: usize,
    /// `start_logit + end_logit` at the chosen span. Unnormalised — use
    /// for comparing spans within one `answer` call, not across calls.
    pub score: f32,
}

/// One labelled token inside a `*ForTokenClassification` prediction.
#[cfg(feature = "tokenizer")]
#[derive(Debug, Clone)]
pub struct TokenPrediction {
    /// The subword string the tokenizer produced for this position.
    pub token: String,
    /// The highest-scoring label for this token.
    pub label: String,
    /// Softmax probability of `label`.
    pub score: f32,
    /// `true` for real tokens, `false` for padding (`attention_mask ==
    /// 0`). Lets callers drop padding without re-tokenising.
    pub attends: bool,
}

/// Default label names when a checkpoint ships without `id2label`.
/// Mirrors HF Python's `LABEL_0`, `LABEL_1`, ...
pub(crate) fn default_labels(n: i64) -> Vec<String> {
    (0..n).map(|i| format!("LABEL_{i}")).collect()
}

/// Clamp `num_labels` to a positive value or surface a loud error.
/// Used by every task-head `on_device` constructor so the error
/// message is consistent across families.
pub(crate) fn check_num_labels(n: i64) -> Result<i64> {
    if n <= 0 {
        return Err(TensorError::new(&format!(
            "num_labels must be > 0, got {n}",
        )));
    }
    Ok(n)
}

/// Borrow the attached tokenizer or surface a consistent
/// "missing tokenizer" error. `method` is the fully-qualified caller,
/// e.g. `"BertForSequenceClassification::predict"`; it's spliced into
/// the returned error so users know which call site needs the
/// tokenizer.
#[cfg(feature = "tokenizer")]
pub(crate) fn require_tokenizer<'a>(
    tokenizer: Option<&'a crate::tokenizer::HfTokenizer>,
    method: &str,
) -> Result<&'a crate::tokenizer::HfTokenizer> {
    tokenizer.ok_or_else(|| {
        TensorError::new(&format!(
            "{method} requires a tokenizer; \
             use from_pretrained or .with_tokenizer(...) first",
        ))
    })
}

/// Argmax over an `f32` slice using `partial_cmp`. NaNs compare as
/// `Ordering::Equal` so they do not poison the search; the caller must
/// already have validated that the slice is non-empty (all current
/// users enforce `num_labels > 0` up front via [`check_num_labels`]).
pub(crate) fn argmax_f32(slice: &[f32]) -> (usize, f32) {
    let (idx, &val) = slice
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .expect("argmax_f32 called on empty slice");
    (idx, val)
}

// ── Loss functions ───────────────────────────────────────────────────────

/// Compute sequence-classification loss.
///
/// `logits`: `[batch, num_labels]` - the raw output of a
/// `*ForSequenceClassification` head's classifier.
///
/// `labels`: `[batch]` Int64 class indices (hard labels) or
/// `[batch, num_labels]` Float soft labels. Auto-detected by libtorch
/// through [`cross_entropy_loss`].
///
/// Mirrors HF Python's `*ForSequenceClassification(..., labels=labels)`
/// loss computation (the `single_label_classification` branch). A
/// dedicated regression branch (`problem_type == "regression"`, MSE) is
/// not yet exposed - pass raw logits to [`flodl::mse_loss`] directly for
/// that case.
pub fn sequence_classification_loss(
    logits: &Variable,
    labels: &Variable,
) -> Result<Variable> {
    let shape = logits.shape();
    if shape.len() != 2 {
        return Err(TensorError::new(&format!(
            "sequence_classification_loss: logits must be [batch, num_labels], got {shape:?}",
        )));
    }
    cross_entropy_loss(logits, labels)
}

/// Compute token-classification loss.
///
/// `logits`: `[batch, seq_len, num_labels]` - the raw output of a
/// `*ForTokenClassification` head.
///
/// `labels`: `[batch, seq_len]` Int64 class indices. Use `-100` at
/// positions the loss should ignore: special tokens (`[CLS]`, `[SEP]`,
/// padding) and any non-first subword of a word under the standard
/// BIO-to-subword alignment rule. Matches HF Python's
/// `CrossEntropyLoss(ignore_index=-100)` default, which flodl's
/// [`cross_entropy_loss`] wires through natively.
pub fn token_classification_loss(
    logits: &Variable,
    labels: &Variable,
) -> Result<Variable> {
    let shape = logits.shape();
    if shape.len() != 3 {
        return Err(TensorError::new(&format!(
            "token_classification_loss: logits must be [batch, seq_len, num_labels], got {shape:?}",
        )));
    }
    let num_labels = shape[2];
    let flat_logits = logits.reshape(&[-1, num_labels])?;
    let flat_labels = labels.reshape(&[-1])?;
    cross_entropy_loss(&flat_logits, &flat_labels)
}

/// Compute masked-language-modelling loss.
///
/// `logits`: `[batch, seq_len, vocab_size]` — the raw output of a
/// `*ForMaskedLM` head's decoder.
///
/// `labels`: `[batch, seq_len]` Int64 token ids. Use `-100` at every
/// position the loss should **ignore** — unmasked tokens, padding, and
/// special tokens — and the original (pre-mask) token id at positions
/// where the model is being asked to predict. Matches HF Python's
/// `CrossEntropyLoss(ignore_index=-100)` default wiring in
/// `BertForMaskedLM`, which flodl's [`cross_entropy_loss`] honours
/// natively.
///
/// Flatten-then-CE: identical implementation to
/// [`token_classification_loss`], but exposed under the HF-canonical
/// name so callers of continued-pretraining / domain-adaptation paths
/// find it by the name they know.
pub fn masked_lm_loss(
    logits: &Variable,
    labels: &Variable,
) -> Result<Variable> {
    let shape = logits.shape();
    if shape.len() != 3 {
        return Err(TensorError::new(&format!(
            "masked_lm_loss: logits must be [batch, seq_len, vocab_size], got {shape:?}",
        )));
    }
    let vocab_size = shape[2];
    let flat_logits = logits.reshape(&[-1, vocab_size])?;
    let flat_labels = labels.reshape(&[-1])?;
    cross_entropy_loss(&flat_logits, &flat_labels)
}

/// Compute extractive question-answering loss.
///
/// `logits`: `[batch, seq_len, 2]` - the raw output of a
/// `*ForQuestionAnswering` head. Start logits are on slice `0` of the
/// last axis, end logits on slice `1`.
///
/// `start_positions`, `end_positions`: `[batch]` Int64 token indices of
/// the gold span's inclusive start and end. Both must lie in
/// `[0, seq_len)`; out-of-bounds positions are a caller error rather
/// than a silently-ignored training signal. (HF Python clamps then uses
/// `ignore_index == seq_len` to drop no-answer examples; flodl's
/// [`cross_entropy_loss`] fixes `ignore_index=-100`, so filter
/// no-answer examples upstream or assign them position `0` per your
/// dataset convention.)
///
/// Returns `(start_loss + end_loss) / 2`, matching HF Python's
/// `*ForQuestionAnswering(..., start_positions=..., end_positions=...)`.
pub fn question_answering_loss(
    logits: &Variable,
    start_positions: &Variable,
    end_positions: &Variable,
) -> Result<Variable> {
    let shape = logits.shape();
    if shape.len() != 3 || shape[2] != 2 {
        return Err(TensorError::new(&format!(
            "question_answering_loss: logits must be [batch, seq_len, 2], got {shape:?}",
        )));
    }
    let start_logits = logits.narrow(-1, 0, 1)?.squeeze(-1)?;
    let end_logits   = logits.narrow(-1, 1, 1)?.squeeze(-1)?;
    let start_loss = cross_entropy_loss(&start_logits, start_positions)?;
    let end_loss   = cross_entropy_loss(&end_logits, end_positions)?;
    start_loss.add(&end_loss)?.mul_scalar(0.5)
}

/// Apply softmax to a `[batch, num_labels]` logits tensor and return a
/// sorted `(label, score)` list per batch entry, descending by score.
pub(crate) fn logits_to_sorted_labels(
    logits: &Variable,
    id2label: &[String],
) -> Result<Vec<Vec<(String, f32)>>> {
    let probs = logits.softmax(-1)?;
    let shape = probs.shape();
    assert_eq!(shape.len(), 2, "expected [batch, num_labels], got {shape:?}");
    let batch = shape[0] as usize;
    let n = shape[1] as usize;
    assert_eq!(
        n,
        id2label.len(),
        "classifier output width {n} != id2label count {}",
        id2label.len(),
    );
    let flat = probs.data().to_f32_vec()?;
    let mut out = Vec::with_capacity(batch);
    for b in 0..batch {
        let mut row: Vec<(String, f32)> = (0..n)
            .map(|k| (id2label[k].clone(), flat[b * n + k]))
            .collect();
        row.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        out.push(row);
    }
    Ok(out)
}

/// Extract best QA spans from a `[B, S, 2]` logits tensor.
///
/// Algorithm matches HF Python's "argmax over valid positions, then
/// multiply": softmax start and end independently along the sequence
/// axis, then pick the `(i, j)` with `i <= j` maximising
/// `start_prob[i] + end_prob[j]`. Valid positions are restricted to
/// tokens with `sequence_ids == 1` (the context region in a
/// question/context pair encoding); `-1` (specials / padding)
/// already excludes padding, so no separate attention-mask check is
/// needed.
///
/// Decoding of the chosen span goes through the attached tokenizer
/// with `skip_special_tokens=true`, matching HF's default behaviour.
///
/// Errors:
/// - if a batch entry has no tokens with `sequence_id == 1`
///   (tokenizer didn't produce a pair encoding), since there's no
///   meaningful span to extract.
/// - on any shape mismatch: `logits.shape()` must be `[B, S, 2]`.
#[cfg(feature = "tokenizer")]
pub(crate) fn extract_best_span(
    logits: &Variable,
    enc: &crate::tokenizer::EncodedBatch,
    tokenizer: &crate::tokenizer::HfTokenizer,
) -> Result<Vec<Answer>> {
    let shape = logits.shape();
    assert_eq!(shape.len(), 3, "expected [B, S, 2], got {shape:?}");
    let batch = shape[0] as usize;
    let seq = shape[1] as usize;
    assert_eq!(shape[2], 2, "QA head must be 2-wide, got {}", shape[2]);

    let starts = logits.narrow(-1, 0, 1)?.softmax(1)?;
    let ends = logits.narrow(-1, 1, 1)?.softmax(1)?;
    let starts_flat = starts.data().to_f32_vec()?;
    let ends_flat = ends.data().to_f32_vec()?;
    let sequence_ids: Vec<i64> = enc.sequence_ids.data().to_i64_vec()?;
    let input_ids: Vec<i64> = enc.input_ids.data().to_i64_vec()?;

    let mut answers = Vec::with_capacity(batch);
    for b in 0..batch {
        let offset = b * seq;
        let valid: Vec<usize> = (0..seq)
            .filter(|&s| sequence_ids[offset + s] == 1)
            .collect();
        if valid.is_empty() {
            return Err(TensorError::new(
                "QA extract: no context tokens (sequence_id == 1) found; \
                 tokenizer did not produce a pair encoding",
            ));
        }
        let mut best = (valid[0], valid[0], f32::NEG_INFINITY);
        for &i in &valid {
            let sp = starts_flat[offset + i];
            for &j in valid.iter().filter(|&&j| j >= i) {
                let ep = ends_flat[offset + j];
                let score = sp + ep;
                if score > best.2 {
                    best = (i, j, score);
                }
            }
        }
        let (start, end, score) = best;
        let span_ids: Vec<u32> = input_ids[offset + start..=offset + end]
            .iter()
            .map(|&x| x as u32)
            .collect();
        let text = tokenizer
            .inner()
            .decode(&span_ids, /*skip_special_tokens=*/ true)
            .map_err(|e| TensorError::new(&format!("qa decode: {e}")))?;
        answers.push(Answer { text, start, end, score });
    }
    Ok(answers)
}

// ═════════════════════════════════════════════════════════════════════════
// Generic task-head bases
// ═════════════════════════════════════════════════════════════════════════
//
// BERT, RoBERTa, and DistilBERT share the same public task-head surface:
// each family exposes `{Family}ForSequenceClassification`,
// `{Family}ForTokenClassification`, `{Family}ForQuestionAnswering`, and
// `{Family}ForMaskedLM`. The only genuine per-family variation is:
//
// - the config type (`BertConfig`, `RobertaConfig`, `DistilBertConfig`)
// - which encoded tensors the graph takes as `forward_multi` inputs
//   (BERT: 4, RoBERTa: 3, DistilBERT: 2 — see [`EncoderInputs`])
// - the MLM mask-token spelling (`[MASK]` vs `<mask>`)
// - the per-family head graph layout (built in each family's
//   `on_device` constructor)
//
// Everything else — accessors, tokenizer-guarded `predict`/`answer`
// dispatch, forward/eval plumbing, loss glue, fill-mask — is
// family-agnostic and lives on the four generic structs below. Each
// family then type-aliases its four public head types to the matching
// generic specialization and adds only its bespoke `on_device`
// constructor (plus `from_pretrained` in `hub.rs`).

/// A model family's per-task-head encoding surface. Each family
/// (BERT, RoBERTa, DistilBERT, …) implements this on its config type.
///
/// `FAMILY_NAME` is spliced into runtime error messages so
/// `{Family}ForSequenceClassification::predict` can say
/// "BertForSequenceClassification::predict requires a tokenizer" even
/// though the method lives on the generic [`ClassificationHead<C>`].
///
/// `MASK_TOKEN` is the text used by [`MaskedLmHead::fill_mask`] to
/// find mask positions via the attached tokenizer. Must match the
/// token the tokenizer's vocabulary holds — `[MASK]` for BERT /
/// DistilBERT, `<mask>` for RoBERTa.
///
/// `encoder_inputs` builds the `forward_multi` input list from a
/// tokenised batch. The order must match the graph's `.input(&[...])`
/// declaration in the family's `on_device` constructors.
#[cfg(feature = "tokenizer")]
pub trait EncoderInputs {
    /// Family display name — `"Bert"`, `"Roberta"`, `"DistilBert"`.
    const FAMILY_NAME: &'static str;
    /// Mask token as it appears in the tokenizer's vocabulary.
    const MASK_TOKEN: &'static str;

    fn encoder_inputs(enc: &crate::tokenizer::EncodedBatch) -> Result<Vec<Variable>>;
}

// ── ClassificationHead ───────────────────────────────────────────────────

/// Generic sequence-classification head shared across families. See the
/// family-specific type aliases (`BertForSequenceClassification`,
/// `RobertaForSequenceClassification`,
/// `DistilBertForSequenceClassification`) for public entry points.
pub struct ClassificationHead<C: Clone> {
    pub(crate) graph: Graph,
    pub(crate) config: C,
    pub(crate) id2label: Vec<String>,
    #[cfg(feature = "tokenizer")]
    pub(crate) tokenizer: Option<crate::tokenizer::HfTokenizer>,
}

impl<C: Clone> ClassificationHead<C> {
    /// Family constructors call this after building the graph to
    /// populate the shared fields. `id2label` falls back to
    /// `["LABEL_0", …]` when the checkpoint config carries no labels.
    ///
    /// Callers are expected to pass an already-validated `num_labels`
    /// (typically via [`check_num_labels`] inside their `on_device`
    /// builder, which also uses the value to size the classifier).
    pub(crate) fn from_graph(
        graph: Graph,
        config: &C,
        num_labels: i64,
        id2label: Option<Vec<String>>,
    ) -> Self {
        let id2label = id2label.unwrap_or_else(|| default_labels(num_labels));
        Self {
            graph,
            config: config.clone(),
            id2label,
            #[cfg(feature = "tokenizer")]
            tokenizer: None,
        }
    }

    /// Borrow the underlying [`Graph`].
    pub fn graph(&self) -> &Graph { &self.graph }
    /// Consume `self` and return the underlying [`Graph`]. Used by
    /// `fdl flodl-hf export --hub` after auto-dispatching on the
    /// upstream `architectures[0]` — the head wrapper isn't needed
    /// past the load, only the graph (with `source_config` already
    /// set by `from_pretrained_on_device`).
    pub fn into_graph(self) -> Graph { self.graph }
    /// Borrow the config this head was built from.
    pub fn config(&self) -> &C { &self.config }
    /// Label names indexed by class id.
    pub fn labels(&self) -> &[String] { &self.id2label }

    /// Attach a tokenizer so [`predict`](Self::predict) can encode raw
    /// text. `from_pretrained` attaches one automatically.
    #[cfg(feature = "tokenizer")]
    pub fn with_tokenizer(mut self, tok: crate::tokenizer::HfTokenizer) -> Self {
        self.tokenizer = Some(tok);
        self
    }
}

#[cfg(feature = "tokenizer")]
impl<C: Clone + EncoderInputs> ClassificationHead<C> {
    /// Raw forward pass returning `[batch, num_labels]` logits. Does
    /// not change train / eval mode — caller's responsibility.
    pub fn forward_encoded(
        &self,
        enc: &crate::tokenizer::EncodedBatch,
    ) -> Result<Variable> {
        let inputs = C::encoder_inputs(enc)?;
        self.graph.forward_multi(&inputs)
    }

    /// Classify a pre-tokenised batch. Returns one label distribution
    /// per input, sorted by descending probability.
    pub fn classify(
        &self,
        enc: &crate::tokenizer::EncodedBatch,
    ) -> Result<Vec<Vec<(String, f32)>>> {
        self.graph.eval();
        let logits = self.forward_encoded(enc)?;
        logits_to_sorted_labels(&logits, &self.id2label)
    }

    /// One-shot text → label distribution. Encodes with the attached
    /// tokenizer, runs the graph in eval mode, softmaxes, and returns
    /// per-input label distributions sorted desc.
    pub fn predict(&self, texts: &[&str]) -> Result<Vec<Vec<(String, f32)>>> {
        let name = format!("{}ForSequenceClassification::predict", C::FAMILY_NAME);
        let tok = require_tokenizer(self.tokenizer.as_ref(), &name)?;
        let enc = tok.encode(texts)?;
        self.classify(&enc)
    }

    /// Forward pass plus sequence-classification loss. See
    /// [`sequence_classification_loss`].
    pub fn compute_loss(
        &self,
        enc: &crate::tokenizer::EncodedBatch,
        labels: &Variable,
    ) -> Result<Variable> {
        let logits = self.forward_encoded(enc)?;
        sequence_classification_loss(&logits, labels)
    }
}

impl<C: Clone> HasGraph for ClassificationHead<C> {
    fn graph(&self) -> &Graph { &self.graph }
}

/// Present the head as a [`Module`] so it can train through the unified
/// [`Trainer::builder`](flodl::Trainer::builder) / cooperative
/// [`Worker`](flodl::Worker) path (`M = ThisHead` directly). Every method
/// delegates to the inner [`Graph`] — that is where the parameters and
/// buffers live, so the defaults (which introspect the struct's own fields)
/// would find nothing and train a zero-parameter model.
///
/// The head's real forward is multi-input (`input_ids` + `attention_mask` via
/// [`Graph::forward_multi`], reached through [`Self::forward_encoded`] /
/// [`Self::compute_loss`]); `Module::forward` is a single-input fallback the
/// cooperative loop never calls (the user runs the head's own loss fn in their
/// `train_fn`). `as_any` presents the inner graph so DDP's multi-input replica
/// paths downcast to it (via `GraphExt::as_graph`) instead of the single-input
/// fallback.
impl<C: Clone> Module for ClassificationHead<C> {
    fn forward(&self, input: &Variable) -> Result<Variable> {
        self.graph.forward(input)
    }
    fn parameters(&self) -> Vec<Parameter> { self.graph.parameters() }
    fn buffers(&self) -> Vec<Buffer> { self.graph.buffers() }
    fn name(&self) -> &str { "classification_head" }
    fn set_training(&self, training: bool) { self.graph.set_training(training); }
    fn as_any(&self) -> Option<&dyn std::any::Any> { Some(&self.graph) }
}

// ── TaggingHead ──────────────────────────────────────────────────────────

/// Generic token-classification head. See the family-specific type
/// aliases `{Family}ForTokenClassification` for public entry points.
pub struct TaggingHead<C: Clone> {
    pub(crate) graph: Graph,
    pub(crate) config: C,
    pub(crate) id2label: Vec<String>,
    #[cfg(feature = "tokenizer")]
    pub(crate) tokenizer: Option<crate::tokenizer::HfTokenizer>,
}

impl<C: Clone> TaggingHead<C> {
    pub(crate) fn from_graph(
        graph: Graph,
        config: &C,
        num_labels: i64,
        id2label: Option<Vec<String>>,
    ) -> Self {
        let id2label = id2label.unwrap_or_else(|| default_labels(num_labels));
        Self {
            graph,
            config: config.clone(),
            id2label,
            #[cfg(feature = "tokenizer")]
            tokenizer: None,
        }
    }

    pub fn graph(&self) -> &Graph { &self.graph }
    /// Consume `self` and return the underlying [`Graph`] (used by
    /// the auto-dispatching Hub-mode export path).
    pub fn into_graph(self) -> Graph { self.graph }
    pub fn config(&self) -> &C { &self.config }
    pub fn labels(&self) -> &[String] { &self.id2label }

    #[cfg(feature = "tokenizer")]
    pub fn with_tokenizer(mut self, tok: crate::tokenizer::HfTokenizer) -> Self {
        self.tokenizer = Some(tok);
        self
    }
}

#[cfg(feature = "tokenizer")]
impl<C: Clone + EncoderInputs> TaggingHead<C> {
    /// Raw forward pass returning `[batch, seq_len, num_labels]`
    /// logits. Does not change train / eval mode.
    pub fn forward_encoded(
        &self,
        enc: &crate::tokenizer::EncodedBatch,
    ) -> Result<Variable> {
        let inputs = C::encoder_inputs(enc)?;
        self.graph.forward_multi(&inputs)
    }

    /// Tag every token in a pre-tokenised batch. Output shape matches
    /// `enc.input_ids`: `result[b][s]` is the top-1 prediction for
    /// batch entry `b`, position `s`. `TokenPrediction::attends`
    /// mirrors the attention mask so callers can drop `[PAD]` entries
    /// without re-tokenising.
    pub fn tag(
        &self,
        enc: &crate::tokenizer::EncodedBatch,
    ) -> Result<Vec<Vec<TokenPrediction>>> {
        let name = format!("{}ForTokenClassification::tag", C::FAMILY_NAME);
        let tok = require_tokenizer(self.tokenizer.as_ref(), &name)?;
        self.graph.eval();
        let logits = self.forward_encoded(enc)?;
        let probs = logits.softmax(-1)?;
        let shape = probs.shape();
        assert_eq!(shape.len(), 3, "expected [B, S, num_labels], got {shape:?}");
        let batch = shape[0] as usize;
        let seq = shape[1] as usize;
        let n = shape[2] as usize;
        let flat = probs.data().to_f32_vec()?;
        let input_ids: Vec<i64> = enc.input_ids.data().to_i64_vec()?;
        let attn_ids: Vec<i64> = enc.attention_mask.data().to_i64_vec()?;

        let mut out = Vec::with_capacity(batch);
        for b in 0..batch {
            let mut row = Vec::with_capacity(seq);
            for s in 0..seq {
                let base = (b * seq + s) * n;
                let (best_k, best_p) = argmax_f32(&flat[base..base + n]);
                let id = input_ids[b * seq + s] as u32;
                let token = tok
                    .inner()
                    .id_to_token(id)
                    .unwrap_or_else(|| format!("<unk_id={id}>"));
                row.push(TokenPrediction {
                    token,
                    label: self.id2label[best_k].clone(),
                    score: best_p,
                    attends: attn_ids[b * seq + s] != 0,
                });
            }
            out.push(row);
        }
        Ok(out)
    }

    /// One-shot text → per-token tags.
    pub fn predict(&self, texts: &[&str]) -> Result<Vec<Vec<TokenPrediction>>> {
        let name = format!("{}ForTokenClassification::predict", C::FAMILY_NAME);
        let tok = require_tokenizer(self.tokenizer.as_ref(), &name)?;
        let enc = tok.encode(texts)?;
        self.tag(&enc)
    }

    pub fn compute_loss(
        &self,
        enc: &crate::tokenizer::EncodedBatch,
        labels: &Variable,
    ) -> Result<Variable> {
        let logits = self.forward_encoded(enc)?;
        token_classification_loss(&logits, labels)
    }
}

impl<C: Clone> HasGraph for TaggingHead<C> {
    fn graph(&self) -> &Graph { &self.graph }
}

// ── QaHead ───────────────────────────────────────────────────────────────

/// Generic extractive question-answering head. See the family-specific
/// type aliases `{Family}ForQuestionAnswering` for public entry points.
pub struct QaHead<C: Clone> {
    pub(crate) graph: Graph,
    pub(crate) config: C,
    #[cfg(feature = "tokenizer")]
    pub(crate) tokenizer: Option<crate::tokenizer::HfTokenizer>,
}

impl<C: Clone> QaHead<C> {
    pub(crate) fn from_graph(graph: Graph, config: &C) -> Self {
        Self {
            graph,
            config: config.clone(),
            #[cfg(feature = "tokenizer")]
            tokenizer: None,
        }
    }

    pub fn graph(&self) -> &Graph { &self.graph }
    /// Consume `self` and return the underlying [`Graph`] (used by
    /// the auto-dispatching Hub-mode export path).
    pub fn into_graph(self) -> Graph { self.graph }
    pub fn config(&self) -> &C { &self.config }

    #[cfg(feature = "tokenizer")]
    pub fn with_tokenizer(mut self, tok: crate::tokenizer::HfTokenizer) -> Self {
        self.tokenizer = Some(tok);
        self
    }
}

#[cfg(feature = "tokenizer")]
impl<C: Clone + EncoderInputs> QaHead<C> {
    /// Raw forward pass returning `[batch, seq_len, 2]` logits. Start
    /// logits on slice `0` of the last axis, end logits on slice `1`.
    /// Does not change train / eval mode.
    pub fn forward_encoded(
        &self,
        enc: &crate::tokenizer::EncodedBatch,
    ) -> Result<Variable> {
        let inputs = C::encoder_inputs(enc)?;
        self.graph.forward_multi(&inputs)
    }

    /// Answer one `(question, context)` pair. Returns the highest-
    /// scoring span over the context tokens.
    pub fn answer(&self, question: &str, context: &str) -> Result<Answer> {
        let mut out = self.answer_batch(&[(question, context)])?;
        Ok(out.pop().expect("answer_batch returns one per input"))
    }

    /// Batched variant of [`answer`](Self::answer).
    pub fn answer_batch(&self, pairs: &[(&str, &str)]) -> Result<Vec<Answer>> {
        let name = format!("{}ForQuestionAnswering::answer", C::FAMILY_NAME);
        let tok = require_tokenizer(self.tokenizer.as_ref(), &name)?;
        let enc = tok.encode_pairs(pairs)?;
        self.extract(&enc)
    }

    /// Run the graph on a pre-tokenised `(question, context)` batch
    /// and extract best spans. See the crate-internal
    /// `extract_best_span` helper for the per-row logit-to-span logic.
    pub fn extract(
        &self,
        enc: &crate::tokenizer::EncodedBatch,
    ) -> Result<Vec<Answer>> {
        let name = format!("{}ForQuestionAnswering::extract", C::FAMILY_NAME);
        let tok = require_tokenizer(self.tokenizer.as_ref(), &name)?;
        self.graph.eval();
        let logits = self.forward_encoded(enc)?;
        extract_best_span(&logits, enc, tok)
    }

    /// Forward pass plus extractive QA loss. See
    /// [`question_answering_loss`].
    pub fn compute_loss(
        &self,
        enc: &crate::tokenizer::EncodedBatch,
        start_positions: &Variable,
        end_positions: &Variable,
    ) -> Result<Variable> {
        let logits = self.forward_encoded(enc)?;
        question_answering_loss(&logits, start_positions, end_positions)
    }
}

impl<C: Clone> HasGraph for QaHead<C> {
    fn graph(&self) -> &Graph { &self.graph }
}

// ── MaskedLmHead ─────────────────────────────────────────────────────────

/// Generic masked-language-modelling head. See the family-specific
/// type aliases `{Family}ForMaskedLM` for public entry points.
pub struct MaskedLmHead<C: Clone> {
    pub(crate) graph: Graph,
    pub(crate) config: C,
    #[cfg(feature = "tokenizer")]
    pub(crate) tokenizer: Option<crate::tokenizer::HfTokenizer>,
}

impl<C: Clone> MaskedLmHead<C> {
    pub(crate) fn from_graph(graph: Graph, config: &C) -> Self {
        Self {
            graph,
            config: config.clone(),
            #[cfg(feature = "tokenizer")]
            tokenizer: None,
        }
    }

    pub fn graph(&self) -> &Graph { &self.graph }
    /// Consume `self` and return the underlying [`Graph`] (used by
    /// the auto-dispatching Hub-mode export path).
    pub fn into_graph(self) -> Graph { self.graph }
    pub fn config(&self) -> &C { &self.config }

    #[cfg(feature = "tokenizer")]
    pub fn with_tokenizer(mut self, tok: crate::tokenizer::HfTokenizer) -> Self {
        self.tokenizer = Some(tok);
        self
    }
}

#[cfg(feature = "tokenizer")]
impl<C: Clone + EncoderInputs> MaskedLmHead<C> {
    /// Raw forward pass returning `[batch, seq_len, vocab_size]`
    /// logits over the vocabulary. Does not change train / eval mode.
    pub fn forward_encoded(
        &self,
        enc: &crate::tokenizer::EncodedBatch,
    ) -> Result<Variable> {
        let inputs = C::encoder_inputs(enc)?;
        self.graph.forward_multi(&inputs)
    }

    /// Forward pass plus masked-LM loss. See [`masked_lm_loss`] for
    /// the label convention (`-100` at ignored positions, original
    /// token id at masked positions).
    pub fn compute_loss(
        &self,
        enc: &crate::tokenizer::EncodedBatch,
        labels: &Variable,
    ) -> Result<Variable> {
        let logits = self.forward_encoded(enc)?;
        masked_lm_loss(&logits, labels)
    }

    /// Fill every mask-token position in `text` with its top-`k`
    /// predicted replacements, sorted by descending softmax
    /// probability. The mask-token spelling comes from
    /// `C::MASK_TOKEN` — `[MASK]` for BERT / DistilBERT, `<mask>`
    /// for RoBERTa.
    pub fn fill_mask(
        &self,
        text: &str,
        top_k: usize,
    ) -> Result<Vec<Vec<(String, f32)>>> {
        if top_k == 0 {
            return Err(TensorError::new("fill_mask: top_k must be > 0"));
        }
        let name = format!("{}ForMaskedLM::fill_mask", C::FAMILY_NAME);
        let tok = require_tokenizer(self.tokenizer.as_ref(), &name)?;
        let mask_tok = C::MASK_TOKEN;
        let mask_id = tok.inner().token_to_id(mask_tok).ok_or_else(|| {
            TensorError::new(&format!(
                "fill_mask: tokenizer has no {mask_tok} token",
            ))
        })? as i64;

        self.graph.eval();
        let enc = tok.encode(&[text])?;
        let logits = self.forward_encoded(&enc)?;
        let probs = logits.data().softmax(-1)?;

        let ids_row = enc.input_ids.data().select(0, 0)?.to_i64_vec()?;
        let mut out = Vec::new();
        for (pos, id) in ids_row.iter().enumerate() {
            if *id != mask_id {
                continue;
            }
            let row = probs.select(0, 0)?.select(0, pos as i64)?;
            let (vals, idxs) = row.topk(top_k as i64, 0, /*largest=*/ true, /*sorted=*/ true)?;
            let score_vec = vals.to_f32_vec()?;
            let id_vec = idxs.to_i64_vec()?;
            let picks: Vec<(String, f32)> = id_vec
                .iter()
                .zip(score_vec.iter())
                .map(|(i, s)| {
                    let tok_str = tok
                        .inner()
                        .id_to_token(*i as u32)
                        .unwrap_or_else(|| format!("[UNK_{i}]"));
                    (tok_str, *s)
                })
                .collect();
            out.push(picks);
        }

        if out.is_empty() {
            return Err(TensorError::new(&format!(
                "fill_mask: input contains no {mask_tok} token",
            )));
        }
        Ok(out)
    }
}

impl<C: Clone> HasGraph for MaskedLmHead<C> {
    fn graph(&self) -> &Graph { &self.graph }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flodl::{DType, Device, Tensor, TensorOptions};
    fn cpu() -> Device { Device::CPU }

    #[test]
    fn default_labels_generates_label_k_fallback() {
        assert_eq!(default_labels(3), vec!["LABEL_0", "LABEL_1", "LABEL_2"]);
        assert!(default_labels(0).is_empty());
    }

    #[test]
    fn check_num_labels_rejects_nonpositive() {
        assert_eq!(check_num_labels(3).unwrap(), 3);
        assert!(check_num_labels(0).is_err());
        assert!(check_num_labels(-1).is_err());
    }

    fn logits_2d(data: &[f32], rows: i64, cols: i64) -> Variable {
        Variable::new(
            Tensor::from_f32(data, &[rows, cols], cpu()).unwrap(),
            true,
        )
    }

    fn labels_1d(data: &[i64], n: i64) -> Variable {
        Variable::new(
            Tensor::from_i64(data, &[n], cpu()).unwrap(),
            false,
        )
    }

    #[test]
    fn sequence_classification_loss_rejects_wrong_rank() {
        let logits = Variable::new(
            Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[4], cpu()).unwrap(),
            true,
        );
        let labels = labels_1d(&[0, 1, 0, 1], 4);
        let err = sequence_classification_loss(&logits, &labels).unwrap_err();
        assert!(err.to_string().contains("must be [batch, num_labels]"));
    }

    #[test]
    fn sequence_classification_loss_backward_flows() {
        // 2 batches, 3 classes. Correct class gets high logit so loss is small.
        let logits = logits_2d(&[5.0, 0.1, 0.1, 0.1, 5.0, 0.1], 2, 3);
        let labels = labels_1d(&[0, 1], 2);
        let loss = sequence_classification_loss(&logits, &labels).unwrap();
        loss.backward().unwrap();
        assert!(logits.grad().is_some(), "logits must receive grad");
        let loss_val = loss.data().to_f32_vec().unwrap()[0];
        assert!(loss_val < 0.1, "expected small loss, got {loss_val}");
    }

    #[test]
    fn token_classification_loss_flattens_and_ignores_minus_100() {
        // batch=2, seq=3, num_labels=2. Labels: [[0, -100, 1], [1, 0, -100]].
        // Position with -100 should not contribute to the loss.
        let logits_data = [
            5.0, 0.0,   0.0, 0.0,   0.0, 5.0,   // batch 0
            0.0, 5.0,   5.0, 0.0,   0.0, 0.0,   // batch 1
        ];
        let logits = Variable::new(
            Tensor::from_f32(&logits_data, &[2, 3, 2], cpu()).unwrap(),
            true,
        );
        let labels = Variable::new(
            Tensor::from_i64(&[0, -100, 1, 1, 0, -100], &[2, 3], cpu()).unwrap(),
            false,
        );
        let loss = token_classification_loss(&logits, &labels).unwrap();
        loss.backward().unwrap();
        assert!(logits.grad().is_some(), "logits must receive grad");
        // All 4 non-ignored positions are confidently correct, so loss is tiny.
        let loss_val = loss.data().to_f32_vec().unwrap()[0];
        assert!(loss_val < 0.1, "expected small loss (all correct), got {loss_val}");
    }

    #[test]
    fn token_classification_loss_rejects_wrong_rank() {
        let logits = logits_2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let labels = labels_1d(&[0, 1], 2);
        let err = token_classification_loss(&logits, &labels).unwrap_err();
        assert!(err.to_string().contains("[batch, seq_len, num_labels]"));
    }

    #[test]
    fn question_answering_loss_averages_two_heads() {
        // batch=2, seq=4. QA logits stack (start, end) on last dim.
        // Gold spans: batch 0 → (start=1, end=2), batch 1 → (start=0, end=3).
        // Logits peaked at the gold positions for both heads: loss ~0.
        let opts = TensorOptions { dtype: DType::Float32, device: cpu() };
        let logits_flat = Tensor::zeros(&[2, 4, 2], opts).unwrap();
        // Write peaks via an inplace fill through addition: build from raw data instead.
        let raw: Vec<f32> = {
            let mut v = vec![0.0_f32; 2 * 4 * 2];
            // helper to compute linear index from (b, s, k)
            let ix = |b: usize, s: usize, k: usize| (b * 4 + s) * 2 + k;
            // batch 0 start=1, end=2
            v[ix(0, 1, 0)] = 10.0;
            v[ix(0, 2, 1)] = 10.0;
            // batch 1 start=0, end=3
            v[ix(1, 0, 0)] = 10.0;
            v[ix(1, 3, 1)] = 10.0;
            v
        };
        drop(logits_flat);
        let logits = Variable::new(
            Tensor::from_f32(&raw, &[2, 4, 2], cpu()).unwrap(),
            true,
        );
        let starts = labels_1d(&[1, 0], 2);
        let ends   = labels_1d(&[2, 3], 2);
        let loss = question_answering_loss(&logits, &starts, &ends).unwrap();
        loss.backward().unwrap();
        assert!(logits.grad().is_some(), "logits must receive grad");
        let loss_val = loss.data().to_f32_vec().unwrap()[0];
        assert!(loss_val < 0.01, "expected tiny loss at peaked logits, got {loss_val}");
    }

    #[test]
    fn masked_lm_loss_flattens_and_ignores_minus_100() {
        // batch=2, seq=3, vocab=4. Labels: [[2, -100, 0], [-100, 1, 3]].
        // The -100 positions must not contribute; the other 4 positions
        // all have their target class peaked, so loss is tiny.
        let logits_data = [
            // batch 0
            0.0, 0.0, 5.0, 0.0,   0.0, 0.0, 0.0, 0.0,   5.0, 0.0, 0.0, 0.0,
            // batch 1
            0.0, 0.0, 0.0, 0.0,   0.0, 5.0, 0.0, 0.0,   0.0, 0.0, 0.0, 5.0,
        ];
        let logits = Variable::new(
            Tensor::from_f32(&logits_data, &[2, 3, 4], cpu()).unwrap(),
            true,
        );
        let labels = Variable::new(
            Tensor::from_i64(&[2, -100, 0, -100, 1, 3], &[2, 3], cpu()).unwrap(),
            false,
        );
        let loss = masked_lm_loss(&logits, &labels).unwrap();
        loss.backward().unwrap();
        assert!(logits.grad().is_some(), "logits must receive grad");
        let loss_val = loss.data().to_f32_vec().unwrap()[0];
        assert!(loss_val < 0.1, "expected small loss (all targets peaked), got {loss_val}");
    }

    #[test]
    fn masked_lm_loss_rejects_wrong_rank() {
        let logits = logits_2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let labels = labels_1d(&[0, 1], 2);
        let err = masked_lm_loss(&logits, &labels).unwrap_err();
        assert!(err.to_string().contains("[batch, seq_len, vocab_size]"));
    }

    #[test]
    fn question_answering_loss_rejects_wrong_last_dim() {
        let logits = Variable::new(
            Tensor::from_f32(&[0.0_f32; 12], &[2, 3, 2], cpu()).unwrap(),
            true,
        );
        let starts = labels_1d(&[0, 1], 2);
        let ends   = labels_1d(&[2, 2], 2);
        assert!(question_answering_loss(&logits, &starts, &ends).is_ok());

        let bad = Variable::new(
            Tensor::from_f32(&[0.0_f32; 18], &[2, 3, 3], cpu()).unwrap(),
            true,
        );
        let err = question_answering_loss(&bad, &starts, &ends).unwrap_err();
        assert!(err.to_string().contains("[batch, seq_len, 2]"));
    }
}
