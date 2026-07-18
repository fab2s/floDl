use std::cell::RefCell;

use crate::autograd::Variable;
use crate::tensor::{Device, DType, Result, RnnParams, Tensor, TensorOptions};

use super::lstmcell::LSTMCell;
use super::parameter::Parameter;
use super::Module;

/// Multi-layer LSTM (Long Short-Term Memory) sequence module.
///
/// Wraps multiple [`LSTMCell`] layers and loops over timesteps, matching
/// the PyTorch `nn.LSTM` interface. Each layer feeds its output sequence
/// as input to the next layer.
///
/// ```ignore
/// let lstm = LSTM::new(4, 8, 2)?;       // input=4, hidden=8, 2 layers
/// let x = Variable::new(Tensor::randn(&[10, 1, 4], opts)?, false); // [seq, batch, input]
/// let (output, (h_n, c_n)) = lstm.forward_seq(&x, None)?;
/// // output: [10, 1, 8], h_n: [2, 1, 8], c_n: [2, 1, 8]
/// ```
pub struct LSTM {
    cells: Vec<LSTMCell>,
    hidden_size: i64,
    num_layers: usize,
    batch_first: bool,
    /// cuDNN param cache (C++ side) + the `data_generation` of every
    /// parameter it was built from; rebuilt when any generation changes
    /// (i.e. `set_data` replaced a parameter tensor: checkpoint load,
    /// cast, device move).
    rnn_params: RefCell<Option<(RnnParams, Vec<u64>)>>,
}

impl LSTM {
    /// Create a multi-layer LSTM on CPU.
    pub fn new(input_size: i64, hidden_size: i64, num_layers: usize) -> Result<Self> {
        Self::on_device(input_size, hidden_size, num_layers, false, Device::CPU)
    }

    /// Create a multi-layer LSTM on a specific device.
    pub fn on_device(
        input_size: i64,
        hidden_size: i64,
        num_layers: usize,
        batch_first: bool,
        device: Device,
    ) -> Result<Self> {
        assert!(num_layers >= 1, "LSTM requires at least 1 layer");
        let mut cells = Vec::with_capacity(num_layers);
        for layer in 0..num_layers {
            let in_size = if layer == 0 { input_size } else { hidden_size };
            cells.push(LSTMCell::on_device(in_size, hidden_size, device)?);
        }
        Ok(LSTM {
            cells,
            hidden_size,
            num_layers,
            batch_first,
            rnn_params: RefCell::new(None),
        })
    }

    /// Set batch_first mode. When true, input/output are `[batch, seq, features]`.
    pub fn batch_first(mut self, batch_first: bool) -> Self {
        self.batch_first = batch_first;
        self
    }

    /// Forward pass over a full sequence.
    ///
    /// - `input`: `[seq_len, batch, input_size]` (or `[batch, seq_len, input_size]` if batch_first)
    /// - `state_0`: optional initial state `(h_0, c_0)` each `[num_layers, batch, hidden_size]`,
    ///   or None for zeros
    ///
    /// Returns `(output, (h_n, c_n))`:
    /// - `output`: `[seq_len, batch, hidden_size]` (or `[batch, seq_len, hidden_size]` if batch_first)
    /// - `h_n`: `[num_layers, batch, hidden_size]`
    /// - `c_n`: `[num_layers, batch, hidden_size]`
    pub fn forward_seq(
        &self,
        input: &Variable,
        state_0: Option<(&Variable, &Variable)>,
    ) -> Result<(Variable, (Variable, Variable))> {
        let shape = input.shape();
        let batch = if self.batch_first { shape[0] } else { shape[1] };
        let nl = self.num_layers as i64;
        let hs = self.hidden_size;
        let opts = TensorOptions {
            dtype: DType::Float32,
            device: self.cells[0].parameters()[0].variable.device(),
        };

        // Initial states (zeros if not provided)
        let (h_0, c_0) = match state_0 {
            Some((h, c)) => (h.data(), c.data()),
            None => (
                Tensor::zeros(&[nl, batch, hs], opts)?,
                Tensor::zeros(&[nl, batch, hs], opts)?,
            ),
        };

        // Lazily create C++ cached params on first forward (with cuDNN
        // flatten); rebuild whenever a parameter tensor was REPLACED
        // (set_data: checkpoint load, dtype cast, device move) — the cache
        // pins the tensors it was built from, so a stale cache keeps
        // computing with the pre-replacement weights. In-place updates
        // (optimizer steps, DDP copy_) write through the pinned tensors
        // and keep the cache valid. Cache hits pass the opaque handle
        // directly — the staleness check is an integer compare, no FFI.
        {
            let cell_params = self.parameters();
            let generations: Vec<u64> = cell_params.iter()
                .map(|p| p.variable.data_generation())
                .collect();
            let mut cache = self.rnn_params.borrow_mut();
            let stale = match cache.as_ref() {
                Some((_, cached)) => *cached != generations,
                None => true,
            };
            if stale {
                let params: Vec<Tensor> = cell_params.iter()
                    .map(|p| p.variable.data())
                    .collect();
                *cache = Some((
                    RnnParams::new(&params, 2, nl, self.batch_first, true)?,
                    generations,
                ));
            }
        }
        let cache = self.rnn_params.borrow();
        let (output, h_n, c_n) = input.data().lstm_seq_cached(
            &h_0, &c_0, &cache.as_ref().unwrap().0, nl, self.batch_first,
        )?;

        Ok((Variable::wrap(output), (Variable::wrap(h_n), Variable::wrap(c_n))))
    }
}

impl Module for LSTM {
    fn name(&self) -> &str { "lstm" }

    /// Module trait forward: runs the full sequence with zero-initialized state.
    /// Returns only the output sequence (not h_n/c_n). Use [`forward_seq`](LSTM::forward_seq)
    /// for explicit state access.
    fn forward(&self, input: &Variable) -> Result<Variable> {
        let (output, _) = self.forward_seq(input, None)?;
        Ok(output)
    }

    fn parameters(&self) -> Vec<Parameter> {
        self.cells.iter().flat_map(|c| c.parameters()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lstm_shapes() {
        let dev = crate::tensor::test_device();
        let opts = crate::tensor::test_opts();
        let lstm = LSTM::on_device(4, 8, 2, false, dev).unwrap();

        let x = Variable::new(Tensor::randn(&[5, 3, 4], opts).unwrap(), false);
        let (output, (h_n, c_n)) = lstm.forward_seq(&x, None).unwrap();

        assert_eq!(output.shape(), vec![5, 3, 8]); // [seq, batch, hidden]
        assert_eq!(h_n.shape(), vec![2, 3, 8]);    // [layers, batch, hidden]
        assert_eq!(c_n.shape(), vec![2, 3, 8]);    // [layers, batch, hidden]
    }

    #[test]
    fn test_lstm_cache_rebuilds_after_set_data() {
        // Regression: the cuDNN param cache pinned the tensors it was
        // built from, so replacing params via set_data (the checkpoint-load
        // path) left forward computing with the pre-load weights forever.
        let dev = crate::tensor::test_device();
        let opts = crate::tensor::test_opts();
        let a = LSTM::on_device(4, 6, 2, false, dev).unwrap();
        let b = LSTM::on_device(4, 6, 2, false, dev).unwrap();
        let x = Variable::new(Tensor::randn(&[3, 2, 4], opts).unwrap(), false);

        let out_a = a.forward_seq(&x, None).unwrap().0.data().to_f32_vec().unwrap();
        let _ = b.forward_seq(&x, None).unwrap(); // builds b's cache from its own init

        // Replace b's params with a's — exactly what load_checkpoint does.
        for (pa, pb) in a.parameters().iter().zip(b.parameters().iter()) {
            pb.variable.set_data(pa.variable.data());
        }
        let out_b = b.forward_seq(&x, None).unwrap().0.data().to_f32_vec().unwrap();
        let max_diff = out_a
            .iter()
            .zip(&out_b)
            .map(|(l, r)| (l - r).abs())
            .fold(0f32, f32::max);
        assert!(
            max_diff < 1e-5,
            "forward after set_data must use the new weights (max diff {max_diff})"
        );
    }

    #[test]
    fn test_lstm_checkpoint_roundtrip_forward_matches() {
        // End-to-end user path: save_checkpoint(a) -> load_checkpoint into a
        // freshly-initialized b -> b's forward must match a's. Complements
        // the set_data cache test: this drives the FULL checkpoint codec +
        // named-param mapping, and load_checkpoint's internal set_data is
        // what must trigger the cuDNN cache rebuild.
        use crate::nn::checkpoint::{load_checkpoint, save_checkpoint};
        let dev = crate::tensor::test_device();
        let opts = crate::tensor::test_opts();
        let a = LSTM::on_device(4, 6, 2, false, dev).unwrap();
        let b = LSTM::on_device(4, 6, 2, false, dev).unwrap();
        let x = Variable::new(Tensor::randn(&[3, 2, 4], opts).unwrap(), false);

        let out_a = a.forward_seq(&x, None).unwrap().0.data().to_f32_vec().unwrap();
        let _ = b.forward_seq(&x, None).unwrap(); // build b's cache from its own init

        // Index-named pairs: same order in a and b, so load maps by name.
        let named = |m: &LSTM| -> Vec<(String, crate::nn::Parameter)> {
            m.parameters().into_iter().enumerate().map(|(i, p)| (i.to_string(), p)).collect()
        };
        let mut buf = Vec::new();
        save_checkpoint(&mut buf, &named(&a), &[], None).unwrap();
        let mut cursor = std::io::Cursor::new(buf);
        load_checkpoint(&mut cursor, &named(&b), &[], None).unwrap();

        let out_b = b.forward_seq(&x, None).unwrap().0.data().to_f32_vec().unwrap();
        let max_diff = out_a.iter().zip(&out_b).map(|(l, r)| (l - r).abs()).fold(0f32, f32::max);
        assert!(
            max_diff < 1e-5,
            "forward after checkpoint load must match source (max diff {max_diff})"
        );
    }

    #[test]
    fn test_lstm_batch_first() {
        let dev = crate::tensor::test_device();
        let opts = crate::tensor::test_opts();
        let lstm = LSTM::on_device(4, 8, 2, true, dev).unwrap();

        let x = Variable::new(Tensor::randn(&[3, 5, 4], opts).unwrap(), false);
        let (output, (h_n, c_n)) = lstm.forward_seq(&x, None).unwrap();

        assert_eq!(output.shape(), vec![3, 5, 8]); // [batch, seq, hidden]
        assert_eq!(h_n.shape(), vec![2, 3, 8]);
        assert_eq!(c_n.shape(), vec![2, 3, 8]);
    }

    #[test]
    fn test_lstm_with_initial_state() {
        let dev = crate::tensor::test_device();
        let opts = crate::tensor::test_opts();
        let lstm = LSTM::on_device(4, 8, 2, false, dev).unwrap();

        let x = Variable::new(Tensor::randn(&[5, 3, 4], opts).unwrap(), false);
        let h0 = Variable::new(Tensor::randn(&[2, 3, 8], opts).unwrap(), false);
        let c0 = Variable::new(Tensor::randn(&[2, 3, 8], opts).unwrap(), false);
        let (output, (h_n, c_n)) = lstm.forward_seq(&x, Some((&h0, &c0))).unwrap();

        assert_eq!(output.shape(), vec![5, 3, 8]);
        assert_eq!(h_n.shape(), vec![2, 3, 8]);
        assert_eq!(c_n.shape(), vec![2, 3, 8]);
    }

    #[test]
    fn test_lstm_single_layer() {
        let dev = crate::tensor::test_device();
        let opts = crate::tensor::test_opts();
        let lstm = LSTM::on_device(4, 8, 1, false, dev).unwrap();

        let x = Variable::new(Tensor::randn(&[5, 3, 4], opts).unwrap(), false);
        let (output, (h_n, c_n)) = lstm.forward_seq(&x, None).unwrap();

        assert_eq!(output.shape(), vec![5, 3, 8]);
        assert_eq!(h_n.shape(), vec![1, 3, 8]);
        assert_eq!(c_n.shape(), vec![1, 3, 8]);
    }

    #[test]
    fn test_lstm_gradient() {
        let dev = crate::tensor::test_device();
        let opts = crate::tensor::test_opts();
        let lstm = LSTM::on_device(4, 8, 2, false, dev).unwrap();

        let x = Variable::new(Tensor::randn(&[5, 3, 4], opts).unwrap(), true);
        let (output, _) = lstm.forward_seq(&x, None).unwrap();
        let loss = output.sum().unwrap();
        loss.backward().unwrap();

        for p in lstm.parameters() {
            assert!(p.variable.grad().is_some(), "missing grad for {}", p.name);
        }
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_lstm_module_forward() {
        let dev = crate::tensor::test_device();
        let opts = crate::tensor::test_opts();
        let lstm = LSTM::on_device(4, 8, 2, false, dev).unwrap();

        let x = Variable::new(Tensor::randn(&[5, 3, 4], opts).unwrap(), false);
        let y = lstm.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![5, 3, 8]);
    }

    #[test]
    fn test_lstm_parameters_count() {
        let dev = crate::tensor::test_device();
        let lstm = LSTM::on_device(4, 8, 2, false, dev).unwrap();

        // Layer 0: 4 params (w_ih, w_hh, b_ih, b_hh)
        // Layer 1: 4 params
        assert_eq!(lstm.parameters().len(), 8);
    }

    #[test]
    fn test_lstm_builder_pattern() {
        let dev = crate::tensor::test_device();
        let lstm = LSTM::on_device(4, 8, 1, false, dev).unwrap().batch_first(true);
        let opts = crate::tensor::test_opts();
        let x = Variable::new(Tensor::randn(&[3, 5, 4], opts).unwrap(), false);
        let (output, _) = lstm.forward_seq(&x, None).unwrap();
        assert_eq!(output.shape(), vec![3, 5, 8]); // [batch, seq, hidden]
    }
}
