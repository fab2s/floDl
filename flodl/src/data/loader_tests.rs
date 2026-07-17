    use super::*;
    use crate::tensor::{DType, TensorOptions, test_device};

    struct SimpleData {
        x: Tensor,
        y: Tensor,
    }

    impl DataSet for SimpleData {
        fn len(&self) -> usize {
            self.x.shape()[0] as usize
        }
        fn get(&self, index: usize) -> Result<Vec<Tensor>> {
            Ok(vec![
                self.x.select(0, index as i64)?,
                self.y.select(0, index as i64)?,
            ])
        }
    }

    struct SequentialData {
        n: usize,
    }

    impl DataSet for SequentialData {
        fn len(&self) -> usize {
            self.n
        }
        fn get(&self, index: usize) -> Result<Vec<Tensor>> {
            Ok(vec![
                Tensor::from_f32(&[index as f32], &[1], Device::CPU)?,
            ])
        }
    }

    struct PairBatch {
        x: Tensor,
        y: Tensor,
    }

    impl BatchDataSet for PairBatch {
        fn len(&self) -> usize {
            self.x.shape()[0] as usize
        }
        fn get_batch(&self, indices: &[usize]) -> Result<Vec<Tensor>> {
            let idx: Vec<i64> = indices.iter().map(|&i| i as i64).collect();
            let idx_t = Tensor::from_i64(&idx, &[idx.len() as i64], Device::CPU)?;
            Ok(vec![
                self.x.index_select(0, &idx_t)?,
                self.y.index_select(0, &idx_t)?,
            ])
        }
    }

    fn make_data(n: usize) -> SimpleData {
        let opts = TensorOptions { dtype: DType::Float32, device: Device::CPU };
        SimpleData {
            x: Tensor::randn(&[n as i64, 4], opts).unwrap(),
            y: Tensor::randn(&[n as i64, 2], opts).unwrap(),
        }
    }

    fn make_cpu_data_for_device(n: usize) -> SimpleData {
        // DataSet contract: return CPU tensors. DataLoader handles device transfer.
        let opts = TensorOptions { dtype: DType::Float32, device: Device::CPU };
        SimpleData {
            x: Tensor::randn(&[n as i64, 4], opts).unwrap(),
            y: Tensor::randn(&[n as i64, 2], opts).unwrap(),
        }
    }

    #[test]
    fn test_basic_epoch_iteration() {
        let data = make_data(20);
        let mut loader = DataLoader::from_dataset(data)
            .batch_size(5)
            .build()
            .unwrap();

        let batches: Vec<Batch> = loader.epoch(0).map(|b| b.unwrap()).collect();
        assert_eq!(batches.len(), 4); // 20 / 5 = 4
        for b in &batches {
            assert_eq!(b.len(), 2); // x and y
            assert_eq!(b[0].shape(), &[5, 4]);
            assert_eq!(b[1].shape(), &[5, 2]);
        }
    }

    #[test]
    fn test_drop_last_true() {
        let data = make_data(22);
        let mut loader = DataLoader::from_dataset(data)
            .batch_size(5)
            .drop_last(true)
            .build()
            .unwrap();

        let batches: Vec<Batch> = loader.epoch(0).map(|b| b.unwrap()).collect();
        assert_eq!(batches.len(), 4); // 22 / 5 = 4, drop remainder of 2
    }

    #[test]
    fn test_drop_last_false() {
        let data = make_data(22);
        let mut loader = DataLoader::from_dataset(data)
            .batch_size(5)
            .drop_last(false)
            .build()
            .unwrap();

        let batches: Vec<Batch> = loader.epoch(0).map(|b| b.unwrap()).collect();
        assert_eq!(batches.len(), 5); // 4 full + 1 partial
        assert_eq!(batches[4][0].shape(), &[2, 4]); // last batch has 2 samples
    }

    #[test]
    fn test_sequential_sampler() {
        let mut loader = DataLoader::from_dataset(SequentialData { n: 10 })
            .batch_size(3)
            .shuffle(false)
            .drop_last(false)
            .build()
            .unwrap();

        // Epoch 0 and epoch 1 should produce the same ordering
        let e0: Vec<f32> = loader
            .epoch(0)
            .flat_map(|b| {
                let b = b.unwrap();
                b[0].to_f32_vec().unwrap()
            })
            .collect();
        let e1: Vec<f32> = loader
            .epoch(1)
            .flat_map(|b| {
                let b = b.unwrap();
                b[0].to_f32_vec().unwrap()
            })
            .collect();
        assert_eq!(e0, e1);
        // And they should be in order
        assert_eq!(e0, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_shuffle_different_epochs() {
        let mut loader = DataLoader::from_dataset(SequentialData { n: 20 })
            .batch_size(20)
            .drop_last(false)
            .build()
            .unwrap();

        let e0: Vec<f32> = loader.epoch(0).next().unwrap().unwrap()[0]
            .to_f32_vec()
            .unwrap();
        let e1: Vec<f32> = loader.epoch(1).next().unwrap().unwrap()[0]
            .to_f32_vec()
            .unwrap();
        // Different epochs should yield different orderings (with overwhelming probability)
        assert_ne!(e0, e1);
    }

    #[test]
    fn test_shuffle_reproducible() {
        let data1 = SequentialData { n: 20 };
        let data2 = SequentialData { n: 20 };
        let mut l1 = DataLoader::from_dataset(data1)
            .batch_size(20)
            .seed(99)
            .drop_last(false)
            .build()
            .unwrap();
        let mut l2 = DataLoader::from_dataset(data2)
            .batch_size(20)
            .seed(99)
            .drop_last(false)
            .build()
            .unwrap();

        let e1: Vec<f32> = l1.epoch(3).next().unwrap().unwrap()[0]
            .to_f32_vec()
            .unwrap();
        let e2: Vec<f32> = l2.epoch(3).next().unwrap().unwrap()[0]
            .to_f32_vec()
            .unwrap();
        assert_eq!(e1, e2);
    }

    #[test]
    fn test_all_samples_visited() {
        let mut loader = DataLoader::from_dataset(SequentialData { n: 10 })
            .batch_size(3)
            .drop_last(false)
            .build()
            .unwrap();

        let mut vals: Vec<f32> = loader
            .epoch(0)
            .flat_map(|b| {
                let b = b.unwrap();
                b[0].to_f32_vec().unwrap()
            })
            .collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(
            vals,
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        );
    }

    #[test]
    fn test_batch_dataset_path() {
        let opts = TensorOptions { dtype: DType::Float32, device: Device::CPU };
        let batch_ds = PairBatch {
            x: Tensor::randn(&[30, 8], opts).unwrap(),
            y: Tensor::randn(&[30, 3], opts).unwrap(),
        };
        let mut loader = DataLoader::from_batch_dataset(batch_ds)
            .batch_size(10)
            .build()
            .unwrap();

        let batches: Vec<Batch> = loader.epoch(0).map(|b| b.unwrap()).collect();
        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0][0].shape(), &[10, 8]);
        assert_eq!(batches[0][1].shape(), &[10, 3]);
    }

    #[test]
    fn test_exact_size_iterator() {
        let data = make_data(20);
        let mut loader = DataLoader::from_dataset(data)
            .batch_size(5)
            .build()
            .unwrap();

        let iter = loader.epoch(0);
        assert_eq!(iter.len(), 4);
    }

    #[test]
    fn test_loader_metadata() {
        let data = make_data(50);
        let loader = DataLoader::from_dataset(data)
            .batch_size(8)
            .build()
            .unwrap();

        assert_eq!(loader.len(), 50);
        assert_eq!(loader.batch_size(), 8);
        assert_eq!(loader.num_batches(), 6); // 50/8 = 6 (drop_last=true)
        assert!(!loader.is_empty());
        assert!(loader.is_resident());
    }

    #[test]
    fn test_empty_dataset_errors() {
        struct Empty;
        impl DataSet for Empty {
            fn len(&self) -> usize { 0 }
            fn get(&self, _: usize) -> Result<Vec<Tensor>> { unreachable!() }
        }

        let result = DataLoader::from_dataset(Empty)
            .batch_size(10)
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_batch_size_errors() {
        let data = make_data(10);
        let result = DataLoader::from_dataset(data)
            .batch_size(0)
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_batch_size_larger_than_dataset() {
        let data = make_data(5);
        let mut loader = DataLoader::from_dataset(data)
            .batch_size(100)
            .drop_last(false)
            .build()
            .unwrap();

        let batches: Vec<Batch> = loader.epoch(0).map(|b| b.unwrap()).collect();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0][0].shape(), &[5, 4]);
    }

    #[test]
    fn test_batch_size_larger_than_dataset_drop_last() {
        let data = make_data(5);
        let mut loader = DataLoader::from_dataset(data)
            .batch_size(100)
            .drop_last(true)
            .build()
            .unwrap();

        // 5 < 100, so the only batch is incomplete -> dropped
        let batches: Vec<Batch> = loader.epoch(0).map(|b| b.unwrap()).collect();
        assert_eq!(batches.len(), 0);
    }

    #[test]
    fn test_device_aware_loading() {
        let data = make_cpu_data_for_device(20);
        let dev = test_device();
        let mut loader = DataLoader::from_dataset(data)
            .batch_size(5)
            .device(dev)
            .build()
            .unwrap();

        assert_eq!(loader.device(), dev);

        let b = loader.epoch(0).next().unwrap().unwrap();
        assert_eq!(b[0].device(), dev);
        assert_eq!(b[1].device(), dev);
    }

    #[test]
    fn test_multi_target_dataset() {
        struct FbrlLike {
            images: Tensor,
            letters: Tensor,
            cases: Tensor,
            origins: Tensor,
        }

        impl DataSet for FbrlLike {
            fn len(&self) -> usize { self.images.shape()[0] as usize }
            fn get(&self, i: usize) -> Result<Vec<Tensor>> {
                Ok(vec![
                    self.images.select(0, i as i64)?,
                    self.letters.select(0, i as i64)?,
                    self.cases.select(0, i as i64)?,
                    self.origins.select(0, i as i64)?,
                ])
            }
        }

        let opts = TensorOptions { dtype: DType::Float32, device: Device::CPU };
        let data = FbrlLike {
            images: Tensor::randn(&[16, 3, 8, 8], opts).unwrap(),
            letters: Tensor::randn(&[16, 26], opts).unwrap(),
            cases: Tensor::randn(&[16, 2], opts).unwrap(),
            origins: Tensor::randn(&[16, 5], opts).unwrap(),
        };

        let mut loader = DataLoader::from_dataset(data)
            .batch_size(4)
            .build()
            .unwrap();

        let b = loader.epoch(0).next().unwrap().unwrap();
        assert_eq!(b.len(), 4);
        assert_eq!(b[0].shape(), &[4, 3, 8, 8]); // images
        assert_eq!(b[1].shape(), &[4, 26]);        // letters
        assert_eq!(b[2].shape(), &[4, 2]);          // cases
        assert_eq!(b[3].shape(), &[4, 5]);          // origins
    }

    // -- Streaming mode tests -------------------------------------------------

    #[test]
    fn test_streaming_basic_epoch() {
        let data = make_data(20);
        let mut loader = DataLoader::from_dataset(data)
            .batch_size(5)
            .streaming()
            .build()
            .unwrap();

        assert!(!loader.is_resident());

        let batches: Vec<Batch> = loader.epoch(0).map(|b| b.unwrap()).collect();
        assert_eq!(batches.len(), 4);
        for b in &batches {
            assert_eq!(b.len(), 2);
            assert_eq!(b[0].shape(), &[5, 4]);
            assert_eq!(b[1].shape(), &[5, 2]);
        }
    }

    #[test]
    fn test_streaming_drop_last() {
        let data = make_data(22);
        let mut loader = DataLoader::from_dataset(data)
            .batch_size(5)
            .drop_last(true)
            .streaming()
            .build()
            .unwrap();

        let batches: Vec<Batch> = loader.epoch(0).map(|b| b.unwrap()).collect();
        assert_eq!(batches.len(), 4); // 22/5 = 4, drop 2
    }

    #[test]
    fn test_streaming_drop_last_false() {
        let data = make_data(22);
        let mut loader = DataLoader::from_dataset(data)
            .batch_size(5)
            .drop_last(false)
            .streaming()
            .build()
            .unwrap();

        let batches: Vec<Batch> = loader.epoch(0).map(|b| b.unwrap()).collect();
        assert_eq!(batches.len(), 5); // 4 full + 1 partial
        assert_eq!(batches[4][0].shape(), &[2, 4]);
    }

    #[test]
    fn test_streaming_all_samples_visited() {
        let mut loader = DataLoader::from_dataset(SequentialData { n: 10 })
            .batch_size(3)
            .drop_last(false)
            .streaming()
            .build()
            .unwrap();

        let mut vals: Vec<f32> = loader
            .epoch(0)
            .flat_map(|b| {
                let b = b.unwrap();
                b[0].to_f32_vec().unwrap()
            })
            .collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(vals, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_streaming_multiple_epochs() {
        let mut loader = DataLoader::from_dataset(SequentialData { n: 20 })
            .batch_size(20)
            .drop_last(false)
            .streaming()
            .build()
            .unwrap();

        let e0: Vec<f32> = loader.epoch(0).next().unwrap().unwrap()[0]
            .to_f32_vec()
            .unwrap();
        let e1: Vec<f32> = loader.epoch(1).next().unwrap().unwrap()[0]
            .to_f32_vec()
            .unwrap();

        // Different epochs should produce different orderings
        assert_ne!(e0, e1);

        // But same number of samples
        assert_eq!(e0.len(), 20);
        assert_eq!(e1.len(), 20);
    }

    #[test]
    fn test_dataset_panic_is_an_err_batch_not_a_dead_loader() {
        // A panic in user dataset code must surface as an Err batch
        // carrying the panic message — and the worker must survive it,
        // so the NEXT epoch still works. Previously the worker thread
        // died (payload discarded) and every later epoch failed with a
        // generic "worker stopped unexpectedly".
        struct PanicsOnce {
            n: usize,
        }
        impl DataSet for PanicsOnce {
            fn len(&self) -> usize {
                self.n
            }
            fn get(&self, index: usize) -> Result<Vec<Tensor>> {
                if index == 7 {
                    panic!("user dataset bug: bad record {index}");
                }
                Ok(vec![Tensor::from_f32(&[index as f32], &[1], Device::CPU)?])
            }
        }

        let mut loader = DataLoader::from_dataset(PanicsOnce { n: 8 })
            .batch_size(4)
            .shuffle(false)
            .drop_last(false)
            .sample_cache(false)
            .streaming()
            .build()
            .unwrap();

        // Epoch 0, sequential: batch 0 = samples 0..4 (fine), batch 1
        // hits sample 7 -> Err carrying the panic message.
        let results: Vec<Result<Batch>> = loader.epoch(0).collect();
        assert!(results.iter().any(|r| matches!(
            r,
            Err(e) if e.to_string().contains("panicked") && e.to_string().contains("bad record 7")
        )), "expected a panic-carrying Err batch, got: {:?}",
            results.iter().map(|r| r.as_ref().map(|_| "ok").map_err(|e| e.to_string())).collect::<Vec<_>>());

        // The worker survived: the next epoch's clean batch still arrives.
        let first = loader.epoch(1).next().unwrap();
        assert!(first.is_ok(), "worker must survive a dataset panic: {:?}", first.err());
    }

    #[test]
    fn test_streaming_sequential() {
        let mut loader = DataLoader::from_dataset(SequentialData { n: 10 })
            .batch_size(3)
            .shuffle(false)
            .drop_last(false)
            .streaming()
            .build()
            .unwrap();

        let vals: Vec<f32> = loader
            .epoch(0)
            .flat_map(|b| b.unwrap()[0].to_f32_vec().unwrap())
            .collect();
        assert_eq!(vals, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_streaming_multi_target() {
        struct Multi {
            a: Tensor,
            b: Tensor,
            c: Tensor,
        }
        impl DataSet for Multi {
            fn len(&self) -> usize { self.a.shape()[0] as usize }
            fn get(&self, i: usize) -> Result<Vec<Tensor>> {
                Ok(vec![
                    self.a.select(0, i as i64)?,
                    self.b.select(0, i as i64)?,
                    self.c.select(0, i as i64)?,
                ])
            }
        }

        let opts = TensorOptions { dtype: DType::Float32, device: Device::CPU };
        let data = Multi {
            a: Tensor::randn(&[12, 4], opts).unwrap(),
            b: Tensor::randn(&[12, 8], opts).unwrap(),
            c: Tensor::randn(&[12, 2], opts).unwrap(),
        };

        let mut loader = DataLoader::from_dataset(data)
            .batch_size(4)
            .streaming()
            .build()
            .unwrap();

        let b = loader.epoch(0).next().unwrap().unwrap();
        assert_eq!(b.len(), 3);
        assert_eq!(b[0].shape(), &[4, 4]);
        assert_eq!(b[1].shape(), &[4, 8]);
        assert_eq!(b[2].shape(), &[4, 2]);
    }

    #[test]
    fn test_streaming_drop_mid_epoch() {
        let data = make_data(100);
        let mut loader = DataLoader::from_dataset(data)
            .batch_size(10)
            .streaming()
            .build()
            .unwrap();

        // Consume only 2 out of 10 batches, then drop the iterator
        {
            let mut iter = loader.epoch(0);
            let _ = iter.next().unwrap().unwrap();
            let _ = iter.next().unwrap().unwrap();
            // drop iter here
        }

        // Should be able to start a new epoch without issues
        let batches: Vec<Batch> = loader.epoch(1).map(|b| b.unwrap()).collect();
        assert_eq!(batches.len(), 10);
    }

    // -- Named Batch tests ---------------------------------------------------

    #[test]
    fn test_named_batch_via_loader() {
        let data = make_data(20);
        let mut loader = DataLoader::from_dataset(data)
            .batch_size(5)
            .names(&["input", "target"])
            .build()
            .unwrap();

        let b = loader.epoch(0).next().unwrap().unwrap();
        assert_eq!(b.names(), &["input", "target"]);
        assert_eq!(b["input"].shape(), &[5, 4]);
        assert_eq!(b["target"].shape(), &[5, 2]);
        assert!(b.has("input"));
        assert!(b.has("target"));
        assert!(!b.has("missing"));
    }

    #[test]
    fn test_named_batch_streaming() {
        let data = make_data(20);
        let mut loader = DataLoader::from_dataset(data)
            .batch_size(5)
            .names(&["x", "y"])
            .streaming()
            .build()
            .unwrap();

        let b = loader.epoch(0).next().unwrap().unwrap();
        assert_eq!(b.names(), &["x", "y"]);
        assert_eq!(b["x"].shape(), &[5, 4]);
        assert_eq!(b["y"].shape(), &[5, 2]);
    }

    #[test]
    fn test_names_count_mismatch_errors() {
        let data = make_data(10);
        let result = DataLoader::from_dataset(data)
            .batch_size(5)
            .names(&["only_one"])
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_auto_names_when_unspecified() {
        let data = make_data(10);
        let mut loader = DataLoader::from_dataset(data)
            .batch_size(5)
            .build()
            .unwrap();

        assert_eq!(loader.names(), &["0", "1"]);
        let b = loader.epoch(0).next().unwrap().unwrap();
        assert_eq!(b["0"].shape(), &[5, 4]);
        assert_eq!(b["1"].shape(), &[5, 2]);
    }

    // -- Graph + DataLoader integration tests --------------------------------

    #[test]
    fn test_graph_set_data_loader_single_gpu() {
        use crate::graph::FlowBuilder;
        use crate::nn::{Adam, Linear, Module, ReLU, mse_loss};

        let model = FlowBuilder::from(Linear::new(4, 8).unwrap())
            .through(ReLU::new())
            .through(Linear::new(8, 2).unwrap())
            .build()
            .unwrap();

        let opts = TensorOptions { dtype: DType::Float32, device: Device::CPU };
        struct TrainData { x: Tensor, y: Tensor }
        impl super::DataSet for TrainData {
            fn len(&self) -> usize { self.x.shape()[0] as usize }
            fn get(&self, i: usize) -> Result<Vec<Tensor>> {
                Ok(vec![
                    self.x.select(0, i as i64)?,
                    self.y.select(0, i as i64)?,
                ])
            }
        }

        let data = TrainData {
            x: Tensor::randn(&[20, 4], opts).unwrap(),
            y: Tensor::randn(&[20, 2], opts).unwrap(),
        };

        let loader = DataLoader::from_dataset(data)
            .batch_size(5)
            .names(&["input", "target"])
            .build()
            .unwrap();

        model.set_data_loader(loader, "input").unwrap();
        model.set_optimizer(|p| Adam::new(p, 0.01));
        model.set_training(true);

        // Snapshot params before training
        let params_before: Vec<f32> = model
            .parameters()
            .iter()
            .flat_map(|p| p.variable.data().to_f32_vec().unwrap())
            .collect();

        // One epoch of training
        let iter = model.epoch(0);
        let active = iter.activate();
        let mut batch_count = 0;
        for batch_result in active {
            let b = batch_result.unwrap();
            assert!(b.has("input"));
            assert!(b.has("target"));
            let out = model.forward_batch(&b).unwrap();
            let target = crate::autograd::Variable::new(b["target"].clone(), false);
            let loss = mse_loss(&out, &target).unwrap();
            loss.backward().unwrap();
            model.step().unwrap();
            batch_count += 1;
        }

        assert_eq!(batch_count, 4); // 20 / 5 = 4

        // Params should have changed
        let params_after: Vec<f32> = model
            .parameters()
            .iter()
            .flat_map(|p| p.variable.data().to_f32_vec().unwrap())
            .collect();

        let changed = params_before
            .iter()
            .zip(&params_after)
            .any(|(a, b)| (a - b).abs() > 1e-8);
        assert!(changed, "parameters should change after training");
    }

    #[test]
    fn test_graph_data_num_batches() {
        use crate::graph::FlowBuilder;
        use crate::nn::Linear;

        let model = FlowBuilder::from(Linear::new(4, 2).unwrap())
            .build()
            .unwrap();

        let data = make_data(20);
        let loader = DataLoader::from_dataset(data)
            .batch_size(5)
            .names(&["x", "y"])
            .build()
            .unwrap();

        model.set_data_loader(loader, "x").unwrap();
        assert_eq!(model.data_num_batches(), 4);
        assert_eq!(model.data_batch_size(), 5);
    }

    #[test]
    fn test_set_data_loader_rejected_while_epoch_iterator_active() {
        // Regression: replacing the loader mid-iteration used to drop it out
        // from under the live epoch iterator (use-after-free); it must now
        // fail loudly while the iterator holds the loader cell.
        use crate::graph::FlowBuilder;
        use crate::nn::Linear;

        let model = FlowBuilder::from(Linear::new(4, 2).unwrap())
            .build()
            .unwrap();
        let loader = DataLoader::from_dataset(make_data(20))
            .batch_size(5)
            .names(&["x", "y"])
            .build()
            .unwrap();
        model.set_data_loader(loader, "x").unwrap();

        let active = model.epoch(0).activate();

        // Metadata reads stay available while the iterator is active
        // (cached at bind time, not routed through the loader cell).
        assert_eq!(model.data_num_batches(), 4);
        assert_eq!(model.data_batch_size(), 5);

        let replacement = DataLoader::from_dataset(make_data(10))
            .batch_size(5)
            .names(&["x", "y"])
            .build()
            .unwrap();
        let err = model.set_data_loader(replacement, "x");
        assert!(err.is_err(), "mid-iteration replace must be rejected");
        assert!(
            err.unwrap_err().to_string().contains("epoch iterator is active"),
            "error should name the cause"
        );

        // Dropping the iterator releases the lease; rebinding works again.
        drop(active);
        let replacement = DataLoader::from_dataset(make_data(10))
            .batch_size(5)
            .names(&["x", "y"])
            .build()
            .unwrap();
        model.set_data_loader(replacement, "x").unwrap();
        assert_eq!(model.data_num_batches(), 2);
    }

    #[test]
    #[should_panic(expected = "already active")]
    fn test_second_active_epoch_iterator_panics() {
        // Regression: two active iterators used to alias `&mut` into the
        // same DataLoader; the second activation must fail loudly.
        use crate::graph::FlowBuilder;
        use crate::nn::Linear;

        let model = FlowBuilder::from(Linear::new(4, 2).unwrap())
            .build()
            .unwrap();
        let loader = DataLoader::from_dataset(make_data(20))
            .batch_size(5)
            .names(&["x", "y"])
            .build()
            .unwrap();
        model.set_data_loader(loader, "x").unwrap();

        let _first = model.epoch(0).activate();
        let _second = model.epoch(0).activate(); // must panic
    }

    #[test]
    fn test_set_data_loader_invalid_input_name() {
        use crate::graph::FlowBuilder;
        use crate::nn::Linear;

        let model = FlowBuilder::from(Linear::new(4, 2).unwrap())
            .build()
            .unwrap();

        let data = make_data(10);
        let loader = DataLoader::from_dataset(data)
            .batch_size(5)
            .names(&["x", "y"])
            .build()
            .unwrap();

        let result = model.set_data_loader(loader, "missing");
        assert!(result.is_err());
    }

    #[test]
    fn test_scatter_fallback_without_data_loader() {
        // Module::forward(&Variable) still works without set_data_loader
        use crate::graph::FlowBuilder;
        use crate::nn::{Linear, Module};

        let model = FlowBuilder::from(Linear::new(4, 2).unwrap())
            .build()
            .unwrap();

        let x = crate::autograd::Variable::new(
            Tensor::randn(&[3, 4], Default::default()).unwrap(),
            false,
        );
        let out = model.forward(&x).unwrap();
        assert_eq!(out.shape(), &[3, 2]);
    }

    // -- Adaptive prefetch tests ----------------------------------------------

    #[test]
    fn test_prefetch_depth_from_vram_cpu() {
        // CPU always returns 2 (double-buffer)
        let depth = prefetch_depth_from_vram(100, 32, Device::CPU, 0.90, 0);
        assert_eq!(depth, 2);
    }

    #[test]
    fn test_prefetch_depth_from_vram_zero_batch() {
        let depth = prefetch_depth_from_vram(0, 32, Device::CPU, 0.90, 0);
        assert_eq!(depth, 2);
    }

    #[test]
    fn test_prefetch_depth_from_vram_zero_bytes() {
        let depth = prefetch_depth_from_vram(100, 0, Device::CPU, 0.90, 0);
        assert_eq!(depth, 2);
    }

    #[test]
    fn test_streaming_prefetch_depth_and_resize() {
        let data = SequentialData { n: 100 };
        let mut loader = DataLoader::from_dataset(data)
            .batch_size(10)
            .streaming()
            .build()
            .unwrap();

        // Should be in streaming mode
        assert!(!loader.is_resident());

        // Initial depth should be at least 2
        let initial = loader.prefetch_depth();
        assert!(initial >= 2, "initial depth should be >= 2, got {initial}");

        // Manual set
        loader.set_prefetch_depth(42);
        assert_eq!(loader.prefetch_depth(), 42);

        // Reset to something sensible
        loader.set_prefetch_depth(4);
        assert_eq!(loader.prefetch_depth(), 4);
    }

    #[test]
    fn test_augment_schedules_k_views_per_sample() {
        // augment(2) on 8 samples, batch 4: an epoch is 16 picks = 4
        // batches, and every sample id appears exactly twice — the
        // realized-work constant is the augmented permutation length.
        for streaming in [false, true] {
            let mut b = DataLoader::from_dataset(SequentialData { n: 8 })
                .batch_size(4)
                .augment(2);
            if streaming {
                b = b.streaming();
            }
            let mut loader = b.build().unwrap();
            assert_eq!(loader.is_resident(), !streaming);

            let mut counts: std::collections::HashMap<usize, usize> =
                std::collections::HashMap::new();
            let mut batches = 0;
            for batch in loader.epoch(0) {
                for v in batch.unwrap()[0].to_f64_vec().unwrap() {
                    *counts.entry(v as usize).or_insert(0) += 1;
                }
                batches += 1;
            }
            assert_eq!(batches, 4, "streaming={streaming}: 16 picks / 4");
            assert_eq!(counts.len(), 8, "streaming={streaming}");
            assert!(
                counts.values().all(|&c| c == 2),
                "streaming={streaming}: each sample exactly k views: {counts:?}"
            );
        }
    }

    #[test]
    fn test_transform_keys_views_and_reproduces() {
        // The transform derives each view from its PickKey: with
        // augment(2) and offset = repeat*100, every sample shows up
        // once raw and once shifted — and the whole realized stream is
        // identical across loaders (deterministic augmentation).
        let build = || {
            DataLoader::from_dataset(SequentialData { n: 8 })
                .batch_size(4)
                .streaming()
                .augment(2)
                .transform(|rows, keys| {
                    let offs: Vec<f32> =
                        keys.iter().map(|k| k.repeat as f32 * 100.0).collect();
                    let o = Tensor::from_f32(
                        &offs,
                        &[offs.len() as i64, 1],
                        Device::CPU,
                    )?;
                    Ok(vec![rows[0].add(&o)?])
                })
                .build()
                .unwrap()
        };
        let collect = |loader: &mut DataLoader| -> Vec<f64> {
            let mut out = Vec::new();
            for batch in loader.epoch(0) {
                out.extend(batch.unwrap()[0].to_f64_vec().unwrap());
            }
            out
        };
        let (mut l1, mut l2) = (build(), build());
        let (v1, v2) = (collect(&mut l1), collect(&mut l2));
        assert_eq!(v1, v2, "same config = identical realized stream");

        let mut got: Vec<i64> = v1.iter().map(|&x| x as i64).collect();
        got.sort_unstable();
        let mut expected: Vec<i64> = (0..8).flat_map(|i| [i, i + 100]).collect();
        expected.sort_unstable();
        assert_eq!(got, expected, "one raw view + one shifted view per sample");
    }

    #[test]
    fn test_transform_never_writes_back_to_the_tiers() {
        // An in-place transform on the delivered rows must not corrupt
        // the retained raw samples: batch assembly materializes fresh
        // storage, so every epoch shows exactly ONE application, never
        // accumulation through the sample cache.
        let mut loader = DataLoader::from_dataset(SequentialData { n: 8 })
            .batch_size(4)
            .streaming()
            .transform(|rows, _keys| {
                rows[0].add_scalar_(1000.0)?;
                Ok(rows)
            })
            .build()
            .unwrap();
        for epoch in 0..3 {
            let mut vals: Vec<i64> = Vec::new();
            for batch in loader.epoch(epoch) {
                vals.extend(
                    batch.unwrap()[0]
                        .to_f64_vec()
                        .unwrap()
                        .into_iter()
                        .map(|v| v as i64),
                );
            }
            vals.sort_unstable();
            assert_eq!(
                vals,
                (1000..1008).collect::<Vec<_>>(),
                "epoch {epoch}: exactly one application on raw bytes"
            );
        }
    }

    #[test]
    fn test_augment_rejects_custom_sampler() {
        let result = DataLoader::from_dataset(SequentialData { n: 8 })
            .batch_size(4)
            .sampler(Box::new(SequentialSampler::new(8)))
            .augment(2)
            .build();
        match result {
            Err(e) => assert!(format!("{e}").contains("augment")),
            Ok(_) => panic!("augment + custom sampler must error loudly"),
        }
    }

    #[test]
    fn test_resident_prefetch_depth_is_zero() {
        let data = SequentialData { n: 20 };
        let mut loader = DataLoader::from_dataset(data)
            .batch_size(5)
            .build()
            .unwrap();

        // CPU defaults to resident
        assert!(loader.is_resident());
        assert_eq!(loader.prefetch_depth(), 0);

        // set/auto_resize are no-ops for resident
        loader.set_prefetch_depth(100);
        assert_eq!(loader.prefetch_depth(), 0);

        let depth = loader.auto_resize();
        assert_eq!(depth, 0);
    }

    #[test]
    fn test_streaming_auto_resize_cpu() {
        let data = SequentialData { n: 100 };
        let mut loader = DataLoader::from_dataset(data)
            .batch_size(10)
            .streaming()
            .build()
            .unwrap();

        // On CPU, auto_resize returns 2 (just double-buffer)
        let depth = loader.auto_resize();
        assert_eq!(depth, 2);
    }

    #[test]
    fn test_streaming_epoch_after_resize() {
        // Verify that changing prefetch depth doesn't break iteration
        let data = SequentialData { n: 50 };
        let mut loader = DataLoader::from_dataset(data)
            .batch_size(10)
            .streaming()
            .build()
            .unwrap();

        loader.set_prefetch_depth(8);

        let mut count = 0;
        for batch in loader.epoch(0) {
            let b = batch.unwrap();
            assert_eq!(b[0].shape(), &[10, 1]);
            count += 1;
        }
        assert_eq!(count, 5);

        // Change depth between epochs
        loader.set_prefetch_depth(2);
        count = 0;
        for batch in loader.epoch(1) {
            batch.unwrap();
            count += 1;
        }
        assert_eq!(count, 5);
    }

    #[test]
    fn test_vram_max_usage_builder() {
        let data = SequentialData { n: 100 };
        let loader = DataLoader::from_dataset(data)
            .batch_size(10)
            .vram_max_usage(0.80) // 80% of total VRAM
            .streaming()
            .build()
            .unwrap();

        assert!(!loader.is_resident());
        assert!(loader.prefetch_depth() >= 2);
    }

    #[test]
    fn test_vram_max_usage_clamped() {
        let data = SequentialData { n: 100 };
        // Extreme values get clamped to [0.50, 0.99]
        let loader = DataLoader::from_dataset(data)
            .batch_size(10)
            .vram_max_usage(0.10) // below min, clamped to 0.50
            .streaming()
            .build()
            .unwrap();

        assert!(!loader.is_resident());
    }

    // -- El Che data routing tests (CPU) --------------------------------------

    #[test]
    fn test_el_che_counts_cell_roundtrip() {
        // Verify Cell<Option<Vec>> semantics for el_che_counts
        let cell: std::cell::Cell<Option<Vec<usize>>> = std::cell::Cell::new(None);
        assert!(cell.take().is_none());

        cell.set(Some(vec![10, 23]));
        let val = cell.take();
        assert_eq!(val, Some(vec![10, 23]));
        // After take, cell is None
        assert!(cell.take().is_none());
    }

    #[test]
    fn test_el_che_batches_cell_roundtrip() {
        // Verify Cell semantics for pending_el_che_batches
        let cell: std::cell::Cell<Option<Vec<Vec<Vec<Tensor>>>>> = std::cell::Cell::new(None);
        assert!(cell.take().is_none());

        let t = Tensor::zeros(&[2, 3], Default::default()).unwrap();
        let batches = vec![vec![vec![t.clone()]], vec![vec![t]]];
        cell.set(Some(batches));
        let val = cell.take();
        assert!(val.is_some());
        let batches = val.unwrap();
        assert_eq!(batches.len(), 2); // 2 ranks
        assert_eq!(batches[0].len(), 1); // 1 batch on rank 0
        assert_eq!(batches[1].len(), 1); // 1 batch on rank 1
    }

    #[test]
    fn test_el_che_clamping_proportional() {
        // Test the clamping logic in next_el_che
        let counts = [10usize, 23];
        let total: usize = counts.iter().sum(); // 33
        let remaining = 20usize;

        // Scale proportionally
        let scale = remaining as f64 / total as f64;
        let mut clamped: Vec<usize> = counts.iter()
            .map(|&c| (c as f64 * scale).floor() as usize)
            .collect();
        let clamped_total: usize = clamped.iter().sum();
        let mut deficit = remaining.saturating_sub(clamped_total);
        for c in &mut clamped {
            if deficit == 0 { break; }
            *c += 1;
            deficit -= 1;
        }
        let final_total: usize = clamped.iter().sum();
        assert_eq!(final_total, remaining);
        // Proportions roughly preserved
        assert!(clamped[0] < clamped[1], "fast device should still get more");
    }

    // -- Edge case tests ------------------------------------------------------

    #[test]
    fn test_single_item_dataset() {
        let dev = test_device();
        let opts = TensorOptions { dtype: DType::Float32, device: Device::CPU };
        let data = SimpleData {
            x: Tensor::randn(&[1, 4], opts).unwrap(),
            y: Tensor::randn(&[1, 2], opts).unwrap(),
        };

        let mut loader = DataLoader::from_dataset(data)
            .batch_size(1)
            .device(dev)
            .drop_last(false)
            .build()
            .unwrap();

        assert_eq!(loader.len(), 1);
        assert_eq!(loader.num_batches(), 1);

        let batches: Vec<Batch> = loader.epoch(0).map(|b| b.unwrap()).collect();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0][0].shape(), &[1, 4]);
        assert_eq!(batches[0][1].shape(), &[1, 2]);
        assert_eq!(batches[0][0].device(), dev);
    }

    #[test]
    fn test_dataset_smaller_than_batch_no_drop() {
        // 3 items, batch_size=10, drop_last=false -> 1 batch with 3 items
        let dev = test_device();
        let opts = TensorOptions { dtype: DType::Float32, device: Device::CPU };
        let data = SimpleData {
            x: Tensor::randn(&[3, 4], opts).unwrap(),
            y: Tensor::randn(&[3, 2], opts).unwrap(),
        };

        let mut loader = DataLoader::from_dataset(data)
            .batch_size(10)
            .device(dev)
            .drop_last(false)
            .build()
            .unwrap();

        let batches: Vec<Batch> = loader.epoch(0).map(|b| b.unwrap()).collect();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0][0].shape(), &[3, 4]);
        assert_eq!(batches[0][1].shape(), &[3, 2]);
    }

    #[test]
    fn test_dataset_smaller_than_batch_drop_last() {
        // 3 items, batch_size=10, drop_last=true -> 0 batches
        let opts = TensorOptions { dtype: DType::Float32, device: Device::CPU };
        let data = SimpleData {
            x: Tensor::randn(&[3, 4], opts).unwrap(),
            y: Tensor::randn(&[3, 2], opts).unwrap(),
        };

        let mut loader = DataLoader::from_dataset(data)
            .batch_size(10)
            .drop_last(true)
            .build()
            .unwrap();

        let batches: Vec<Batch> = loader.epoch(0).map(|b| b.unwrap()).collect();
        assert_eq!(batches.len(), 0);
    }

    #[test]
    fn test_drop_last_exact_division() {
        // 100 items, batch_size=10, drop_last=true -> exactly 10 batches
        let data = make_data(100);
        let mut loader = DataLoader::from_dataset(data)
            .batch_size(10)
            .drop_last(true)
            .build()
            .unwrap();

        assert_eq!(loader.num_batches(), 10);
        let batches: Vec<Batch> = loader.epoch(0).map(|b| b.unwrap()).collect();
        assert_eq!(batches.len(), 10);
        for b in &batches {
            assert_eq!(b[0].shape(), &[10, 4]);
        }
    }

    #[test]
    fn test_drop_last_with_remainder() {
        // 105 items, batch_size=10, drop_last=true -> 10 batches (5 dropped)
        let data = make_data(105);
        let mut loader = DataLoader::from_dataset(data)
            .batch_size(10)
            .drop_last(true)
            .build()
            .unwrap();

        assert_eq!(loader.num_batches(), 10);
        let batches: Vec<Batch> = loader.epoch(0).map(|b| b.unwrap()).collect();
        assert_eq!(batches.len(), 10);
        for b in &batches {
            assert_eq!(b[0].shape(), &[10, 4]);
        }
    }

    #[test]
    fn test_two_epoch_consistency() {
        // Run two epochs, verify total items seen matches dataset size each time
        let n = 25;
        let mut loader = DataLoader::from_dataset(SequentialData { n })
            .batch_size(7)
            .drop_last(false)
            .build()
            .unwrap();

        for epoch in 0..2 {
            let mut vals: Vec<f32> = loader
                .epoch(epoch)
                .flat_map(|b| b.unwrap()[0].to_f32_vec().unwrap())
                .collect();
            assert_eq!(vals.len(), n, "epoch {epoch}: should see all {n} items");
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let expected: Vec<f32> = (0..n).map(|i| i as f32).collect();
            assert_eq!(vals, expected, "epoch {epoch}: no data lost or duplicated");
        }
    }

    #[test]
    fn test_sequential_sampler_batch_ordering() {
        // With sequential sampler, each batch should contain consecutive indices
        let mut loader = DataLoader::from_dataset(SequentialData { n: 12 })
            .batch_size(4)
            .shuffle(false)
            .drop_last(false)
            .build()
            .unwrap();

        let batches: Vec<Vec<f32>> = loader
            .epoch(0)
            .map(|b| b.unwrap()[0].to_f32_vec().unwrap())
            .collect();

        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0], vec![0.0, 1.0, 2.0, 3.0]);
        assert_eq!(batches[1], vec![4.0, 5.0, 6.0, 7.0]);
        assert_eq!(batches[2], vec![8.0, 9.0, 10.0, 11.0]);
    }

    #[test]
    fn test_empty_iteration_no_leak() {
        // Build a loader, call epoch() but don't consume any items.
        // Should not panic or leak resources.
        let data = make_data(20);
        let mut loader = DataLoader::from_dataset(data)
            .batch_size(5)
            .build()
            .unwrap();

        // Create and immediately drop the epoch iterator
        {
            let _iter = loader.epoch(0);
        }

        // Should still be usable for subsequent epochs
        let batches: Vec<Batch> = loader.epoch(1).map(|b| b.unwrap()).collect();
        assert_eq!(batches.len(), 4);
    }

    #[test]
    fn test_named_and_positional_access() {
        let data = make_data(10);
        let mut loader = DataLoader::from_dataset(data)
            .batch_size(5)
            .names(&["features", "labels"])
            .build()
            .unwrap();

        let b = loader.epoch(0).next().unwrap().unwrap();

        // Positional access
        let by_pos = b[0].shape().to_vec();
        // Named access
        let by_name = b["features"].shape().to_vec();
        assert_eq!(by_pos, by_name);

        let by_pos_1 = b[1].shape().to_vec();
        let by_name_1 = b["labels"].shape().to_vec();
        assert_eq!(by_pos_1, by_name_1);

        // get_named returns Option
        assert!(b.get_named("features").is_some());
        assert!(b.get_named("nonexistent").is_none());
    }

    #[test]
    fn test_multiple_tensors_per_sample() {
        // Dataset returning 3 tensors: input, target, mask
        struct TripleData {
            input: Tensor,
            target: Tensor,
            mask: Tensor,
        }
        impl DataSet for TripleData {
            fn len(&self) -> usize { self.input.shape()[0] as usize }
            fn get(&self, i: usize) -> Result<Vec<Tensor>> {
                Ok(vec![
                    self.input.select(0, i as i64)?,
                    self.target.select(0, i as i64)?,
                    self.mask.select(0, i as i64)?,
                ])
            }
        }

        let dev = test_device();
        let opts = TensorOptions { dtype: DType::Float32, device: Device::CPU };
        let data = TripleData {
            input: Tensor::randn(&[16, 10], opts).unwrap(),
            target: Tensor::randn(&[16, 5], opts).unwrap(),
            mask: Tensor::ones(&[16, 10], opts).unwrap(),
        };

        let mut loader = DataLoader::from_dataset(data)
            .batch_size(4)
            .device(dev)
            .names(&["input", "target", "mask"])
            .build()
            .unwrap();

        assert_eq!(loader.names(), &["input", "target", "mask"]);

        let batches: Vec<Batch> = loader.epoch(0).map(|b| b.unwrap()).collect();
        assert_eq!(batches.len(), 4); // 16 / 4 = 4

        for b in &batches {
            assert_eq!(b.len(), 3);
            assert_eq!(b["input"].shape(), &[4, 10]);
            assert_eq!(b["target"].shape(), &[4, 5]);
            assert_eq!(b["mask"].shape(), &[4, 10]);
            assert_eq!(b["input"].device(), dev);
            assert_eq!(b["target"].device(), dev);
            assert_eq!(b["mask"].device(), dev);
        }
    }

    #[test]
    fn test_exact_size_iterator_with_drop_last() {
        // ExactSizeIterator should report correct len with drop_last
        let data = make_data(23);
        let mut loader = DataLoader::from_dataset(data)
            .batch_size(5)
            .drop_last(true)
            .build()
            .unwrap();

        let iter = loader.epoch(0);
        assert_eq!(iter.len(), 4); // 23/5 = 4 full batches, remainder dropped
    }

    #[test]
    fn test_exact_size_iterator_no_drop_last() {
        let data = make_data(23);
        let mut loader = DataLoader::from_dataset(data)
            .batch_size(5)
            .drop_last(false)
            .build()
            .unwrap();

        let iter = loader.epoch(0);
        assert_eq!(iter.len(), 5); // 4 full + 1 partial
    }

    // ── Depth governor / adaptive sizing ─────────────────────────────

    #[test]
    fn test_governor_streaming_delivers_all_batches_across_epochs() {
        // The in-flight gate must never wedge the pipeline: two full
        // epochs through the governed streaming path deliver every
        // batch, and the per-epoch counters stay coherent.
        let data = make_data(24);
        let mut loader = DataLoader::from_dataset(data)
            .batch_size(4)
            .streaming()
            .build()
            .unwrap();

        for epoch in 0..2 {
            let batches: Vec<Batch> = loader.epoch(epoch).map(|b| b.unwrap()).collect();
            assert_eq!(batches.len(), 6);
        }
        match &loader.inner {
            LoaderInner::Streaming(l) => {
                use std::sync::atomic::Ordering;
                assert_eq!(l.governor.consumed.load(Ordering::Relaxed), 6);
                assert_eq!(l.governor.run_consumed.load(Ordering::Relaxed), 12);
                assert!(l.governor.honest_resize_done.load(Ordering::Relaxed));
            }
            LoaderInner::Resident(_) => panic!("expected streaming loader"),
        }
    }

    #[test]
    fn test_governor_honest_resize_fires_on_second_consumed_batch() {
        // The honest resize is keyed to CONSUMPTION (first training
        // step demonstrably done), not to epoch boundaries: it must
        // fire inside the very first epoch.
        let data = make_data(20);
        let mut loader = DataLoader::from_dataset(data)
            .batch_size(4)
            .streaming()
            .build()
            .unwrap();

        use std::sync::atomic::Ordering;
        let g = std::sync::Arc::clone(loader_governor(&loader));
        let mut iter = loader.epoch(0);
        let _b1 = iter.next().unwrap().unwrap();
        assert!(!g.honest_resize_done.load(Ordering::Relaxed));
        let _b2 = iter.next().unwrap().unwrap();
        assert!(g.honest_resize_done.load(Ordering::Relaxed));
    }

    #[test]
    fn test_governor_iterator_drop_mid_epoch_recovers() {
        // Dropping an epoch iterator mid-epoch sets `abandoned`, which
        // unblocks the worker's gate; the next epoch must run in full.
        let data = make_data(40);
        let mut loader = DataLoader::from_dataset(data)
            .batch_size(4)
            .streaming()
            .prefetch(2)
            .build()
            .unwrap();

        {
            let mut iter = loader.epoch(0);
            let _one = iter.next().unwrap().unwrap();
            // drop mid-epoch
        }
        let batches: Vec<Batch> = loader.epoch(1).map(|b| b.unwrap()).collect();
        assert_eq!(batches.len(), 10);
    }

    #[test]
    fn test_user_pinned_depth_skips_honest_resize() {
        // .prefetch(n) pins the governor: no adaptive resize may fire.
        // The honest-probe latch still must: it marks "a probe now sees
        // step memory", and the VRAM sample pool's one-shot budget
        // decision (`maybe_install`) gates on it — a user-set depth
        // used to leave the latch unset and silently disabled the pool
        // tier for the whole run.
        let data = make_data(20);
        let mut loader = DataLoader::from_dataset(data)
            .batch_size(4)
            .streaming()
            .prefetch(3)
            .build()
            .unwrap();

        let batches: Vec<Batch> = loader.epoch(0).map(|b| b.unwrap()).collect();
        assert_eq!(batches.len(), 5);
        let g = loader_governor(&loader);
        use std::sync::atomic::Ordering;
        assert!(
            g.honest_resize_done.load(Ordering::Relaxed),
            "honest-probe latch must fire under a user-set depth"
        );
        assert_eq!(g.target.load(Ordering::Relaxed), 3, "depth stays pinned");
    }

    fn loader_governor(loader: &DataLoader) -> &std::sync::Arc<crate::data::prefetch::GovernorCtl> {
        match &loader.inner {
            LoaderInner::Streaming(l) => &l.governor,
            LoaderInner::Resident(_) => panic!("expected streaming loader"),
        }
    }

    #[test]
    fn test_initial_fill_target_gradation() {
        // Graduated first-fill discount by information available.
        assert_eq!(initial_fill_target(30, ReserveSource::Bare), 10);
        assert_eq!(initial_fill_target(30, ReserveSource::Auto), 15);
        assert_eq!(initial_fill_target(30, ReserveSource::User), 30);
        // Never zero, even when the budget rounds down to nothing.
        assert_eq!(initial_fill_target(0, ReserveSource::Bare), 1);
        assert_eq!(initial_fill_target(2, ReserveSource::Bare), 1);
    }

    #[test]
    fn test_activation_reserve_user_beats_auto() {
        let data = make_data(20);
        let mut loader = DataLoader::from_dataset(data)
            .batch_size(4)
            .streaming()
            .activation_reserve(1 << 20)
            .build()
            .unwrap();

        // Framework auto-wiring must not override the user's value.
        loader.set_activation_reserve_auto(1 << 30);
        match &loader.inner {
            LoaderInner::Streaming(l) => {
                assert_eq!(l.activation_reserve, 1 << 20);
                assert_eq!(l.reserve_source, ReserveSource::User);
            }
            LoaderInner::Resident(_) => panic!("expected streaming loader"),
        }

        // And auto fills the gap when nothing was declared.
        let data = make_data(20);
        let mut bare = DataLoader::from_dataset(data)
            .batch_size(4)
            .streaming()
            .build()
            .unwrap();
        bare.set_activation_reserve_auto(1 << 30);
        match &bare.inner {
            LoaderInner::Streaming(l) => {
                assert_eq!(l.activation_reserve, 1 << 30);
                assert_eq!(l.reserve_source, ReserveSource::Auto);
            }
            LoaderInner::Resident(_) => panic!("expected streaming loader"),
        }
    }

    #[test]
    fn test_retry_on_oom_policy() {
        use crate::data::prefetch::{retry_on_oom, OOM_RETRY_ATTEMPTS};
        use crate::data::vram_pool::VramSamplePool;

        let mut pool = VramSamplePool::new(Device::CPU, false);

        // Transient OOM: fails twice, then succeeds. Retried through,
        // with the retry ordinal visible to the back-off.
        let mut calls = 0;
        let mut backoffs = Vec::new();
        let out: Result<usize> = retry_on_oom(
            &mut pool,
            |_pool| {
                calls += 1;
                if calls <= 2 {
                    Err(TensorError::new("CUDA out of memory"))
                } else {
                    Ok(7)
                }
            },
            |_pool, attempt| backoffs.push(attempt),
        );
        assert_eq!(out.unwrap(), 7);
        assert_eq!(calls, 3);
        assert_eq!(backoffs, vec![0, 1]);

        // Non-OOM error: no retry, surfaces immediately.
        let mut calls = 0;
        let out: Result<usize> = retry_on_oom(
            &mut pool,
            |_pool| {
                calls += 1;
                Err(TensorError::new("dataset index out of range"))
            },
            |_pool, _attempt| panic!("must not back off on non-OOM errors"),
        );
        assert!(out.is_err());
        assert_eq!(calls, 1);

        // Persistent OOM: bounded retries, then the error surfaces.
        let mut calls = 0;
        let out: Result<usize> = retry_on_oom(
            &mut pool,
            |_pool| {
                calls += 1;
                Err(TensorError::new("CUDA out of memory"))
            },
            |_pool, _attempt| {},
        );
        assert!(out.unwrap_err().is_cuda_oom());
        assert_eq!(calls, OOM_RETRY_ATTEMPTS + 1);
    }

    #[test]
    fn test_sample_cache_budget_anchored_not_ratcheting() {
        const GIB: u64 = 1 << 30;
        let a0 = 100 * GIB; // MemAvailable before any admission
        let r = 0.5;

        // Multi-epoch simulation: the cache fills to its budget each
        // epoch, the next probe sees what is left. The cap must hold at
        // r*A0 — with the held bytes added back AFTER taking the share
        // (held + r*available) it ratcheted toward all of A0 instead.
        let mut held = 0u64;
        for _ in 0..10 {
            let available = a0 - held;
            let budget = sample_cache_budget(available, held, 0, r);
            assert_eq!(budget, (a0 as f64 * r) as u64, "cap must stay anchored");
            held = held.max(budget);
        }
        assert_eq!(held, 50 * GIB, "admissions stop at r*A0, not A0");

        // The ring's slice comes off the top; the share clamps at 0.90;
        // a ring larger than the share saturates to 0 (no admissions).
        assert_eq!(sample_cache_budget(100, 0, 10, 0.5), 40);
        assert_eq!(sample_cache_budget(100, 0, 0, 1.5), 90);
        assert_eq!(sample_cache_budget(10, 0, 100, 0.5), 0);
    }

    #[test]
    fn test_ring_slots_from_ram_budget_math() {
        const GIB: u64 = 1 << 30;
        // 1 MiB/sample x 1024 batch = 1 GiB per batch.
        let per_sample = 1 << 20;
        let bs = 1024;

        // 60 GiB available, share 0.5 -> 30 GiB budget -> 30 ring
        // slots. Total RAM does not enter: only what is actually free
        // is priced (permanent fixtures like pinned VM memory already
        // fall out of MemAvailable).
        let mem = Some(60 * GIB);
        assert_eq!(ring_slots_from_ram(per_sample, bs, 0.50, mem, 100), 30);

        // Capped at the epoch's batch count.
        assert_eq!(ring_slots_from_ram(per_sample, bs, 0.50, mem, 3), 3);

        // Not enough free RAM for even one batch: single-stage.
        let tight = Some(GIB); // 0.5 GiB budget < 1 GiB batch
        assert_eq!(ring_slots_from_ram(per_sample, bs, 0.50, tight, 100), 0);

        // 0.0 disables the reader stage outright, RAM info or not.
        assert_eq!(ring_slots_from_ram(per_sample, bs, 0.0, mem, 100), 0);
        assert_eq!(ring_slots_from_ram(per_sample, bs, 0.0, None, 100), 0);

        // No RAM visibility: conservative fixed ring, epoch-capped.
        assert_eq!(
            ring_slots_from_ram(per_sample, bs, 0.50, None, 100),
            RING_SLOTS_FALLBACK
        );
        assert_eq!(ring_slots_from_ram(per_sample, bs, 0.50, None, 2), 2);

        // Unpriceable batches: same conservative fallback.
        assert_eq!(
            ring_slots_from_ram(0, bs, 0.50, mem, 100),
            RING_SLOTS_FALLBACK
        );

        // Fraction is capped at 0.90 even if a raw value sneaks past
        // the builder clamp.
        assert_eq!(
            ring_slots_from_ram(per_sample, bs, 5.0, mem, 1000),
            54 // 60 GiB available x 0.90 = 54 GiB budget
        );
    }

    /// Batches identifiable by content: one Int64 tensor carrying the
    /// requested indices.
    struct IndexBatch {
        n: usize,
    }

    impl BatchDataSet for IndexBatch {
        fn len(&self) -> usize {
            self.n
        }
        fn get_batch(&self, indices: &[usize]) -> Result<Vec<Tensor>> {
            let v: Vec<i64> = indices.iter().map(|&i| i as i64).collect();
            Ok(vec![Tensor::from_i64(&v, &[v.len() as i64], Device::CPU)?])
        }
    }

    #[test]
    fn test_straggling_sent_increment_cannot_wedge_next_epoch() {
        // Regression for the full-suite `test_streaming_multiple_epochs`
        // wedge: the worker counts `sent` AFTER publishing a batch, so a
        // consumer that consumed the batch, dropped the epoch, and armed
        // the next one inside that window had the straggling increment
        // counted against the FRESH epoch. At `target=1` (the
        // pre-honest-resize CPU depth) `governor_gate` then saw
        // `sent=1, consumed=0` forever — worker spinning in its 1ms
        // sleep, consumer parked in `recv`, deadlock. The fix moves the
        // counter reset into the worker's `StartEpoch` processing,
        // ordered after any straggler by the command channel.
        //
        // Deterministic form of the race: inject the straggler after
        // the consumer-side arm. Old protocol (consumer-side reset)
        // wedges and the recv below times out; new protocol wipes the
        // straggler at `StartEpoch` receipt and the batch arrives.
        use crate::data::prefetch::{GovernorCtl, PrefetchWorker};
        use std::sync::atomic::Ordering;
        use std::sync::Arc;

        let dataset: Arc<dyn BatchDataSet> = Arc::new(IndexBatch { n: 4 });
        let worker = PrefetchWorker::new(Arc::clone(&dataset), Device::CPU, 8, false, 1);
        let governor = Arc::new(GovernorCtl::new(1));

        // Epoch 0: one batch, consumed cleanly.
        governor.begin_epoch(1);
        let rx = worker.start_epoch((0..4).collect(), 4, false, Arc::clone(&governor), 0);
        rx.recv().unwrap().unwrap();
        governor.consumed.fetch_add(1, Ordering::Relaxed);
        drop(rx);

        // Epoch 1: arm, then land the straggling `sent` increment the
        // preempted worker would emit at exactly this point.
        governor.begin_epoch(1);
        governor.sent.fetch_add(1, Ordering::Relaxed);

        let rx = worker.start_epoch((0..4).collect(), 4, false, Arc::clone(&governor), 0);
        let batch = rx.recv_timeout(std::time::Duration::from_secs(30));
        assert!(
            batch.is_ok(),
            "epoch after a straggling sent increment must still deliver \
             (governor gate wedged on leaked in-flight accounting)",
        );
    }

    #[test]
    fn test_two_stage_pipeline_delivers_all_batches_in_order() {
        // Drive the worker directly with ring_slots > 0. The loader
        // only enables the reader ring for CUDA targets (policy), but
        // the mechanism is device-agnostic, so the full two-stage
        // pipeline is exercised here on CPU: reader thread -> ring ->
        // transfer stage -> batch channel, order preserved.
        use crate::data::prefetch::{GovernorCtl, PrefetchWorker};
        use std::sync::atomic::Ordering;
        use std::sync::Arc;

        let dataset: Arc<dyn BatchDataSet> = Arc::new(IndexBatch { n: 10 });
        let worker = PrefetchWorker::new(Arc::clone(&dataset), Device::CPU, 8, false, 1);
        let governor = Arc::new(GovernorCtl::new(4));
        governor.begin_epoch(4);

        let rx = worker.start_epoch((0..10).collect(), 2, true, Arc::clone(&governor), 3);
        let mut batches = Vec::new();
        for _ in 0..5 {
            let batch = rx.recv().unwrap().unwrap();
            governor.consumed.fetch_add(1, Ordering::Relaxed);
            batches.push(batch.tensors[0].to_i64_vec().unwrap());
        }
        assert_eq!(
            batches,
            vec![vec![0, 1], vec![2, 3], vec![4, 5], vec![6, 7], vec![8, 9]]
        );
        assert!(rx.recv().is_err(), "channel closes after the epoch");
    }

    #[test]
    fn test_vram_pool_mixed_assembly_preserves_batch_content() {
        // Pool-enabled worker over content-identifiable batches. Epoch
        // 1 covers only the even indices (fills the pool with them),
        // epoch 2 interleaves even and odd, so on a CUDA device every
        // epoch-2 batch mixes pooled rows (gathered on device) with
        // fresh rows (uploaded) and the stitch must restore caller
        // order exactly. On a CPU device the pool is pass-through and
        // this degrades to a plain pipeline-order test.
        use crate::data::prefetch::{GovernorCtl, PrefetchWorker};
        use std::sync::atomic::Ordering;
        use std::sync::Arc;

        let device = test_device();
        let dataset: Arc<dyn BatchDataSet> = Arc::new(IndexBatch { n: 20 });
        let worker = PrefetchWorker::new(Arc::clone(&dataset), device, 8, true, 1);
        let governor = Arc::new(GovernorCtl::new(4));
        // The honest probe has "fired": the pool may take its budget
        // decision at the first batch.
        governor.honest_resize_done.store(true, Ordering::Relaxed);

        let drain = |indices: Vec<usize>, batch_size: usize| -> Vec<Vec<i64>> {
            governor.begin_epoch(4);
            let rx = worker.start_epoch(indices, batch_size, true, Arc::clone(&governor), 2);
            let mut batches = Vec::new();
            while let Ok(b) = rx.recv() {
                let b = b.unwrap();
                #[cfg(feature = "cuda")]
                if let Some(e) = &b.ready_event {
                    e.synchronize().unwrap();
                }
                governor.consumed.fetch_add(1, Ordering::Relaxed);
                batches.push(b.tensors[0].to_i64_vec().unwrap());
            }
            batches
        };

        // Epoch 1: evens only -> pool holds {0, 2, .., 18} afterwards.
        let evens: Vec<usize> = (0..20).step_by(2).collect();
        let got = drain(evens.clone(), 5);
        assert_eq!(got.concat(), evens.iter().map(|&i| i as i64).collect::<Vec<_>>());

        // Epoch 2: interleaved evens (pooled) and odds (fresh), out of
        // order within each batch.
        let mixed: Vec<usize> = vec![1, 0, 3, 2, 18, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15];
        let got = drain(mixed.clone(), 4);
        assert_eq!(got.concat(), mixed.iter().map(|&i| i as i64).collect::<Vec<_>>());

        // Epoch 3: everything again (odds captured in epoch 2 now hit).
        let all: Vec<usize> = (0..20).collect();
        let got = drain(all.clone(), 4);
        assert_eq!(got.concat(), all.iter().map(|&i| i as i64).collect::<Vec<_>>());
    }

    #[test]
    fn test_two_stage_respects_drop_last() {
        use crate::data::prefetch::{GovernorCtl, PrefetchWorker};
        use std::sync::atomic::Ordering;
        use std::sync::Arc;

        let dataset: Arc<dyn BatchDataSet> = Arc::new(IndexBatch { n: 10 });
        let worker = PrefetchWorker::new(Arc::clone(&dataset), Device::CPU, 8, false, 1);
        let governor = Arc::new(GovernorCtl::new(8));
        governor.begin_epoch(8);

        // drop_last = false: the short remainder batch comes through.
        let rx = worker.start_epoch((0..10).collect(), 3, false, Arc::clone(&governor), 2);
        let mut lens = Vec::new();
        while let Ok(b) = rx.recv() {
            governor.consumed.fetch_add(1, Ordering::Relaxed);
            lens.push(b.unwrap().tensors[0].shape()[0]);
        }
        assert_eq!(lens, vec![3, 3, 3, 1]);

        // drop_last = true: the remainder is dropped by the reader.
        governor.begin_epoch(8);
        let rx = worker.start_epoch((0..10).collect(), 3, true, Arc::clone(&governor), 2);
        let mut count = 0;
        while let Ok(b) = rx.recv() {
            governor.consumed.fetch_add(1, Ordering::Relaxed);
            assert_eq!(b.unwrap().tensors[0].shape()[0], 3);
            count += 1;
        }
        assert_eq!(count, 3);
    }

    #[test]
    fn test_two_stage_abandoned_epoch_unwinds_and_recovers() {
        // Abandoning an epoch mid-way must unwind BOTH stages (the
        // transfer loop breaks at the gate, dropping the ring unwinds
        // the reader), and the same worker must then run a fresh epoch
        // in full.
        use crate::data::prefetch::{GovernorCtl, PrefetchWorker};
        use std::sync::atomic::Ordering;
        use std::sync::Arc;

        let dataset: Arc<dyn BatchDataSet> = Arc::new(IndexBatch { n: 20 });
        let worker = PrefetchWorker::new(Arc::clone(&dataset), Device::CPU, 8, false, 1);
        let governor = Arc::new(GovernorCtl::new(4));
        governor.begin_epoch(4);

        let rx = worker.start_epoch((0..20).collect(), 2, true, Arc::clone(&governor), 2);
        let _one = rx.recv().unwrap().unwrap();
        governor.consumed.fetch_add(1, Ordering::Relaxed);
        governor.abandoned.store(true, Ordering::Relaxed);
        drop(rx);

        // Fresh epoch on the same worker delivers everything. The cmd
        // channel serializes epochs, so this also proves the previous
        // epoch's reader thread was joined, not orphaned.
        governor.begin_epoch(4);
        let rx = worker.start_epoch((0..10).collect(), 2, true, Arc::clone(&governor), 2);
        let mut count = 0;
        while let Ok(b) = rx.recv() {
            b.unwrap();
            governor.consumed.fetch_add(1, Ordering::Relaxed);
            count += 1;
        }
        assert_eq!(count, 5);
    }

    /// Per-item dataset that counts `get()` calls (shared handle) and
    /// returns the index as the sample value.
    struct CountingData {
        n: usize,
        calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    impl DataSet for CountingData {
        fn len(&self) -> usize {
            self.n
        }
        fn get(&self, index: usize) -> Result<Vec<Tensor>> {
            self.calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(vec![Tensor::from_f32(&[index as f32], &[1], Device::CPU)?])
        }
    }

    fn loader_sample_cache(
        loader: &DataLoader,
    ) -> std::sync::Arc<crate::data::sample_cache::SampleCache> {
        match &loader.inner {
            LoaderInner::Streaming(l) => {
                std::sync::Arc::clone(l.sample_cache.as_ref().expect("cache wired"))
            }
            LoaderInner::Resident(_) => panic!("expected streaming loader"),
        }
    }

    #[test]
    fn test_sample_cache_serves_later_epochs_from_ram() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let calls = std::sync::Arc::new(AtomicUsize::new(0));
        let mut loader = DataLoader::from_dataset(CountingData {
            n: 12,
            calls: std::sync::Arc::clone(&calls),
        })
        .batch_size(4)
        .streaming()
        .build()
        .unwrap();
        let cache = loader_sample_cache(&loader);

        // Epoch 0 populates read-through: build probe (1) + 12 samples.
        let batches: Vec<Batch> = loader.epoch(0).map(|b| b.unwrap()).collect();
        assert_eq!(batches.len(), 3);
        assert_eq!(calls.load(Ordering::Relaxed), 13);

        if cache.bytes() == 0 {
            // Host genuinely has no RAM headroom under the cap right
            // now; admission legitimately stayed closed. Nothing to
            // assert about hits.
            eprintln!("skipping cache-hit assertions: no RAM headroom on this host");
            return;
        }

        // Later epochs read from the cache: zero new dataset calls.
        for epoch in 1..3 {
            let batches: Vec<Batch> = loader.epoch(epoch).map(|b| b.unwrap()).collect();
            assert_eq!(batches.len(), 3);
        }
        assert_eq!(calls.load(Ordering::Relaxed), 13, "epochs 1-2 fully cached");
    }

    #[test]
    fn test_sample_cache_off_switch_refetches_every_epoch() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let calls = std::sync::Arc::new(AtomicUsize::new(0));
        let mut loader = DataLoader::from_dataset(CountingData {
            n: 8,
            calls: std::sync::Arc::clone(&calls),
        })
        .batch_size(4)
        .streaming()
        .sample_cache(false)
        .build()
        .unwrap();

        match &loader.inner {
            LoaderInner::Streaming(l) => assert!(l.sample_cache.is_none()),
            LoaderInner::Resident(_) => panic!("expected streaming loader"),
        }

        for epoch in 0..2 {
            let batches: Vec<Batch> = loader.epoch(epoch).map(|b| b.unwrap()).collect();
            assert_eq!(batches.len(), 2);
        }
        // Build probe (1) + 8 per epoch, nothing retained.
        assert_eq!(calls.load(Ordering::Relaxed), 17);
    }

    #[test]
    fn test_sample_cache_content_identical_across_epochs() {
        use std::sync::atomic::AtomicUsize;

        let calls = std::sync::Arc::new(AtomicUsize::new(0));
        let mut loader = DataLoader::from_dataset(CountingData {
            n: 8,
            calls,
        })
        .batch_size(4)
        .streaming()
        .sampler(Box::new(SequentialSampler::new(8)))
        .build()
        .unwrap();

        // Sequential order: cached epoch must reproduce the exact
        // same values the read-through epoch delivered.
        for epoch in 0..2 {
            let values: Vec<f64> = loader
                .epoch(epoch)
                .map(|b| b.unwrap()[0].to_f64_vec().unwrap())
                .collect::<Vec<_>>()
                .concat();
            assert_eq!(values, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        }
    }

    #[test]
    fn test_batch_dataset_loader_has_no_sample_cache() {
        let opts = TensorOptions { dtype: DType::Float32, device: Device::CPU };
        let data = PairBatch {
            x: Tensor::randn(&[20, 4], opts).unwrap(),
            y: Tensor::randn(&[20, 2], opts).unwrap(),
        };
        let loader = DataLoader::from_batch_dataset(data)
            .batch_size(4)
            .streaming()
            .build()
            .unwrap();
        match &loader.inner {
            LoaderInner::Streaming(l) => assert!(
                l.sample_cache.is_none(),
                "opaque BatchDataSet loaders stay uncached by design"
            ),
            LoaderInner::Resident(_) => panic!("expected streaming loader"),
        }
    }

    #[test]
    fn test_disk_stage_wires_and_cleans_up() {
        use std::sync::atomic::AtomicUsize;

        let calls = std::sync::Arc::new(AtomicUsize::new(0));
        let mut loader = DataLoader::from_dataset(CountingData {
            n: 8,
            calls,
        })
        .batch_size(4)
        .streaming()
        .disk_stage(1)
        .disk_stage_dir(std::env::temp_dir().join("flodl-stage-tests"))
        .build()
        .unwrap();

        let cache = loader_sample_cache(&loader);
        let path = cache.disk().expect("disk stage attached").path().to_path_buf();
        assert!(path.exists(), "pack file created at build");

        // Epochs deliver normally through the cascaded cache.
        for epoch in 0..2 {
            let batches: Vec<Batch> = loader.epoch(epoch).map(|b| b.unwrap()).collect();
            assert_eq!(batches.len(), 2);
        }

        drop(cache);
        drop(loader);
        assert!(!path.exists(), "pack file removed when the loader drops");
    }

    #[test]
    fn test_disk_stage_errors_without_sample_layer() {
        // Opaque BatchDataSet: no per-sample access to stage.
        let opts = TensorOptions { dtype: DType::Float32, device: Device::CPU };
        let data = PairBatch {
            x: Tensor::randn(&[20, 4], opts).unwrap(),
            y: Tensor::randn(&[20, 2], opts).unwrap(),
        };
        let err = match DataLoader::from_batch_dataset(data)
            .batch_size(4)
            .streaming()
            .disk_stage(1)
            .build()
        {
            Err(e) => e,
            Ok(_) => panic!("expected disk_stage error on BatchDataSet loader"),
        };
        assert!(err.to_string().contains("disk_stage requires the sample layer"));

        // sample_cache(false) disables the tier the stage overflows from.
        use std::sync::atomic::AtomicUsize;
        let calls = std::sync::Arc::new(AtomicUsize::new(0));
        let err = match DataLoader::from_dataset(CountingData { n: 8, calls })
            .batch_size(4)
            .streaming()
            .sample_cache(false)
            .disk_stage(1)
            .build()
        {
            Err(e) => e,
            Ok(_) => panic!("expected disk_stage error with sample_cache(false)"),
        };
        assert!(err.to_string().contains("disk_stage requires the sample layer"));
    }

    #[test]
    fn test_ram_max_usage_builder() {
        let data = make_data(20);
        let loader = DataLoader::from_dataset(data)
            .batch_size(4)
            .streaming()
            .ram_max_usage(0.30)
            .build()
            .unwrap();
        match &loader.inner {
            LoaderInner::Streaming(l) => assert_eq!(l.ram_max_usage, 0.30),
            LoaderInner::Resident(_) => panic!("expected streaming loader"),
        }

        // Clamped to [0.0, 0.90] — 0.0 is a valid "off" value.
        let data = make_data(20);
        let loader = DataLoader::from_dataset(data)
            .batch_size(4)
            .streaming()
            .ram_max_usage(5.0)
            .build()
            .unwrap();
        match &loader.inner {
            LoaderInner::Streaming(l) => assert_eq!(l.ram_max_usage, 0.90),
            LoaderInner::Resident(_) => panic!("expected streaming loader"),
        }
        let data = make_data(20);
        let loader = DataLoader::from_dataset(data)
            .batch_size(4)
            .streaming()
            .ram_max_usage(-1.0)
            .build()
            .unwrap();
        match &loader.inner {
            LoaderInner::Streaming(l) => assert_eq!(l.ram_max_usage, 0.0),
            LoaderInner::Resident(_) => panic!("expected streaming loader"),
        }
    }
