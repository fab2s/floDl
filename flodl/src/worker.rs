//! Background CPU work queue.
//!
//! A single-threaded worker that accepts closures via an mpsc channel and
//! executes them in order. Designed for offloading CPU-bound work (checkpoints,
//! evaluation, file I/O) off the GPU training thread.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Sender};
use std::thread::{self, JoinHandle};

/// A single-threaded background worker that executes closures in order.
///
/// ```ignore
/// let worker = CpuWorker::new();
/// worker.submit(|| {
///     save_checkpoint(&snapshot, "model.fdl").unwrap();
/// });
/// // GPU training continues immediately
/// worker.finish(); // blocks until all queued work completes
/// ```
pub struct CpuWorker {
    tx: Option<Sender<Box<dyn FnOnce() + Send>>>,
    handle: Option<JoinHandle<()>>,
    busy: Arc<AtomicBool>,
}

impl CpuWorker {
    /// Spawn the background worker thread.
    pub fn new() -> Self {
        let (tx, rx) = mpsc::channel::<Box<dyn FnOnce() + Send>>();
        let busy = Arc::new(AtomicBool::new(false));
        let busy2 = busy.clone();

        let handle = thread::spawn(move || {
            for task in rx {
                busy2.store(true, Ordering::Release);
                task();
                busy2.store(false, Ordering::Release);
            }
        });

        CpuWorker {
            tx: Some(tx),
            handle: Some(handle),
            busy,
        }
    }

    /// Submit a closure to run on the background thread.
    pub fn submit<F: FnOnce() + Send + 'static>(&self, f: F) {
        if let Some(ref tx) = self.tx {
            let _ = tx.send(Box::new(f));
        }
    }

    /// Check whether the worker is idle (not currently executing a task).
    ///
    /// Useful for skip-if-busy semantics: only submit a new checkpoint
    /// if the previous one has finished.
    pub fn is_idle(&self) -> bool {
        !self.busy.load(Ordering::Acquire)
    }

    /// Drop the sender and join the worker thread, blocking until all
    /// queued tasks have completed.
    pub fn finish(&mut self) {
        self.tx.take(); // drop sender → rx iterator ends
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

impl Default for CpuWorker {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for CpuWorker {
    fn drop(&mut self) {
        self.finish();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn submit_and_finish() {
        let flag = Arc::new(AtomicBool::new(false));
        let flag2 = flag.clone();

        let mut worker = CpuWorker::new();
        worker.submit(move || {
            flag2.store(true, Ordering::Release);
        });
        worker.finish();

        assert!(flag.load(Ordering::Acquire), "closure should have run");
    }

    #[test]
    fn tasks_execute_in_order() {
        let log = Arc::new(std::sync::Mutex::new(Vec::new()));

        let mut worker = CpuWorker::new();
        for i in 0..5 {
            let log2 = log.clone();
            worker.submit(move || {
                log2.lock().unwrap().push(i);
            });
        }
        worker.finish();

        assert_eq!(*log.lock().unwrap(), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn drop_joins_thread() {
        let counter = Arc::new(AtomicUsize::new(0));
        let counter2 = counter.clone();

        {
            let worker = CpuWorker::new();
            worker.submit(move || {
                counter2.fetch_add(1, Ordering::Release);
            });
            // drop here
        }

        assert_eq!(counter.load(Ordering::Acquire), 1);
    }
}
