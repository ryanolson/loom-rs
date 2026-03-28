use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::sync::Arc;

use super::time::SimTime;

/// Type-erased closure executed at a scheduled virtual time.
pub(crate) type Action = Box<dyn FnOnce() + Send>;

/// Cancellation token for a scheduled action.
pub(crate) struct CancelHandle {
    canceled: Arc<AtomicBool>,
}

impl CancelHandle {
    pub(crate) fn cancel(&self) {
        self.canceled.store(true, AtomicOrdering::Release);
    }
}

/// An action tagged with its scheduled time and insertion order.
struct TimedAction {
    time: SimTime,
    seq: u64,
    canceled: Arc<AtomicBool>,
    action: Action,
}

// Sort by (time ASC, seq ASC). We use `Reverse<TimedAction>` in the max-heap
// so the *smallest* (time, seq) is popped first.
impl Eq for TimedAction {}

impl PartialEq for TimedAction {
    fn eq(&self, other: &Self) -> bool {
        self.time == other.time && self.seq == other.seq
    }
}

impl Ord for TimedAction {
    fn cmp(&self, other: &Self) -> Ordering {
        self.time
            .cmp(&other.time)
            .then_with(|| self.seq.cmp(&other.seq))
    }
}

impl PartialOrd for TimedAction {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Priority queue of scheduled actions ordered by virtual time (FIFO within same time).
pub(crate) struct EventQueue {
    heap: BinaryHeap<std::cmp::Reverse<TimedAction>>,
    seq_counter: u64,
}

impl EventQueue {
    pub(crate) fn new() -> Self {
        Self {
            heap: BinaryHeap::new(),
            seq_counter: 0,
        }
    }

    /// Insert an action to be executed at the given virtual time.
    pub(crate) fn insert(&mut self, time: SimTime, action: Action) {
        let _ = self.insert_cancellable(time, action);
    }

    /// Insert a cancelable action to be executed at the given virtual time.
    pub(crate) fn insert_cancellable(&mut self, time: SimTime, action: Action) -> CancelHandle {
        let seq = self.seq_counter;
        self.seq_counter += 1;
        let canceled = Arc::new(AtomicBool::new(false));
        self.heap.push(std::cmp::Reverse(TimedAction {
            time,
            seq,
            canceled: canceled.clone(),
            action,
        }));
        CancelHandle { canceled }
    }

    fn discard_canceled_head(&mut self) {
        while self
            .heap
            .peek()
            .is_some_and(|entry| entry.0.canceled.load(AtomicOrdering::Acquire))
        {
            self.heap.pop();
        }
    }

    /// Peek at the earliest scheduled time without removing anything.
    pub(crate) fn peek_time(&mut self) -> Option<SimTime> {
        self.discard_canceled_head();
        self.heap.peek().map(|r| r.0.time)
    }

    /// Remove and return the next action if its time equals `at`.
    ///
    /// Returns `None` if the queue is empty or the next action is at a later time.
    pub(crate) fn pull_if_at(&mut self, at: SimTime) -> Option<Action> {
        self.discard_canceled_head();
        if self.peek_time() == Some(at) {
            Some(self.heap.pop().unwrap().0.action)
        } else {
            None
        }
    }

    #[allow(dead_code)]
    pub(crate) fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    #[allow(dead_code)]
    pub(crate) fn len(&self) -> usize {
        self.heap.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
    use std::sync::Arc;
    use std::time::Duration;

    #[test]
    fn test_fifo_ordering_at_same_time() {
        let mut q = EventQueue::new();
        let order = Arc::new(AtomicUsize::new(0));

        let t = Duration::from_secs(1);

        for i in 0..3 {
            let o = order.clone();
            q.insert(
                t,
                Box::new(move || {
                    let prev = o.fetch_add(1, AtomicOrdering::SeqCst);
                    assert_eq!(prev, i);
                }),
            );
        }

        // All three should fire at t=1s in insertion order
        let a1 = q.pull_if_at(t).unwrap();
        let a2 = q.pull_if_at(t).unwrap();
        let a3 = q.pull_if_at(t).unwrap();
        assert!(q.pull_if_at(t).is_none());

        a1();
        a2();
        a3();
        assert_eq!(order.load(AtomicOrdering::SeqCst), 3);
    }

    #[test]
    fn test_time_ordering() {
        let mut q = EventQueue::new();

        q.insert(Duration::from_secs(3), Box::new(|| {}));
        q.insert(Duration::from_secs(1), Box::new(|| {}));
        q.insert(Duration::from_secs(2), Box::new(|| {}));

        assert_eq!(q.peek_time(), Some(Duration::from_secs(1)));
        assert!(q.pull_if_at(Duration::from_secs(1)).is_some());

        assert_eq!(q.peek_time(), Some(Duration::from_secs(2)));
        assert!(q.pull_if_at(Duration::from_secs(2)).is_some());

        assert_eq!(q.peek_time(), Some(Duration::from_secs(3)));
        assert!(q.pull_if_at(Duration::from_secs(3)).is_some());

        assert!(q.is_empty());
    }

    #[test]
    fn test_pull_if_at_wrong_time() {
        let mut q = EventQueue::new();
        q.insert(Duration::from_secs(5), Box::new(|| {}));

        // Asking for a different time should return None
        assert!(q.pull_if_at(Duration::from_secs(3)).is_none());
        assert!(!q.is_empty());
    }

    #[test]
    fn test_empty_queue() {
        let mut q = EventQueue::new();
        assert!(q.is_empty());
        assert_eq!(q.peek_time(), None);
        assert!(q.pull_if_at(Duration::ZERO).is_none());
    }

    #[test]
    fn test_canceled_head_is_skipped() {
        let mut q = EventQueue::new();

        let canceled = q.insert_cancellable(Duration::from_secs(1), Box::new(|| {}));
        q.insert(Duration::from_secs(2), Box::new(|| {}));
        canceled.cancel();

        assert_eq!(q.peek_time(), Some(Duration::from_secs(2)));
        assert!(q.pull_if_at(Duration::from_secs(1)).is_none());
        assert!(q.pull_if_at(Duration::from_secs(2)).is_some());
    }
}
