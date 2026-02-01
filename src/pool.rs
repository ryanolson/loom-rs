//! Compute task pool registry for zero-allocation spawning.
//!
//! This module provides a per-type object pool for `TaskState` objects used by
//! `spawn_compute()`. After warmup, `spawn_compute()` can reuse pooled state
//! objects, achieving zero allocation.
//!
//! # Architecture
//!
//! ```text
//! ComputePoolRegistry
//!   └── HashMap<TypeId, Box<dyn PoolWrapper>>
//!         └── PoolWrapperImpl<R>
//!               └── Arc<TypedPool<R>>
//!                     └── ArrayQueue<Arc<TaskState<R>>>
//! ```
//!
//! Each unique result type `R` used with `spawn_compute::<F, R>()` gets its own
//! lock-free pool of reusable `TaskState<R>` objects.

use std::any::TypeId;
use std::collections::HashMap;
use std::sync::Arc;

use crossbeam_queue::ArrayQueue;
use parking_lot::RwLock;

use crate::bridge::TaskState;

/// Default pool size per result type.
pub const DEFAULT_POOL_SIZE: usize = 64;

/// Registry of per-type compute pools.
///
/// Thread-safe registry that manages one pool per result type. Pools are created
/// lazily on first use and shared across all threads.
pub(crate) struct ComputePoolRegistry {
    pools: RwLock<HashMap<TypeId, Box<dyn PoolWrapper>>>,
    default_size: usize,
}

/// Trait for type-erased pool wrapper.
trait PoolWrapper: Send + Sync {
    /// Clone the inner Arc as a type-erased box.
    fn clone_arc(&self) -> Box<dyn PoolWrapper>;
}

/// Wrapper that holds Arc<TypedPool<R>> for cloning.
struct PoolWrapperImpl<R: Send + 'static> {
    pool: Arc<TypedPool<R>>,
}

impl<R: Send + 'static> PoolWrapper for PoolWrapperImpl<R> {
    fn clone_arc(&self) -> Box<dyn PoolWrapper> {
        Box::new(PoolWrapperImpl {
            pool: self.pool.clone(),
        })
    }
}

impl<R: Send + 'static> PoolWrapperImpl<R> {
    fn get_pool(&self) -> Arc<TypedPool<R>> {
        self.pool.clone()
    }
}

/// A lock-free pool for a specific result type.
pub(crate) struct TypedPool<R> {
    queue: ArrayQueue<Arc<TaskState<R>>>,
}

impl<R: Send + 'static> TypedPool<R> {
    /// Create a new pool with the given capacity.
    fn new(capacity: usize) -> Self {
        Self {
            queue: ArrayQueue::new(capacity),
        }
    }

    /// Try to get a reusable TaskState from the pool.
    pub fn pop(&self) -> Option<Arc<TaskState<R>>> {
        self.queue.pop()
    }

    /// Return a TaskState to the pool for reuse.
    ///
    /// If the pool is full, the state is dropped.
    pub fn push(&self, state: Arc<TaskState<R>>) {
        // Ignore error if pool is full - the state will just be dropped
        let _ = self.queue.push(state);
    }
}

impl ComputePoolRegistry {
    /// Create a new registry with the given default pool size.
    pub fn new(default_size: usize) -> Self {
        Self {
            pools: RwLock::new(HashMap::new()),
            default_size,
        }
    }

    /// Get or create a pool for type R.
    ///
    /// The first call for a given type creates the pool (slow path with write lock).
    /// Subsequent calls use the fast path (read lock only).
    pub fn get_or_create<R: Send + 'static>(&self) -> Arc<TypedPool<R>> {
        let type_id = TypeId::of::<R>();

        // Fast path: read lock
        {
            let pools = self.pools.read();
            if let Some(wrapper) = pools.get(&type_id) {
                // SAFETY: We only insert PoolWrapperImpl<R> for TypeId::of::<R>()
                // downcast_ref isn't available for Box<dyn Trait>, so we use a different approach
                // We clone the wrapper and extract the pool from the clone
                let cloned = wrapper.clone_arc();
                // SAFETY: The wrapper for this TypeId is always PoolWrapperImpl<R>
                let wrapper_impl = unsafe {
                    &*(cloned.as_ref() as *const dyn PoolWrapper as *const PoolWrapperImpl<R>)
                };
                return wrapper_impl.get_pool();
            }
        }

        // Slow path: write lock, create pool
        let mut pools = self.pools.write();

        // Double-check after acquiring write lock
        if let Some(wrapper) = pools.get(&type_id) {
            let cloned = wrapper.clone_arc();
            let wrapper_impl = unsafe {
                &*(cloned.as_ref() as *const dyn PoolWrapper as *const PoolWrapperImpl<R>)
            };
            return wrapper_impl.get_pool();
        }

        // Create new pool
        let pool = Arc::new(TypedPool::new(self.default_size));
        let wrapper = Box::new(PoolWrapperImpl { pool: pool.clone() });
        pools.insert(type_id, wrapper);
        pool
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_typed_pool_basic() {
        let pool: TypedPool<i32> = TypedPool::new(4);

        // Initially empty
        assert!(pool.pop().is_none());

        // Add some items
        let state1 = Arc::new(TaskState::<i32>::new());
        let state2 = Arc::new(TaskState::<i32>::new());

        pool.push(state1.clone());
        pool.push(state2.clone());

        // Should get items back
        assert!(pool.pop().is_some());
        assert!(pool.pop().is_some());
        assert!(pool.pop().is_none());
    }

    #[test]
    fn test_typed_pool_overflow() {
        let pool: TypedPool<i32> = TypedPool::new(2);

        // Fill the pool
        pool.push(Arc::new(TaskState::new()));
        pool.push(Arc::new(TaskState::new()));

        // This should not panic, just drop the item
        pool.push(Arc::new(TaskState::new()));

        // Still only 2 items
        assert!(pool.pop().is_some());
        assert!(pool.pop().is_some());
        assert!(pool.pop().is_none());
    }

    #[test]
    fn test_registry_creates_per_type_pools() {
        let registry = ComputePoolRegistry::new(4);

        // Get pool for i32
        let pool_i32a = registry.get_or_create::<i32>();
        let pool_i32b = registry.get_or_create::<i32>();

        // Should be the same pool
        assert!(Arc::ptr_eq(&pool_i32a, &pool_i32b));

        // Get pool for String - should be different
        let pool_string = registry.get_or_create::<String>();

        // Add to i32 pool
        pool_i32a.push(Arc::new(TaskState::new()));

        // String pool should still be empty
        assert!(pool_string.pop().is_none());

        // i32 pool should have the item
        assert!(pool_i32a.pop().is_some());
    }

    #[test]
    fn test_registry_concurrent_access() {
        use std::thread;

        let registry = Arc::new(ComputePoolRegistry::new(64));
        let mut handles = vec![];

        for _ in 0..4 {
            let reg = registry.clone();
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    let pool = reg.get_or_create::<u64>();
                    let state = Arc::new(TaskState::new());
                    pool.push(state);
                    let _ = pool.pop();
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }
    }
}
