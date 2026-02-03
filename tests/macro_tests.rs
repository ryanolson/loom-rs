//! Integration tests for the #[loom_rs::test] macro.

/// Test basic usage with defaults (1 tokio, 2 rayon, no pinning)
#[loom_rs::test]
async fn test_basic_spawn_compute() {
    let result = loom_rs::spawn_compute(|| 42).await;
    assert_eq!(result, 42);
}

/// Test that the runtime is accessible via current_runtime()
#[loom_rs::test]
async fn test_current_runtime_available() {
    let runtime = loom_rs::current_runtime().expect("runtime should be available");
    assert_eq!(runtime.tokio_threads(), 1);
    assert_eq!(runtime.rayon_threads(), 2);
}

/// Test custom tokio thread count
#[loom_rs::test(tokio_thread_count = 2)]
async fn test_custom_tokio_threads() {
    let runtime = loom_rs::current_runtime().expect("runtime should be available");
    assert_eq!(runtime.tokio_threads(), 2);
    assert_eq!(runtime.rayon_threads(), 2); // Default
}

/// Test custom rayon thread count
#[loom_rs::test(rayon_thread_count = 4)]
async fn test_custom_rayon_threads() {
    let runtime = loom_rs::current_runtime().expect("runtime should be available");
    assert_eq!(runtime.tokio_threads(), 1); // Default
    assert_eq!(runtime.rayon_threads(), 4);
}

/// Test both custom thread counts
#[loom_rs::test(tokio_thread_count = 2, rayon_thread_count = 4)]
async fn test_custom_both_thread_counts() {
    let runtime = loom_rs::current_runtime().expect("runtime should be available");
    assert_eq!(runtime.tokio_threads(), 2);
    assert_eq!(runtime.rayon_threads(), 4);
}

/// Test spawn_async works
#[loom_rs::test]
async fn test_spawn_async() {
    let runtime = loom_rs::current_runtime().unwrap();
    let handle = runtime.spawn_async(async { 100 });
    let result = handle.await.unwrap();
    assert_eq!(result, 100);
}

/// Test that scope_compute works with borrowed data
#[loom_rs::test]
async fn test_scope_compute() {
    use std::sync::atomic::{AtomicI32, Ordering};

    let data = [1, 2, 3, 4, 5, 6, 7, 8];
    let sum = AtomicI32::new(0);

    loom_rs::scope_compute(|s| {
        let (left, right) = data.split_at(data.len() / 2);
        let sum_ref = &sum;

        s.spawn(move |_| {
            sum_ref.fetch_add(left.iter().sum::<i32>(), Ordering::Relaxed);
        });
        s.spawn(move |_| {
            sum_ref.fetch_add(right.iter().sum::<i32>(), Ordering::Relaxed);
        });
    })
    .await;

    assert_eq!(sum.load(Ordering::Relaxed), 36);
}

/// Test that parallel iterators work via install
#[loom_rs::test]
async fn test_install_parallel_iter() {
    use rayon::prelude::*;

    let runtime = loom_rs::current_runtime().unwrap();
    let result = runtime.install(|| (0..100).into_par_iter().sum::<i32>());
    assert_eq!(result, 4950);
}

/// Test that panics in scope_compute propagate correctly
/// Note: We use scope_compute because it has panic handling.
/// spawn_compute panics will abort rayon's thread pool.
#[loom_rs::test]
#[should_panic(expected = "test panic")]
async fn test_panic_propagation() {
    loom_rs::scope_compute(|_s| {
        panic!("test panic");
    })
    .await;
}

/// Test adaptive spawn
#[loom_rs::test]
async fn test_spawn_adaptive() {
    let result = loom_rs::spawn_adaptive(|| 123).await;
    assert_eq!(result, 123);
}

// =============================================================================
// Tests demonstrating block_until_idle behavior
// =============================================================================

/// Test that fire-and-forget async tasks complete before test exits.
/// The macro calls block_until_idle() which waits for all tracked tasks.
#[loom_rs::test]
async fn test_fire_and_forget_async_tasks_complete() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    let completed = Arc::new(AtomicBool::new(false));
    let completed_clone = completed.clone();

    let runtime = loom_rs::current_runtime().unwrap();

    // Spawn a fire-and-forget task - we don't await the handle
    runtime.spawn_async(async move {
        // Simulate some async work
        tokio::task::yield_now().await;
        completed_clone.store(true, Ordering::Release);
    });

    // Test body exits here, but block_until_idle() will wait for the task.
    // We can't check `completed` here since the task may not have run yet,
    // but if the test passes without hanging, it means block_until_idle worked.
    let _ = completed; // Just to show we have the Arc
}

/// Test that multiple fire-and-forget tasks all complete.
#[loom_rs::test]
async fn test_multiple_fire_and_forget_tasks() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    let counter = Arc::new(AtomicUsize::new(0));
    let runtime = loom_rs::current_runtime().unwrap();

    // Spawn several fire-and-forget tasks
    for _ in 0..10 {
        let counter_clone = counter.clone();
        runtime.spawn_async(async move {
            tokio::task::yield_now().await;
            counter_clone.fetch_add(1, Ordering::Relaxed);
        });
    }

    // All 10 tasks will complete before the test exits due to block_until_idle()
    let _ = counter;
}

/// Test that nested task spawning works correctly.
#[loom_rs::test]
async fn test_nested_task_spawning() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    let counter = Arc::new(AtomicUsize::new(0));
    let runtime = loom_rs::current_runtime().unwrap();

    let counter_clone = counter.clone();
    runtime.spawn_async(async move {
        let rt = loom_rs::current_runtime().unwrap();
        let counter_inner = counter_clone.clone();

        // Spawn a nested task
        rt.spawn_async(async move {
            counter_inner.fetch_add(1, Ordering::Relaxed);
        });

        counter_clone.fetch_add(1, Ordering::Relaxed);
    });

    // Both the outer and inner tasks will complete
    let _ = counter;
}

/// Test mixing spawn_async and spawn_compute in the same test.
#[loom_rs::test]
async fn test_mixed_async_and_compute() {
    use std::sync::atomic::{AtomicI32, Ordering};
    use std::sync::Arc;

    let sum = Arc::new(AtomicI32::new(0));
    let runtime = loom_rs::current_runtime().unwrap();

    // Spawn some async tasks
    for i in 0..5 {
        let sum_clone = sum.clone();
        runtime.spawn_async(async move {
            sum_clone.fetch_add(i, Ordering::Relaxed);
        });
    }

    // Spawn some compute tasks (these we await)
    let compute_result: i32 = runtime.spawn_compute(|| (0..5).sum()).await;

    assert_eq!(compute_result, 10); // 0+1+2+3+4

    // The async tasks will also complete due to block_until_idle()
    let _ = sum;
}

/// Test that compute tasks spawned without await still complete.
/// Note: spawn_compute returns a future that must be awaited to get the result,
/// but the task itself is tracked and will complete even if the future is dropped.
#[loom_rs::test]
async fn test_compute_task_tracking() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    let completed = Arc::new(AtomicBool::new(false));
    let completed_clone = completed.clone();

    let runtime = loom_rs::current_runtime().unwrap();

    // Start a compute task but don't await it
    let _future = runtime.spawn_compute(move || {
        completed_clone.store(true, Ordering::Release);
        42
    });

    // Even though we didn't await, the compute task is tracked and will complete
    // before block_until_idle() returns.
    let _ = completed;
}

/// Test using the free function spawn_compute (not on runtime).
#[loom_rs::test]
async fn test_free_function_spawn_compute() {
    let result = loom_rs::spawn_compute(|| {
        // Some CPU-bound work
        (0..1000).map(|x| x * x).sum::<i64>()
    })
    .await;

    assert_eq!(result, 332833500); // sum of squares 0..1000
}

/// Test using the free function scope_compute with borrowed data.
#[loom_rs::test]
async fn test_free_function_scope_compute() {
    let data = [1, 2, 3, 4, 5];

    let sum = loom_rs::scope_compute(|_s| {
        // Borrow data inside the scope
        data.iter().sum::<i32>()
    })
    .await;

    assert_eq!(sum, 15);
    // data is still valid after scope_compute
    assert_eq!(data.len(), 5);
}

/// Test that the macro supports Result<()> return type.
#[loom_rs::test]
async fn test_result_return_type() -> Result<(), Box<dyn std::error::Error>> {
    let result = loom_rs::spawn_compute(|| 42).await;
    assert_eq!(result, 42);
    Ok(())
}

/// Test Result return type with error propagation using ?.
#[loom_rs::test]
async fn test_result_with_question_mark() -> Result<(), Box<dyn std::error::Error>> {
    let value: i32 = "42".parse()?;
    let result = loom_rs::spawn_compute(move || value * 2).await;
    assert_eq!(result, 84);
    Ok(())
}
