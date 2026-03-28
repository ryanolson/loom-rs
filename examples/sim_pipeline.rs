//! Simulated async processing pipeline.
//!
//! A multi-stage pipeline where each stage takes simulated time. Requests
//! flow: Client -> Network -> Server -> Compute -> Server -> Network -> Client.
//! Demonstrates:
//!
//! - `SimHandle::delay().await` for suspending tasks in virtual time
//! - `spawn_async()` inside simulation for async task pipelines
//! - `tokio::sync` channels working under virtual time
//! - Mixed DES scheduling + async/await patterns
//!
//! ```text
//!   Client           Network (5ms)      Server         Compute (20ms)
//!     |--- request ----->|--- delay --->|                    |
//!     |                  |              |--- process ------->|
//!     |                  |              |<-- result ---------|
//!     |<--- response ----|<-- delay ----|                    |
//! ```
//!
//! Run with: `cargo run --features sim --example sim_pipeline`

use std::time::Duration;

use loom_rs::sim::{SimHandle, SimulationRuntime};
use tokio::sync::{mpsc, oneshot};

/// A request from client to server.
struct Request {
    id: u32,
    payload: String,
    respond_to: oneshot::Sender<Response>,
}

/// A response from server to client.
struct Response {
    id: u32,
    result: String,
    server_processing_time: Duration,
}

/// Simulated server that processes requests with a compute delay.
async fn server(sim: SimHandle, mut inbox: mpsc::Receiver<Request>) {
    let compute_delay = Duration::from_millis(20);

    while let Some(req) = inbox.recv().await {
        let start = sim.now();
        println!("  [{:>6.1?}] Server: processing request #{}", start, req.id);

        // Simulate compute work
        sim.delay(compute_delay).await;

        let end = sim.now();
        let elapsed = end - start;
        println!(
            "  [{:>6.1?}] Server: request #{} done ({:?} compute)",
            end, req.id, elapsed
        );

        let _ = req.respond_to.send(Response {
            id: req.id,
            result: format!("processed({})", req.payload),
            server_processing_time: elapsed,
        });
    }
}

/// Simulated client that sends requests and collects responses.
async fn client(sim: SimHandle, server_tx: mpsc::Sender<Request>, count: u32) {
    let network_delay = Duration::from_millis(5);

    for id in 1..=count {
        let send_time = sim.now();
        println!("  [{:>6.1?}] Client: sending request #{}", send_time, id);

        // Simulate network latency to server
        sim.delay(network_delay).await;

        let (tx, rx) = oneshot::channel();
        server_tx
            .send(Request {
                id,
                payload: format!("data-{}", id),
                respond_to: tx,
            })
            .await
            .unwrap();

        // Wait for response
        let resp = rx.await.unwrap();

        // Simulate network latency from server
        sim.delay(network_delay).await;

        let receive_time = sim.now();
        let rtt = receive_time - send_time;
        println!(
            "  [{:>6.1?}] Client: got response #{}: {:?} (RTT: {:?}, server: {:?})",
            receive_time, resp.id, resp.result, rtt, resp.server_processing_time
        );
    }

    println!("  [{:>6.1?}] Client: all requests complete", sim.now());
}

fn main() -> loom_rs::Result<()> {
    let wall_start = std::time::Instant::now();

    let mut sim = SimulationRuntime::new()?;
    let setup_elapsed = wall_start.elapsed();

    let handle = sim.handle();

    let (server_tx, server_rx) = mpsc::channel::<Request>(16);

    // Spawn server and client as async tasks
    let server_handle = handle.clone();
    sim.loom().spawn_async(server(server_handle, server_rx));

    let client_handle = handle.clone();
    sim.loom().spawn_async(client(client_handle, server_tx, 3));

    println!("=== Pipeline Simulation ===\n");
    println!("  Network delay: 5ms each way");
    println!("  Compute delay: 20ms per request");
    println!("  Expected RTT:  30ms (5 + 20 + 5)\n");

    let sim_start = std::time::Instant::now();
    let final_time = sim.run()?;
    let sim_elapsed = sim_start.elapsed();

    println!("\n=== Simulation Complete ===");
    println!("  Virtual time simulated: {:?}", final_time);
    println!("  Wall-clock runtime setup: {:?}", setup_elapsed);
    println!("  Wall-clock simulation:    {:?}", sim_elapsed);
    println!(
        "  Speedup: {:.0}x faster than real time",
        final_time.as_secs_f64() / sim_elapsed.as_secs_f64()
    );

    Ok(())
}
