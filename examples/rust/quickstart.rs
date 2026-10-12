//! AXMANet Rust Quickstart Example.
//! Demonstrates constructing a Port-Hypergraph and executing zero-copy forward inference.

use axman_rs::{AxmanModel, Hypergraph};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("============================================================");
    println!("       AXMANet Rust Quickstart: Formal Reasoning Inference  ");
    println!("============================================================");

    let feat_dim: usize = 256;
    let num_nodes: usize = 4;
    let num_edges: usize = 2;

    // 1. Prepare node feature buffers and topological connectivity
    let mut features = vec![0.0f32; num_nodes * feat_dim];
    for i in 0..num_nodes {
        for d in 0..feat_dim {
            features[i * feat_dim + d] = 0.05 * (i as f32 + 1.0);
        }
    }

    // Binary hyperedge: (Premise 1 [node 1], Premise 2 [node 2]) -> Goal [node 3]
    // Unary edge: Hypothesis [node 0] -> Premise 1 [node 1]
    let sources: Vec<i64> = vec![1, 2, 0];
    let ports: Vec<i32> = vec![0, 1, 0];
    let targets: Vec<i64> = vec![3, 1];
    let offsets: Vec<usize> = vec![0, 2, 3];

    // 2. Build and validate Port-Hypergraph
    let graph = Hypergraph::new(
        num_nodes,
        num_edges,
        &features,
        feat_dim,
        &sources,
        &ports,
        &targets,
        &offsets,
    )?;

    println!("[1] Hypergraph assembled: {} nodes, {} hyperedges.", num_nodes, num_edges);

    let structural_hash = graph.structural_hash()?;
    println!("[2] Topology valid. Structural Hash: 0x{:016x}", structural_hash);

    // 3. Model Configuration (JSON)
    let config_json = r#"{
        "d_model": 256,
        "num_layers": 4,
        "num_heads": 8,
        "port_dim": 64,
        "num_actions": 17,
        "device": "cpu"
    }"#;

    // 4. Instantiate Model and execute zero-copy forward pass
    println!("[3] Running zero-copy forward inference...");
    let model = AxmanModel::load::<&str>(None, Some(config_json))?;
    let output = model.forward(&graph)?;

    let val = output.value_estimate();
    let policy_slice = output.policy_logits();
    let latent_slice = output.latent_embedding();

    println!("  -> Value Estimate (Distance to Closure): {:.4}", val);
    println!("  -> Policy Logits elements: {}", policy_slice.len());
    println!("  -> Latent Invariant elements: {}", latent_slice.len());

    println!("\n[SUCCESS] AXMANet Rust Quickstart completed successfully!");
    Ok(())
}
