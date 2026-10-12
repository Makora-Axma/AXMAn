use axman_rs::{AxmanModel, Hypergraph};
use std::sync::Arc;

#[test]
fn test_version_string() {
    let ver = axman_rs::version();
    assert_eq!(ver, "1.0.0");
}

#[test]
fn test_model_lifecycle_and_forward() {
    let config = r#"{"d_model": 64, "num_layers": 2, "num_heads": 4, "num_actions": 16, "in_features": 32}"#;
    let model = AxmanModel::load::<&str>(None, Some(config)).expect("Failed to initialize model");

    let num_nodes = 3;
    let num_edges = 2;
    let feat_dim = 32;

    let node_features = vec![0.5f32; num_nodes * feat_dim];
    let edge_sources = [0i64, 0, 1];
    let edge_ports = [0i32, 0, 1];
    let edge_targets = [1i64, 2];
    let edge_offsets = [0usize, 1, 3];

    let graph = Hypergraph::new(
        num_nodes,
        num_edges,
        &node_features,
        feat_dim,
        &edge_sources,
        &edge_ports,
        &edge_targets,
        &edge_offsets,
    )
    .expect("Failed to create hypergraph");

    let hash = graph.structural_hash().expect("Failed to compute hash");
    assert_ne!(hash, 0);

    let output = model.forward(&graph).expect("Inference failed");

    assert!((0.0..=1.0).contains(&output.value_estimate()));

    let policy = output.policy_logits();
    assert_eq!(policy.shape(), &[3, 16]);
    assert_eq!(policy.as_slice().len(), 3 * 16);

    let latent = output.latent_embedding();
    assert_eq!(latent.shape(), &[1, 64]);
    assert_eq!(latent.as_slice().len(), 64);
}

#[test]
fn test_multithreaded_rayon_inference() {
    use rayon::prelude::*;

    let config = r#"{"d_model": 64, "num_layers": 1, "num_heads": 2, "num_actions": 8, "in_features": 16}"#;
    let model = Arc::new(AxmanModel::load::<&str>(None, Some(config)).unwrap());

    let num_nodes = 2;
    let num_edges = 1;
    let feat_dim = 16;
    let node_features = vec![0.25f32; num_nodes * feat_dim];
    let edge_sources = [0i64];
    let edge_ports = [0i32];
    let edge_targets = [1i64];
    let edge_offsets = [0usize, 1];

    let graph = Hypergraph::new(
        num_nodes,
        num_edges,
        &node_features,
        feat_dim,
        &edge_sources,
        &edge_ports,
        &edge_targets,
        &edge_offsets,
    )
    .unwrap();

    // Run 50 parallel inferences across threads using Rayon
    let results: Vec<f32> = (0..50)
        .into_par_iter()
        .map(|_| {
            let out = model.forward(&graph).unwrap();
            out.value_estimate()
        })
        .collect();

    assert_eq!(results.len(), 50);
    for v in results {
        assert!((0.0..=1.0).contains(&v));
    }
}

#[test]
fn test_topology_validation_error() {
    let num_nodes = 2;
    let num_edges = 1;
    let feat_dim = 16;
    let node_features = vec![0.25f32; num_nodes * feat_dim];
    let edge_sources = [0i64];
    let edge_ports = [0i32];
    let edge_targets = [99i64]; // Out of bounds
    let edge_offsets = [0usize, 1];

    let res = Hypergraph::new(
        num_nodes,
        num_edges,
        &node_features,
        feat_dim,
        &edge_sources,
        &edge_ports,
        &edge_targets,
        &edge_offsets,
    );

    assert!(res.is_err());
}

#[test]
fn test_tensor_view_ergonomics_and_accessors() {
    let config = r#"{"d_model": 32, "num_layers": 1, "num_heads": 2, "num_actions": 8, "in_features": 16}"#;
    let model = AxmanModel::from_config(config).expect("Failed to initialize model from config");

    let num_nodes = 2;
    let num_edges = 1;
    let feat_dim = 16;
    let node_features = vec![0.1f32; num_nodes * feat_dim];
    let edge_sources = [0i64];
    let edge_ports = [0i32];
    let edge_targets = [1i64];
    let edge_offsets = [0usize, 1];

    let graph = Hypergraph::new(
        num_nodes,
        num_edges,
        &node_features,
        feat_dim,
        &edge_sources,
        &edge_ports,
        &edge_targets,
        &edge_offsets,
    )
    .expect("Failed to create hypergraph");

    assert_eq!(graph.num_nodes(), 2);
    assert_eq!(graph.num_hyperedges(), 1);
    assert_eq!(graph.node_feat_dim(), 16);
    assert_eq!(graph.arity(0), Some(1));
    assert_eq!(graph.arity(1), None);

    let output = model.forward(&graph).expect("Inference forward failed");

    let policy = output.policy_logits();
    assert_eq!(policy.len(), 2 * 8);
    assert!(!policy.is_empty());
    assert_eq!(policy.ndim(), 2);
    assert_eq!(policy.shape(), &[2, 8]);

    // Test Index<usize> and Deref
    let val_0 = policy[0];
    assert_eq!(val_0, policy.as_slice()[0]);

    // Test get_2d
    assert!(policy.get_2d(0, 0).is_some());
    assert!(policy.get_2d(1, 7).is_some());
    assert!(policy.get_2d(2, 0).is_none()); // Row out of bounds
    assert!(policy.get_2d(0, 8).is_none()); // Col out of bounds

    let latent = output.latent_embedding();
    assert_eq!(latent.len(), 32);
    assert!(!latent.is_empty());
    assert_eq!(latent.shape(), &[1, 32]);
}

#[test]
fn test_model_config_presets() {
    let small = axman_rs::ModelConfig::small();
    assert_eq!(small.d_model, 256);
    assert_eq!(small.num_layers, 6);

    let base = axman_rs::ModelConfig::base();
    assert_eq!(base.d_model, 512);

    let large = axman_rs::ModelConfig::large();
    assert_eq!(large.d_model, 1024);

    let json = small.to_json();
    assert!(json.contains("\"d_model\": 256"));

    let model = AxmanModel::from_model_config(&small).expect("Failed to create model from ModelConfig");
    assert!(!model.version().is_empty());
}

#[test]
fn test_safetensors_file_load_and_forward() {
    use std::fs::File;
    use std::io::Write;

    let test_path = std::env::temp_dir().join("axman_rust_test_model.safetensors");

    // Construct valid safetensors binary archive
    // Dimensions: d_model=2, layers=1, heads=1, in_features=2
    let w_data: [f32; 4] = [1.5, -2.0, 3.25, 0.75];
    let b_data: [f32; 2] = [0.1, -0.2];

    let json_header = concat!(
        r#"{"__metadata__":{"d_model":"2","num_layers":"1","num_heads":"1","in_features":"2"},"#,
        r#""node_encoder.weight":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},"#,
        r#""node_encoder.bias":{"dtype":"F32","shape":[2],"data_offsets":[16,24]}}"#
    );

    let header_bytes = json_header.as_bytes();
    let header_len = header_bytes.len() as u64;

    {
        let mut f = File::create(&test_path).expect("Failed to create test safetensors file");
        f.write_all(&header_len.to_le_bytes()).unwrap();
        f.write_all(header_bytes).unwrap();
        for &w in &w_data {
            f.write_all(&w.to_le_bytes()).unwrap();
        }
        for &b in &b_data {
            f.write_all(&b.to_le_bytes()).unwrap();
        }
        f.flush().unwrap();
    }

    // Load model using AxmanModel::from_file
    let model = AxmanModel::from_file(&test_path, None).expect("Failed to load model from safetensors");

    let num_nodes = 2;
    let num_edges = 1;
    let feat_dim = 2;
    let node_features = [1.0f32, 0.5, -1.0, 2.0];
    let edge_sources = [0i64];
    let edge_ports = [0i32];
    let edge_targets = [1i64];
    let edge_offsets = [0usize, 1];

    let graph = Hypergraph::new(
        num_nodes,
        num_edges,
        &node_features,
        feat_dim,
        &edge_sources,
        &edge_ports,
        &edge_targets,
        &edge_offsets,
    )
    .expect("Failed to construct graph");

    let output = model.forward(&graph).expect("Inference failed with safetensors model");
    assert!((0.0..=1.0).contains(&output.value_estimate()));

    let _ = std::fs::remove_file(&test_path);
}
