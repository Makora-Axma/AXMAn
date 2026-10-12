use axman_rs::{AxmanModel, Hypergraph};
use std::fs::File;
use std::io::{Read, Write};
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let base_dir = if args.len() > 1 {
        PathBuf::from(&args[1])
    } else {
        PathBuf::from(".")
    };

    let config_path = base_dir.join("golden_config.json");
    let bin_path = base_dir.join("golden_sample.bin");
    let out_path = base_dir.join("golden_output_rust.json");

    let mut config_file = File::open(&config_path).expect("Open golden_config.json");
    let mut config_json = String::new();
    config_file.read_to_string(&mut config_json).unwrap();

    let mut bin_file = File::open(&bin_path).expect("Open golden_sample.bin");
    let mut magic = [0u8; 4];
    bin_file.read_exact(&mut magic).unwrap();

    let mut header = [0u8; 32];
    bin_file.read_exact(&mut header).unwrap();
    let num_nodes = u64::from_le_bytes(header[0..8].try_into().unwrap()) as usize;
    let feat_dim = u64::from_le_bytes(header[8..16].try_into().unwrap()) as usize;
    let num_edges = u64::from_le_bytes(header[16..24].try_into().unwrap()) as usize;
    let num_sources = u64::from_le_bytes(header[24..32].try_into().unwrap()) as usize;

    let mut feat_bytes = vec![0u8; num_nodes * feat_dim * 4];
    bin_file.read_exact(&mut feat_bytes).unwrap();
    let features: Vec<f32> = feat_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    let mut src_bytes = vec![0u8; num_sources * 8];
    bin_file.read_exact(&mut src_bytes).unwrap();
    let sources: Vec<i64> = src_bytes
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect();

    let mut port_bytes = vec![0u8; num_sources * 4];
    bin_file.read_exact(&mut port_bytes).unwrap();
    let ports: Vec<i32> = port_bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    let mut target_bytes = vec![0u8; num_edges * 8];
    bin_file.read_exact(&mut target_bytes).unwrap();
    let targets: Vec<i64> = target_bytes
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect();

    let mut offset_bytes = vec![0u8; (num_edges + 1) * 8];
    bin_file.read_exact(&mut offset_bytes).unwrap();
    let offsets: Vec<usize> = offset_bytes
        .chunks_exact(8)
        .map(|c| usize::from_le_bytes(c.try_into().unwrap()))
        .collect();

    // Model & Forward
    let model = AxmanModel::load::<&str>(None, Some(&config_json)).expect("Load rust model");
    let graph = Hypergraph::new(
        num_nodes,
        num_edges,
        &features,
        feat_dim,
        &sources,
        &ports,
        &targets,
        &offsets,
    )
    .expect("Create hypergraph");

    let structural_hash = graph.structural_hash().expect("Hash in Rust");
    let output = model.forward(&graph).expect("Forward in Rust");

    let val = output.value_estimate();
    let policy = output.policy_logits();
    let latent = output.latent_embedding();

    let policy_slice = policy.as_slice();
    let latent_slice = latent.as_slice();

    let mut out_file = File::create(&out_path).expect("Create output json");
    writeln!(out_file, "{{").unwrap();
    writeln!(out_file, "  \"language\": \"rust\",").unwrap();
    writeln!(out_file, "  \"structural_hash\": {},", structural_hash).unwrap();
    writeln!(out_file, "  \"value_estimate\": {},", val).unwrap();
    write!(out_file, "  \"policy_logits\": [").unwrap();
    for (i, v) in policy_slice.iter().enumerate() {
        if i > 0 { write!(out_file, ", ").unwrap(); }
        write!(out_file, "{}", v).unwrap();
    }
    writeln!(out_file, "],").unwrap();
    write!(out_file, "  \"latent_embedding\": [").unwrap();
    for (i, v) in latent_slice.iter().enumerate() {
        if i > 0 { write!(out_file, ", ").unwrap(); }
        write!(out_file, "{}", v).unwrap();
    }
    writeln!(out_file, "]").unwrap();
    writeln!(out_file, "}}").unwrap();

    println!("[Rust] Executed successfully. Structural hash: {}, value: {}", structural_hash, val);
}
