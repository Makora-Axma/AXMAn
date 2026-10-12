package org.axma.axman;

import java.lang.foreign.Arena;

/**
 * AXMANet Java 22+ Quickstart Example using Project Panama (FFM API / JEP 454).
 * Demonstrates assembling a Port-Hypergraph and executing zero-copy forward inference.
 */
public class Quickstart {
    public static void main(String[] args) throws Exception {
        System.out.println("============================================================");
        System.out.println("       AXMANet Java 22+ Quickstart: Panama FFM Inference    ");
        System.out.println("============================================================");

        int featDim = 256;
        long numNodes = 4;
        long numEdges = 2;

        // 1. Prepare continuous node features
        float[] features = new float[(int) (numNodes * featDim)];
        for (int i = 0; i < numNodes; i++) {
            for (int d = 0; d < featDim; d++) {
                features[i * featDim + d] = (float) (0.05 * (i + 1));
            }
        }

        // Binary hyperedge: (Premise 1 [node 1], Premise 2 [node 2]) -> Goal [node 3]
        // Unary edge: Hypothesis [node 0] -> Premise 1 [node 1]
        long[] sources = new long[] {1, 2, 0};
        int[] ports = new int[] {0, 1, 0};
        long[] targets = new long[] {3, 1};
        long[] offsets = new long[] {0, 2, 3};

        // 2. Allocate and initialize Hypergraph within confined Arena
        try (Arena arena = Arena.ofConfined()) {
            Hypergraph graph = new Hypergraph(
                    arena,
                    numNodes, featDim,
                    features,
                    sources, ports, targets, offsets
            );

            System.out.println("[1] Hypergraph assembled: " + numNodes + " nodes, " + numEdges + " hyperedges.");

            graph.validate();
            long structuralHash = graph.structuralHash();
            System.out.printf("[2] Topology valid. Structural Hash: 0x%016x%n", structuralHash);

            // 3. Model Configuration (JSON)
            String configJson = """
                {
                    "d_model": 256,
                    "num_layers": 4,
                    "num_heads": 8,
                    "port_dim": 64,
                    "num_actions": 17,
                    "device": "cpu"
                }
                """;

            // 4. Instantiate Model and execute zero-copy forward pass
            System.out.println("[3] Running zero-copy forward inference...");
            try (AxmanModel model = AxmanModel.load(configJson, arena)) {
                try (InferenceResult result = model.forward(graph)) {
                    float valueEstimate = result.getValueEstimate();
                    float[] policyLogits = result.getPolicyLogits();
                    float[] latentEmbedding = result.getLatentEmbedding();

                    System.out.printf("  -> Value Estimate (Distance to Closure): %.4f%n", valueEstimate);
                    System.out.println("  -> Policy Logits elements: " + policyLogits.length);
                    System.out.println("  -> Latent Invariant elements: " + latentEmbedding.length);
                }
            }

            System.out.println("\n[SUCCESS] AXMANet Java Quickstart completed successfully!");
        }
    }
}
