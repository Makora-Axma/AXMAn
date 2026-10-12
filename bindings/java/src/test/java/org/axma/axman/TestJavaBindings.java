package org.axma.axman;

import java.io.OutputStream;
import java.lang.foreign.Arena;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;

public class TestJavaBindings {

    public static void main(String[] args) throws Exception {
        System.out.println("=====================================================");
        System.out.println("       AXMANet PHASE 4.4: JAVA 22+ PANAMA FFM        ");
        System.out.println("=====================================================");

        testVersion();
        testHypergraphValidationAndHash();
        testHypergraphBuilder();
        testModelConfigPresets();
        testModelForward();
        testDirectConstructorAndSafetensorsLoad();
        testMultithreadedInference();
        testTopologyErrorHandling();
        testCsrOffsetValidationErrors();
        testResourceSafetyAndDoubleClose();

        System.out.println("=====================================================");
        System.out.println("   >>> ALL PHASE 4.4 TESTS PASSED SUCCESSFULLY! <<<  ");
        System.out.println("=====================================================");
    }

    private static void testVersion() {
        System.out.println("[Test 1] Testing version string...");
        String ver = AxmanModel.getVersion();
        if (!"1.0.0".equals(ver)) {
            throw new AssertionError("Expected 1.0.0, got: " + ver);
        }
        System.out.println("  -> OK (Version: " + ver + ")");
    }

    private static void testHypergraphValidationAndHash() {
        System.out.println("[Test 2] Testing Hypergraph creation & structural hash...");
        try (Arena arena = Arena.ofConfined()) {
            float[] nodeFeatures = new float[]{
                    0.5f, 0.5f, 0.5f, 0.5f,
                    0.6f, 0.6f, 0.6f, 0.6f,
                    0.7f, 0.7f, 0.7f, 0.7f
            };
            long[] sources = new long[]{0, 0, 1};
            int[] ports = new int[]{0, 0, 1};
            long[] targets = new long[]{1, 2};
            long[] offsets = new long[]{0, 1, 3};

            Hypergraph graph = new Hypergraph(
                    arena,
                    3, 4,
                    nodeFeatures,
                    sources, ports, targets, offsets
            );

            graph.validate();
            long hash = graph.structuralHash();
            if (hash == 0) {
                throw new AssertionError("Structural hash should not be zero");
            }
            System.out.println("  -> OK (Structural Hash: 0x" + Long.toHexString(hash) + ")");
        }
    }

    private static void testHypergraphBuilder() {
        System.out.println("[Test 3] Testing HypergraphBuilder fluent API...");
        try (Arena arena = Arena.ofConfined()) {
            HypergraphBuilder builder = new HypergraphBuilder();
            builder.addNode(new float[]{0.1f, 0.2f, 0.3f, 0.4f})
                    .addNodeFill(0.5f, 4)
                    .addNode(new float[]{0.9f, 0.8f, 0.7f, 0.6f})
                    .addEdge(0, 0, 1)
                    .addHyperedge(new long[]{0, 1}, new int[]{0, 1}, 2);

            Hypergraph graph = builder.build(arena);
            if (graph.getNumNodes() != 3) {
                throw new AssertionError("Expected 3 nodes, got: " + graph.getNumNodes());
            }
            if (graph.getNumHyperedges() != 2) {
                throw new AssertionError("Expected 2 hyperedges, got: " + graph.getNumHyperedges());
            }
            if (graph.getNodeFeatDim() != 4) {
                throw new AssertionError("Expected feat dim 4, got: " + graph.getNodeFeatDim());
            }

            graph.validate();
            long hash = graph.structuralHash();
            if (hash == 0) {
                throw new AssertionError("Expected non-zero structural hash");
            }
            System.out.println("  -> OK (Builder constructed valid graph, Hash: 0x" + Long.toHexString(hash) + ")");
        }
    }

    private static void testModelConfigPresets() {
        System.out.println("[Test 4] Testing ModelConfig presets & serialization...");
        ModelConfig small = ModelConfig.small();
        if (small.dModel() != 256 || small.numLayers() != 6) {
            throw new AssertionError("Unexpected small config values");
        }

        ModelConfig base = ModelConfig.base();
        if (base.dModel() != 512 || base.numLayers() != 12) {
            throw new AssertionError("Unexpected base config values");
        }

        ModelConfig large = ModelConfig.large();
        if (large.dModel() != 1024 || large.numLayers() != 24) {
            throw new AssertionError("Unexpected large config values");
        }

        String json = small.toJson();
        if (!json.contains("\"d_model\": 256") || !json.contains("\"num_layers\": 6")) {
            throw new AssertionError("Malformed JSON from config: " + json);
        }

        try (Arena arena = Arena.ofConfined()) {
            try (AxmanModel model = AxmanModel.fromConfig(small, arena)) {
                // Instantiated successfully
                if (model == null) {
                    throw new AssertionError("Model creation from config returned null");
                }
            }
        }
        System.out.println("  -> OK (Presets small/base/large verified, instantiated from config)");
    }

    private static void testModelForward() {
        System.out.println("[Test 5] Testing Model Load & Forward Pass...");
        String customConfig = "{\"d_model\": 64, \"num_layers\": 2, \"num_heads\": 4, \"num_actions\": 16, \"in_features\": 32}";

        try (Arena modelArena = Arena.ofConfined()) {
            try (AxmanModel model = AxmanModel.load(customConfig, modelArena)) {
                try (Arena graphArena = Arena.ofConfined()) {
                    float[] features = new float[3 * 32];
                    for (int i = 0; i < features.length; i++) features[i] = 0.25f;

                    long[] sources = new long[]{0, 0, 1};
                    int[] ports = new int[]{0, 0, 1};
                    long[] targets = new long[]{1, 2};
                    long[] offsets = new long[]{0, 1, 3};

                    Hypergraph graph = new Hypergraph(
                            graphArena,
                            3, 32,
                            features,
                            sources, ports, targets, offsets
                    );

                    try (InferenceResult result = model.forward(graph)) {
                        float val = result.getValueEstimate();
                        if (val < 0.0f || val > 1.0f) {
                            throw new AssertionError("Value estimate out of range: " + val);
                        }

                        long[] pShape = result.getPolicyShape();
                        if (pShape.length != 2 || pShape[0] != 3 || pShape[1] != 16) {
                            throw new AssertionError("Unexpected policy shape: " + java.util.Arrays.toString(pShape));
                        }

                        long[] lShape = result.getLatentShape();
                        if (lShape.length != 2 || lShape[0] != 1 || lShape[1] != 64) {
                            throw new AssertionError("Unexpected latent shape: " + java.util.Arrays.toString(lShape));
                        }

                        System.out.println("  -> OK (Value: " + val + ", Policy Shape: [" + pShape[0] + ", " + pShape[1] + "])");
                    }
                }
            }
        }
    }

    private static void testDirectConstructorAndSafetensorsLoad() throws Exception {
        System.out.println("[Test 6] Testing direct `new AxmanModel(path, arena)` with binary Safetensors...");
        Path tempPath = Files.createTempFile("axman_java_test", ".safetensors");

        try {
            float[] wData = new float[]{1.2f, -1.8f, 2.5f, 0.5f};
            float[] bData = new float[]{0.05f, -0.1f};

            String jsonHeader = "{\"__metadata__\":{\"d_model\":\"2\",\"num_layers\":\"1\",\"num_heads\":\"1\",\"in_features\":\"2\"},"
                    + "\"node_encoder.weight\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,16]},"
                    + "\"node_encoder.bias\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[16,24]}}";

            byte[] headerBytes = jsonHeader.getBytes(StandardCharsets.UTF_8);
            long headerLen = headerBytes.length;

            ByteBuffer buf = ByteBuffer.allocate(8 + (int) headerLen + (wData.length + bData.length) * 4);
            buf.order(ByteOrder.LITTLE_ENDIAN);
            buf.putLong(headerLen);
            buf.put(headerBytes);
            for (float w : wData) buf.putFloat(w);
            for (float b : bData) buf.putFloat(b);

            try (OutputStream os = Files.newOutputStream(tempPath)) {
                os.write(buf.array());
            }

            try (Arena arena = Arena.ofConfined()) {
                // Exactly tests `new AxmanModel(path, arena)` syntax specified in ROADMAP.md
                try (AxmanModel model = new AxmanModel(tempPath.toString(), arena)) {
                    HypergraphBuilder builder = new HypergraphBuilder();
                    builder.addNodeFill(1.0f, 2)
                            .addNodeFill(0.5f, 2)
                            .addEdge(0, 0, 1);

                    Hypergraph graph = builder.build(arena);
                    try (InferenceResult out = model.forward(graph)) {
                        float val = out.getValueEstimate();
                        if (val < 0.0f || val > 1.0f) {
                            throw new AssertionError("Value estimate out of range: " + val);
                        }
                        long[] pShape = out.getPolicyShape();
                        if (pShape[0] != 2) {
                            throw new AssertionError("Expected 2 node action logits rows, got: " + pShape[0]);
                        }
                        System.out.println("  -> OK (Safetensors loaded & verified, Value: " + val + ")");
                    }
                }
            }
        } finally {
            Files.deleteIfExists(tempPath);
        }
    }

    private static void testMultithreadedInference() throws Exception {
        System.out.println("[Test 7] Testing Multithreaded Parallel Inference (20 workers)...");
        String config = "{\"d_model\": 64, \"num_layers\": 1, \"num_heads\": 2, \"num_actions\": 8, \"in_features\": 16}";

        try (Arena modelArena = Arena.ofShared()) {
            AxmanModel model = AxmanModel.load(config, modelArena);

            int threads = 20;
            ExecutorService pool = Executors.newFixedThreadPool(threads);
            List<Future<Float>> futures = new ArrayList<>();

            for (int i = 0; i < threads; i++) {
                futures.add(pool.submit(() -> {
                    try (Arena callArena = Arena.ofConfined()) {
                        float[] feats = new float[2 * 16];
                        for (int j = 0; j < feats.length; j++) feats[j] = 0.5f;

                        Hypergraph g = new Hypergraph(
                                callArena,
                                2, 16,
                                feats,
                                new long[]{0},
                                new int[]{0},
                                new long[]{1},
                                new long[]{0, 1}
                        );

                        try (InferenceResult res = model.forward(g)) {
                            return res.getValueEstimate();
                        }
                    }
                }));
            }

            pool.shutdown();
            if (!pool.awaitTermination(10, TimeUnit.SECONDS)) {
                throw new TimeoutException("ThreadPool execution timed out");
            }

            for (Future<Float> f : futures) {
                float v = f.get();
                if (v < 0.0f || v > 1.0f) {
                    throw new AssertionError("Invalid value estimate: " + v);
                }
            }
            model.close();
            System.out.println("  -> OK (20 parallel tasks completed)");
        }
    }

    private static void testTopologyErrorHandling() {
        System.out.println("[Test 8] Testing Topology Validation & C-ABI Error Handling...");
        try (Arena arena = Arena.ofConfined()) {
            Hypergraph badGraph = new Hypergraph(
                    arena,
                    2, 16,
                    new float[2 * 16],
                    new long[]{0},
                    new int[]{0},
                    new long[]{999}, // Out of bounds!
                    new long[]{0, 1}
            );

            try {
                badGraph.validate();
                throw new AssertionError("Expected AxmanException on invalid topology");
            } catch (AxmanException e) {
                if (e.getStatusCode() != AxmanCabi.AXMAN_ERR_INDEX_OUT_OF_BOUNDS) {
                    throw new AssertionError("Expected AXMAN_ERR_INDEX_OUT_OF_BOUNDS, got: " + e.getStatusCode());
                }
                System.out.println("  -> OK (Intercepted expected exception: " + e.getMessage() + ")");
            }
        }
    }

    private static void testCsrOffsetValidationErrors() {
        System.out.println("[Test 9] Testing CSR offset invariant validation...");
        try (Arena arena = Arena.ofConfined()) {
            // Case 1: First offset != 0
            try {
                new Hypergraph(
                        arena, 2, 4,
                        new float[8],
                        new long[]{0}, new int[]{0}, new long[]{1},
                        new long[]{1, 1} // invalid first offset!
                );
                throw new AssertionError("Expected IllegalArgumentException for offset[0] != 0");
            } catch (IllegalArgumentException e) {
                // Expected
            }

            // Case 2: Last offset != sources.length
            try {
                new Hypergraph(
                        arena, 2, 4,
                        new float[8],
                        new long[]{0}, new int[]{0}, new long[]{1},
                        new long[]{0, 2} // last offset 2 != sources length 1!
                );
                throw new AssertionError("Expected IllegalArgumentException for last offset mismatch");
            } catch (IllegalArgumentException e) {
                // Expected
            }

            // Case 3: Decreasing offsets
            try {
                new Hypergraph(
                        arena, 3, 4,
                        new float[12],
                        new long[]{0, 1}, new int[]{0, 1}, new long[]{1, 2},
                        new long[]{0, 2, 1} // non-monotonic!
                );
                throw new AssertionError("Expected IllegalArgumentException for decreasing offsets");
            } catch (IllegalArgumentException e) {
                // Expected
            }

            System.out.println("  -> OK (All CSR offset validation invariants strictly enforced)");
        }
    }

    private static void testResourceSafetyAndDoubleClose() {
        System.out.println("[Test 10] Testing RAII resource safety & double close...");
        try (Arena arena = Arena.ofConfined()) {
            AxmanModel model = AxmanModel.load(arena);
            model.close();
            // Second close must be safe (idempotent)
            model.close();

            HypergraphBuilder builder = new HypergraphBuilder();
            builder.addNodeFill(1.0f, 64)
                    .addNodeFill(1.0f, 64)
                    .addEdge(0, 0, 1);
            Hypergraph g = builder.build(arena);

            try {
                model.forward(g);
                throw new AssertionError("Expected IllegalStateException when calling forward on closed model");
            } catch (IllegalStateException e) {
                // Expected
            }
            System.out.println("  -> OK (Double-close idempotent, post-close invocation safely rejected)");
        }
    }
}
