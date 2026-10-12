package org.axma.axman;

import java.io.FileInputStream;
import java.io.FileWriter;
import java.lang.foreign.Arena;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;

public class RunnerJava {
    public static void main(String[] args) throws Exception {
        String baseDir = args.length > 0 ? args[0] : ".";
        Path configPath = Path.of(baseDir, "golden_config.json");
        Path binPath = Path.of(baseDir, "golden_sample.bin");
        Path outPath = Path.of(baseDir, "golden_output_java.json");

        String configJson = Files.readString(configPath);
        byte[] binBytes = Files.readAllBytes(binPath);
        ByteBuffer buf = ByteBuffer.wrap(binBytes).order(ByteOrder.LITTLE_ENDIAN);

        byte[] magic = new byte[4];
        buf.get(magic); // "AXMG"
        long numNodes = buf.getLong();
        long featDim = buf.getLong();
        long numEdges = buf.getLong();
        long numSources = buf.getLong();

        float[] features = new float[(int) (numNodes * featDim)];
        for (int i = 0; i < features.length; i++) features[i] = buf.getFloat();

        long[] sources = new long[(int) numSources];
        for (int i = 0; i < sources.length; i++) sources[i] = buf.getLong();

        int[] ports = new int[(int) numSources];
        for (int i = 0; i < ports.length; i++) ports[i] = buf.getInt();

        long[] targets = new long[(int) numEdges];
        for (int i = 0; i < targets.length; i++) targets[i] = buf.getLong();

        long[] offsets = new long[(int) (numEdges + 1)];
        for (int i = 0; i < offsets.length; i++) offsets[i] = buf.getLong();

        try (Arena arena = Arena.ofConfined()) {
            Hypergraph graph = new Hypergraph(
                    arena,
                    numNodes, featDim,
                    features,
                    sources, ports, targets, offsets
            );
            graph.validate();
            long structuralHash = graph.structuralHash();

            try (AxmanModel model = AxmanModel.load(configJson, arena)) {
                try (InferenceResult res = model.forward(graph)) {
                    float val = res.getValueEstimate();
                    float[] policy = res.getPolicyLogits();
                    float[] latent = res.getLatentEmbedding();

                    try (FileWriter fw = new FileWriter(outPath.toFile())) {
                        fw.write("{\n");
                        fw.write("  \"language\": \"java\",\n");
                        // Format structural hash as unsigned 64-bit string or numeric
                        fw.write("  \"structural_hash\": " + Long.toUnsignedString(structuralHash) + ",\n");
                        fw.write("  \"value_estimate\": " + val + ",\n");
                        fw.write("  \"policy_logits\": [");
                        for (int i = 0; i < policy.length; i++) {
                            if (i > 0) fw.write(", ");
                            fw.write(Float.toString(policy[i]));
                        }
                        fw.write("],\n");
                        fw.write("  \"latent_embedding\": [");
                        for (int i = 0; i < latent.length; i++) {
                            if (i > 0) fw.write(", ");
                            fw.write(Float.toString(latent[i]));
                        }
                        fw.write("]\n");
                        fw.write("}\n");
                    }

                    System.out.println("[Java] Executed successfully. Structural hash: " + Long.toUnsignedString(structuralHash) + ", value: " + val);
                }
            }
        }
    }
}
