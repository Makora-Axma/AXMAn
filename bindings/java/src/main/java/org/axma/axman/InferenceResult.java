package org.axma.axman;

import java.lang.foreign.*;

/**
 * Output container from model forward pass.
 * Implements AutoCloseable to ensure axman_output_free is called.
 */
public final class InferenceResult implements AutoCloseable {
    private final MemorySegment rawOutput;
    private final Arena resultArena;
    private final float valueEstimate;
    private final float[] policyLogits;
    private final long[] policyShape;
    private final float[] latentEmbedding;
    private final long[] latentShape;
    private boolean closed = false;

    InferenceResult(MemorySegment rawOutput, Arena resultArena) {
        this.rawOutput = rawOutput;
        this.resultArena = resultArena;

        this.valueEstimate = rawOutput.get(
                ValueLayout.JAVA_FLOAT,
                AxmanCabi.OUTPUT_LAYOUT.byteOffset(MemoryLayout.PathElement.groupElement("value_estimate"))
        );

        // Read policy logits
        long policyOffset = AxmanCabi.OUTPUT_LAYOUT.byteOffset(MemoryLayout.PathElement.groupElement("policy_logits"));
        MemorySegment policyData = rawOutput.get(ValueLayout.ADDRESS, policyOffset + AxmanCabi.TENSOR_LAYOUT.byteOffset(MemoryLayout.PathElement.groupElement("data")));
        MemorySegment policyShapeSeg = rawOutput.get(ValueLayout.ADDRESS, policyOffset + AxmanCabi.TENSOR_LAYOUT.byteOffset(MemoryLayout.PathElement.groupElement("shape")));
        long policyNdim = rawOutput.get(ValueLayout.JAVA_LONG, policyOffset + AxmanCabi.TENSOR_LAYOUT.byteOffset(MemoryLayout.PathElement.groupElement("ndim")));

        this.policyShape = readShape(policyShapeSeg, policyNdim);
        this.policyLogits = readFloatArray(policyData, computeTotalElements(policyShape));

        // Read latent embedding
        long latentOffset = AxmanCabi.OUTPUT_LAYOUT.byteOffset(MemoryLayout.PathElement.groupElement("latent_embedding"));
        MemorySegment latentData = rawOutput.get(ValueLayout.ADDRESS, latentOffset + AxmanCabi.TENSOR_LAYOUT.byteOffset(MemoryLayout.PathElement.groupElement("data")));
        MemorySegment latentShapeSeg = rawOutput.get(ValueLayout.ADDRESS, latentOffset + AxmanCabi.TENSOR_LAYOUT.byteOffset(MemoryLayout.PathElement.groupElement("shape")));
        long latentNdim = rawOutput.get(ValueLayout.JAVA_LONG, latentOffset + AxmanCabi.TENSOR_LAYOUT.byteOffset(MemoryLayout.PathElement.groupElement("ndim")));

        this.latentShape = readShape(latentShapeSeg, latentNdim);
        this.latentEmbedding = readFloatArray(latentData, computeTotalElements(latentShape));
    }

    private static long[] readShape(MemorySegment shapeSeg, long ndim) {
        if (shapeSeg.equals(MemorySegment.NULL) || ndim <= 0) {
            return new long[0];
        }
        MemorySegment sized = shapeSeg.reinterpret(ndim * ValueLayout.JAVA_LONG.byteSize());
        long[] res = new long[(int) ndim];
        MemorySegment.copy(sized, ValueLayout.JAVA_LONG, 0L, res, 0, (int) ndim);
        return res;
    }

    private static float[] readFloatArray(MemorySegment dataSeg, int total) {
        if (dataSeg.equals(MemorySegment.NULL) || total <= 0) {
            return new float[0];
        }
        MemorySegment sized = dataSeg.reinterpret((long) total * ValueLayout.JAVA_FLOAT.byteSize());
        float[] res = new float[total];
        MemorySegment.copy(sized, ValueLayout.JAVA_FLOAT, 0L, res, 0, total);
        return res;
    }

    private static int computeTotalElements(long[] shape) {
        if (shape.length == 0) return 0;
        int total = 1;
        for (long s : shape) {
            total *= (int) s;
        }
        return total;
    }

    public float getValueEstimate() {
        return valueEstimate;
    }

    public float[] getPolicyLogits() {
        return policyLogits;
    }

    public long[] getPolicyShape() {
        return policyShape;
    }

    public float[] getLatentEmbedding() {
        return latentEmbedding;
    }

    public long[] getLatentShape() {
        return latentShape;
    }

    @Override
    public void close() {
        if (!closed) {
            try {
                AxmanCabi.MH_OUTPUT_FREE.invokeExact(rawOutput);
            } catch (Throwable t) {
                throw new RuntimeException("Failed to free native output buffers", t);
            } finally {
                resultArena.close();
                closed = true;
            }
        }
    }
}
