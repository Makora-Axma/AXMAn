package org.axma.axman;

import java.lang.foreign.*;
import java.util.Arrays;

/**
 * High-level zero-copy Port-Hypergraph container backed by Panama MemorySegment.
 */
public final class Hypergraph {
    private final Arena arena;
    private final MemorySegment structSegment;
    private final long numNodes;
    private final long numHyperedges;
    private final long nodeFeatDim;

    public Hypergraph(
            Arena arena,
            long numNodes,
            long nodeFeatDim,
            float[] nodeFeatures,
            long[] edgeSources,
            int[] edgePorts,
            long[] edgeTargets,
            long[] edgeOffsets
    ) {
        if (nodeFeatures.length != numNodes * nodeFeatDim) {
            throw new IllegalArgumentException("nodeFeatures length does not match numNodes * nodeFeatDim");
        }
        if (edgeTargets.length != edgeOffsets.length - 1) {
            throw new IllegalArgumentException("edgeOffsets length must be edgeTargets.length + 1");
        }
        if (edgeSources.length != edgePorts.length) {
            throw new IllegalArgumentException("edgeSources and edgePorts must have the same length");
        }
        if (edgeOffsets.length == 0 || edgeOffsets[0] != 0L) {
            throw new IllegalArgumentException("First edge offset must be 0");
        }
        if (edgeOffsets[edgeOffsets.length - 1] != (long) edgeSources.length) {
            throw new IllegalArgumentException("Last edge offset must match total number of edge sources (" + edgeSources.length + ")");
        }
        for (int i = 0; i < edgeOffsets.length - 1; i++) {
            if (edgeOffsets[i] > edgeOffsets[i + 1]) {
                throw new IllegalArgumentException("edgeOffsets must be monotonically non-decreasing at index " + i);
            }
        }

        this.arena = arena;
        this.numNodes = numNodes;
        this.numHyperedges = edgeTargets.length;
        this.nodeFeatDim = nodeFeatDim;

        // Allocate native buffers in the arena (pass NULL if empty)
        MemorySegment featData = nodeFeatures.length > 0
                ? arena.allocateFrom(ValueLayout.JAVA_FLOAT, nodeFeatures)
                : MemorySegment.NULL;
        MemorySegment shapeSeg = numNodes > 0
                ? arena.allocateFrom(ValueLayout.JAVA_LONG, numNodes, nodeFeatDim)
                : MemorySegment.NULL;
        MemorySegment sourcesSeg = edgeSources.length > 0
                ? arena.allocateFrom(ValueLayout.JAVA_LONG, edgeSources)
                : MemorySegment.NULL;
        MemorySegment portsSeg = edgePorts.length > 0
                ? arena.allocateFrom(ValueLayout.JAVA_INT, edgePorts)
                : MemorySegment.NULL;
        MemorySegment targetsSeg = edgeTargets.length > 0
                ? arena.allocateFrom(ValueLayout.JAVA_LONG, edgeTargets)
                : MemorySegment.NULL;
        MemorySegment offsetsSeg = edgeOffsets.length > 0
                ? arena.allocateFrom(ValueLayout.JAVA_LONG, edgeOffsets)
                : MemorySegment.NULL;

        // Allocate struct axman_hypergraph_t
        this.structSegment = arena.allocate(AxmanCabi.HYPERGRAPH_LAYOUT);

        // Populate fields
        structSegment.set(ValueLayout.JAVA_LONG, AxmanCabi.HYPERGRAPH_LAYOUT.byteOffset(MemoryLayout.PathElement.groupElement("num_nodes")), numNodes);
        structSegment.set(ValueLayout.JAVA_LONG, AxmanCabi.HYPERGRAPH_LAYOUT.byteOffset(MemoryLayout.PathElement.groupElement("num_hyperedges")), numHyperedges);

        // Populate nested node_features tensor
        long featOffset = AxmanCabi.HYPERGRAPH_LAYOUT.byteOffset(MemoryLayout.PathElement.groupElement("node_features"));
        structSegment.set(ValueLayout.ADDRESS, featOffset + AxmanCabi.TENSOR_LAYOUT.byteOffset(MemoryLayout.PathElement.groupElement("data")), featData);
        structSegment.set(ValueLayout.ADDRESS, featOffset + AxmanCabi.TENSOR_LAYOUT.byteOffset(MemoryLayout.PathElement.groupElement("shape")), shapeSeg);
        structSegment.set(ValueLayout.JAVA_LONG, featOffset + AxmanCabi.TENSOR_LAYOUT.byteOffset(MemoryLayout.PathElement.groupElement("ndim")), 2L);
        structSegment.set(ValueLayout.JAVA_INT, featOffset + AxmanCabi.TENSOR_LAYOUT.byteOffset(MemoryLayout.PathElement.groupElement("dtype")), AxmanCabi.AXMAN_DTYPE_FLOAT32);
        structSegment.set(ValueLayout.JAVA_INT, featOffset + AxmanCabi.TENSOR_LAYOUT.byteOffset(MemoryLayout.PathElement.groupElement("device")), AxmanCabi.AXMAN_DEVICE_CPU);

        // Pointer fields
        structSegment.set(ValueLayout.ADDRESS, AxmanCabi.HYPERGRAPH_LAYOUT.byteOffset(MemoryLayout.PathElement.groupElement("edge_sources")), sourcesSeg);
        structSegment.set(ValueLayout.ADDRESS, AxmanCabi.HYPERGRAPH_LAYOUT.byteOffset(MemoryLayout.PathElement.groupElement("edge_ports")), portsSeg);
        structSegment.set(ValueLayout.ADDRESS, AxmanCabi.HYPERGRAPH_LAYOUT.byteOffset(MemoryLayout.PathElement.groupElement("edge_targets")), targetsSeg);
        structSegment.set(ValueLayout.ADDRESS, AxmanCabi.HYPERGRAPH_LAYOUT.byteOffset(MemoryLayout.PathElement.groupElement("edge_offsets")), offsetsSeg);
    }

    public long getNumNodes() {
        return numNodes;
    }

    public long getNumHyperedges() {
        return numHyperedges;
    }

    public long getNodeFeatDim() {
        return nodeFeatDim;
    }

    MemorySegment getSegment() {
        return structSegment;
    }

    public void validate() {
        try {
            int status = (int) AxmanCabi.MH_HYPERGRAPH_VALIDATE.invokeExact(structSegment);
            if (status != AxmanCabi.AXMAN_SUCCESS) {
                throw makeException(status);
            }
        } catch (AxmanException e) {
            throw e;
        } catch (Throwable t) {
            throw new RuntimeException("Validation invoke failed", t);
        }
    }

    public long structuralHash() {
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment hashSeg = tmp.allocate(ValueLayout.JAVA_LONG);
            int status = (int) AxmanCabi.MH_HYPERGRAPH_HASH.invokeExact(structSegment, hashSeg);
            if (status != AxmanCabi.AXMAN_SUCCESS) {
                throw makeException(status);
            }
            return hashSeg.get(ValueLayout.JAVA_LONG, 0);
        } catch (AxmanException e) {
            throw e;
        } catch (Throwable t) {
            throw new RuntimeException("Structural hash invoke failed", t);
        }
    }

    static AxmanException makeException(int status) {
        try {
            MemorySegment msgPtr = (MemorySegment) AxmanCabi.MH_GET_LAST_ERROR_MESSAGE.invokeExact();
            String msg = null;
            if (!msgPtr.equals(MemorySegment.NULL)) {
                String raw = msgPtr.reinterpret(1024).getString(0);
                if (!raw.isBlank() && !raw.contains("No error reported")) {
                    msg = raw;
                }
            }
            if (msg == null) {
                MemorySegment statusStrPtr = (MemorySegment) AxmanCabi.MH_STATUS_TO_STRING.invokeExact(status);
                if (!statusStrPtr.equals(MemorySegment.NULL)) {
                    msg = statusStrPtr.reinterpret(256).getString(0);
                } else {
                    msg = "Error status " + status;
                }
            }
            return new AxmanException(status, msg);
        } catch (Throwable t) {
            return new AxmanException(status, "Status code: " + status);
        }
    }
}
