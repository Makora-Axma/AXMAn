package org.axma.axman;

import java.lang.foreign.Arena;
import java.util.ArrayList;
import java.util.List;

/**
 * Fluent builder for constructing zero-copy Port-Hypergraphs.
 */
public final class HypergraphBuilder {
    private int nodeFeatDim = -1;
    private final List<float[]> nodes = new ArrayList<>();
    private final List<Long> edgeSources = new ArrayList<>();
    private final List<Integer> edgePorts = new ArrayList<>();
    private final List<Long> edgeTargets = new ArrayList<>();
    private final List<Long> edgeOffsets = new ArrayList<>();

    public HypergraphBuilder() {
        edgeOffsets.add(0L);
    }

    /**
     * Add a node with arbitrary feature vector.
     */
    public HypergraphBuilder addNode(float[] features) {
        if (features == null || features.length == 0) {
            throw new IllegalArgumentException("Node features cannot be null or empty");
        }
        if (nodeFeatDim == -1) {
            nodeFeatDim = features.length;
        } else if (features.length != nodeFeatDim) {
            throw new IllegalArgumentException(
                    "Feature dimension mismatch: expected " + nodeFeatDim + ", got " + features.length
            );
        }
        nodes.add(features.clone());
        return this;
    }

    /**
     * Add a node filled with a scalar value across all feature dimensions.
     */
    public HypergraphBuilder addNodeFill(float val, int featDim) {
        if (featDim <= 0) {
            throw new IllegalArgumentException("Feature dimension must be positive");
        }
        float[] feats = new float[featDim];
        java.util.Arrays.fill(feats, val);
        return addNode(feats);
    }

    /**
     * Add a standard directed binary edge: source -> target with port index.
     */
    public HypergraphBuilder addEdge(long source, int port, long target) {
        edgeSources.add(source);
        edgePorts.add(port);
        edgeTargets.add(target);
        edgeOffsets.add((long) edgeSources.size());
        return this;
    }

    /**
     * Add an n-ary hyperedge: (sources[0]..sources[k-1]) -> target with corresponding ports.
     */
    public HypergraphBuilder addHyperedge(long[] sources, int[] ports, long target) {
        if (sources == null || ports == null) {
            throw new IllegalArgumentException("Sources and ports cannot be null");
        }
        if (sources.length != ports.length) {
            throw new IllegalArgumentException("Sources and ports length must match");
        }
        if (sources.length == 0) {
            throw new IllegalArgumentException("Hyperedge arity must be at least 1");
        }
        if (sources.length > 16) {
            throw new IllegalArgumentException("Hyperedge arity exceeds AXMAN_MAX_HYPEREDGE_ARITY (16)");
        }

        for (int i = 0; i < sources.length; i++) {
            edgeSources.add(sources[i]);
            edgePorts.add(ports[i]);
        }
        edgeTargets.add(target);
        edgeOffsets.add((long) edgeSources.size());
        return this;
    }

    /**
     * Build the finalized Hypergraph in the specified Arena.
     */
    public Hypergraph build(Arena arena) {
        int numNodes = nodes.size();
        int featDim = (nodeFeatDim == -1) ? 0 : nodeFeatDim;

        float[] flatFeatures = new float[numNodes * featDim];
        for (int i = 0; i < numNodes; i++) {
            System.arraycopy(nodes.get(i), 0, flatFeatures, i * featDim, featDim);
        }

        long[] sourcesArr = new long[edgeSources.size()];
        for (int i = 0; i < edgeSources.size(); i++) sourcesArr[i] = edgeSources.get(i);

        int[] portsArr = new int[edgePorts.size()];
        for (int i = 0; i < edgePorts.size(); i++) portsArr[i] = edgePorts.get(i);

        long[] targetsArr = new long[edgeTargets.size()];
        for (int i = 0; i < edgeTargets.size(); i++) targetsArr[i] = edgeTargets.get(i);

        long[] offsetsArr = new long[edgeOffsets.size()];
        for (int i = 0; i < edgeOffsets.size(); i++) offsetsArr[i] = edgeOffsets.get(i);

        return new Hypergraph(
                arena,
                numNodes,
                featDim,
                flatFeatures,
                sourcesArr,
                portsArr,
                targetsArr,
                offsetsArr
        );
    }
}
