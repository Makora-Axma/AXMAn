package org.axma.axman;

import java.lang.foreign.*;

/**
 * High-level AXMANet model wrapper backed by Panama FFM API.
 * Implements AutoCloseable to guarantee invocation of `axman_model_destroy`.
 */
public final class AxmanModel implements AutoCloseable {
    private final MemorySegment modelHandle;
    private boolean closed = false;

    public AxmanModel(String modelPath, String configJson, Arena lifecycleArena) {
        try {
            MemorySegment pathSeg = modelPath == null ? MemorySegment.NULL : lifecycleArena.allocateFrom(modelPath);
            MemorySegment jsonSeg = configJson == null ? MemorySegment.NULL : lifecycleArena.allocateFrom(configJson);
            MemorySegment outHandleSeg = lifecycleArena.allocate(ValueLayout.ADDRESS);

            int status = (int) AxmanCabi.MH_MODEL_CREATE.invokeExact(pathSeg, jsonSeg, outHandleSeg);
            if (status != AxmanCabi.AXMAN_SUCCESS) {
                throw Hypergraph.makeException(status);
            }

            this.modelHandle = outHandleSeg.get(ValueLayout.ADDRESS, 0);
            if (this.modelHandle.equals(MemorySegment.NULL)) {
                throw new AxmanException(AxmanCabi.AXMAN_ERR_NULL_POINTER, "Model handle is NULL after create");
            }
        } catch (AxmanException e) {
            throw e;
        } catch (Throwable t) {
            throw new RuntimeException("Model creation failed", t);
        }
    }

    public AxmanModel(String modelPath, Arena lifecycleArena) {
        this(modelPath, (String) null, lifecycleArena);
    }

    public AxmanModel(ModelConfig config, Arena lifecycleArena) {
        this(null, config != null ? config.toJson() : null, lifecycleArena);
    }

    public static AxmanModel fromFile(String modelPath, Arena arena) {
        return new AxmanModel(modelPath, (String) null, arena);
    }

    public static AxmanModel fromConfig(ModelConfig config, Arena arena) {
        return new AxmanModel(config, arena);
    }

    public static AxmanModel fromConfig(String configJson, Arena arena) {
        return new AxmanModel(null, configJson, arena);
    }

    public static AxmanModel load(String configJson, Arena arena) {
        return new AxmanModel(null, configJson, arena);
    }

    public static AxmanModel load(Arena arena) {
        return new AxmanModel(null, null, arena);
    }

    public InferenceResult forward(Hypergraph graph) {
        if (closed || modelHandle.equals(MemorySegment.NULL)) {
            throw new IllegalStateException("Model is closed or invalid");
        }

        Arena resultArena = Arena.ofConfined();
        try {
            MemorySegment outputSeg = resultArena.allocate(AxmanCabi.OUTPUT_LAYOUT);

            int status = (int) AxmanCabi.MH_MODEL_FORWARD.invokeExact(modelHandle, graph.getSegment(), outputSeg);
            if (status != AxmanCabi.AXMAN_SUCCESS) {
                resultArena.close();
                throw Hypergraph.makeException(status);
            }

            return new InferenceResult(outputSeg, resultArena);
        } catch (AxmanException e) {
            resultArena.close();
            throw e;
        } catch (Throwable t) {
            resultArena.close();
            throw new RuntimeException("Forward inference failed", t);
        }
    }

    public static String getVersion() {
        try {
            MemorySegment vPtr = (MemorySegment) AxmanCabi.MH_GET_VERSION.invokeExact();
            if (vPtr.equals(MemorySegment.NULL)) {
                return "0.1.0";
            }
            return vPtr.reinterpret(64).getString(0);
        } catch (Throwable t) {
            return "0.1.0";
        }
    }

    @Override
    public void close() {
        if (!closed && !modelHandle.equals(MemorySegment.NULL)) {
            try {
                int status = (int) AxmanCabi.MH_MODEL_DESTROY.invokeExact(modelHandle);
            } catch (Throwable t) {
                throw new RuntimeException("Failed to destroy model handle", t);
            }
            closed = true;
        }
    }
}
