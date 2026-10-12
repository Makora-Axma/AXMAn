package org.axma.axman;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.nio.file.Path;

/**
 * Low-level FFI bindings to libaxman using Java 22 Project Panama (FFM API / JEP 454).
 */
final class AxmanCabi {

    // Status codes
    static final int AXMAN_SUCCESS = 0;
    static final int AXMAN_ERR_INVALID_ARGUMENT = 1;
    static final int AXMAN_ERR_OUT_OF_MEMORY = 2;
    static final int AXMAN_ERR_DEVICE_UNAVAILABLE = 3;
    static final int AXMAN_ERR_INFERENCE_FAILURE = 4;
    static final int AXMAN_ERR_NULL_POINTER = 5;
    static final int AXMAN_ERR_ARITY_OVERFLOW = 6;
    static final int AXMAN_ERR_INDEX_OUT_OF_BOUNDS = 7;
    static final int AXMAN_ERR_INVALID_TOPOLOGY = 8;
    static final int AXMAN_ERR_INTERNAL = 99;

    // Dtype & Device
    static final int AXMAN_DTYPE_FLOAT32 = 0;
    static final int AXMAN_DEVICE_CPU = 0;

    // Memory Layouts
    // struct axman_tensor_t { void* data; int64_t* shape; size_t ndim; int32_t dtype; int32_t device; }
    static final GroupLayout TENSOR_LAYOUT = MemoryLayout.structLayout(
            ValueLayout.ADDRESS.withName("data"),
            ValueLayout.ADDRESS.withName("shape"),
            ValueLayout.JAVA_LONG.withName("ndim"),
            ValueLayout.JAVA_INT.withName("dtype"),
            ValueLayout.JAVA_INT.withName("device")
    );

    // struct axman_hypergraph_t
    static final GroupLayout HYPERGRAPH_LAYOUT = MemoryLayout.structLayout(
            ValueLayout.JAVA_LONG.withName("num_nodes"),
            ValueLayout.JAVA_LONG.withName("num_hyperedges"),
            TENSOR_LAYOUT.withName("node_features"),
            ValueLayout.ADDRESS.withName("edge_sources"),
            ValueLayout.ADDRESS.withName("edge_ports"),
            ValueLayout.ADDRESS.withName("edge_targets"),
            ValueLayout.ADDRESS.withName("edge_offsets")
    );

    // struct axman_output_t { axman_tensor_t policy_logits; float value_estimate; axman_tensor_t latent_embedding; }
    static final GroupLayout OUTPUT_LAYOUT = MemoryLayout.structLayout(
            TENSOR_LAYOUT.withName("policy_logits"),
            ValueLayout.JAVA_FLOAT.withName("value_estimate"),
            MemoryLayout.paddingLayout(4), // alignment padding to 8 bytes for nested struct
            TENSOR_LAYOUT.withName("latent_embedding")
    );

    // Downcall MethodHandles
    static final MethodHandle MH_MODEL_CREATE;
    static final MethodHandle MH_MODEL_FORWARD;
    static final MethodHandle MH_MODEL_DESTROY;
    static final MethodHandle MH_OUTPUT_FREE;
    static final MethodHandle MH_HYPERGRAPH_VALIDATE;
    static final MethodHandle MH_HYPERGRAPH_HASH;
    static final MethodHandle MH_GET_LAST_ERROR_MESSAGE;
    static final MethodHandle MH_GET_VERSION;
    static final MethodHandle MH_STATUS_TO_STRING;

    static {
        // Resolve libaxman library
        String libName = System.mapLibraryName("axman");
        Path[] candidatePaths = new Path[]{
                Path.of("libaxman.dll").toAbsolutePath(),
                Path.of("build/libaxman.dll").toAbsolutePath(),
                Path.of("../../libaxman.dll").toAbsolutePath(),
                Path.of("../../build/libaxman.dll").toAbsolutePath(),
                Path.of("D:/DEV/Makora/Makora AXMA/axman/libaxman.dll").toAbsolutePath(),
                Path.of("D:/DEV/Makora/Makora AXMA/axman/build/libaxman.dll").toAbsolutePath(),
        };

        Path resolved = null;
        for (Path p : candidatePaths) {
            if (p.toFile().exists()) {
                resolved = p;
                break;
            }
        }

        Arena globalArena = Arena.global();
        SymbolLookup lookup;
        if (resolved != null) {
            lookup = SymbolLookup.libraryLookup(resolved, globalArena);
        } else {
            lookup = SymbolLookup.libraryLookup(libName, globalArena);
        }

        Linker linker = Linker.nativeLinker();

        MH_MODEL_CREATE = linker.downcallHandle(
                lookup.find("axman_model_create").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS)
        );

        MH_MODEL_FORWARD = linker.downcallHandle(
                lookup.find("axman_model_forward").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS)
        );

        MH_MODEL_DESTROY = linker.downcallHandle(
                lookup.find("axman_model_destroy").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS)
        );

        MH_OUTPUT_FREE = linker.downcallHandle(
                lookup.find("axman_output_free").orElseThrow(),
                FunctionDescriptor.ofVoid(ValueLayout.ADDRESS)
        );

        MH_HYPERGRAPH_VALIDATE = linker.downcallHandle(
                lookup.find("axman_hypergraph_validate").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS)
        );

        MH_HYPERGRAPH_HASH = linker.downcallHandle(
                lookup.find("axman_hypergraph_hash").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS)
        );

        MH_GET_LAST_ERROR_MESSAGE = linker.downcallHandle(
                lookup.find("axman_get_last_error_message").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.ADDRESS)
        );

        MH_GET_VERSION = linker.downcallHandle(
                lookup.find("axman_get_version").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.ADDRESS)
        );

        MH_STATUS_TO_STRING = linker.downcallHandle(
                lookup.find("axman_status_to_string").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.JAVA_INT)
        );
    }

    private AxmanCabi() {}
}
