package org.axma.axman;

/**
 * Strongly-typed architectural configuration for AXMANet models.
 */
public record ModelConfig(
        int dModel,
        int numLayers,
        int numHeads,
        int portDim,
        int numKinds,
        int inFeatures,
        int numActions,
        int maxPorts
) {
    public static ModelConfig small() {
        return new ModelConfig(256, 6, 8, 64, 16, 64, 128, 16);
    }

    public static ModelConfig base() {
        return new ModelConfig(512, 12, 16, 128, 16, 64, 256, 16);
    }

    public static ModelConfig large() {
        return new ModelConfig(1024, 24, 16, 256, 16, 128, 512, 16);
    }

    public String toJson() {
        return String.format(
                "{\"d_model\": %d, \"num_layers\": %d, \"num_heads\": %d, \"port_dim\": %d, \"num_kinds\": %d, \"in_features\": %d, \"num_actions\": %d, \"max_ports\": %d}",
                dModel, numLayers, numHeads, portDim, numKinds, inFeatures, numActions, maxPorts
        );
    }
}
