package org.axma.axman;

public class AxmanException extends RuntimeException {
    private final int statusCode;

    public AxmanException(int statusCode, String message) {
        super("AXMAN Error [" + statusCode + "]: " + message);
        this.statusCode = statusCode;
    }

    public int getStatusCode() {
        return statusCode;
    }
}
