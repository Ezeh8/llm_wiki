class AdapterValidationError(Exception):
    """Raised when an adapter's output fails its post-conditions.

    Adapters validate outputs before returning to nodes (D-7/D-11). This signals a
    contract violation from the underlying store — not a transient fault. Nodes own
    retry policy; adapters never retry.
    """
