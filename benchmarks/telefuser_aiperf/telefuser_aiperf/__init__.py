"""TeleFuser-owned AIPerf integration."""

from __future__ import annotations

from aiperf.streaming.adapters import register_stream_adapter

from telefuser_aiperf.adapter import TeleFuserLiveKitAdapter
from telefuser_aiperf.sglang_adapter import SGLangRealtimeAdapter
from telefuser_aiperf.vla_structured import (
    ENDPOINT_METADATA,
    TRANSPORT_METADATA,
    TeleFuserStructuredHttpTransport,
    TeleFuserVlaStructuredEndpoint,
)


def register_adapters(*, replace: bool = False) -> None:
    """Register TeleFuser adapters in the current AIPerf process."""

    register_stream_adapter(
        "telefuser_livekit",
        TeleFuserLiveKitAdapter,
        replace=replace,
    )
    register_stream_adapter(
        "sglang_realtime",
        SGLangRealtimeAdapter,
        replace=replace,
    )


def register_plugins(*, replace: bool = False) -> None:
    """Register repository-owned AIPerf batch endpoint and transport plugins."""
    from aiperf.plugin import plugins
    from aiperf.plugin.enums import EndpointType, TransportType

    if "telefuser_vla_structured" not in EndpointType:
        EndpointType.register("TELEFUSER_VLA_STRUCTURED", "telefuser_vla_structured")
    if "telefuser_structured_http" not in TransportType:
        TransportType.register("TELEFUSER_STRUCTURED_HTTP", "telefuser_structured_http")

    definitions = (
        (
            "endpoint",
            "telefuser_vla_structured",
            TeleFuserVlaStructuredEndpoint,
            ENDPOINT_METADATA,
        ),
        (
            "transport",
            "telefuser_structured_http",
            TeleFuserStructuredHttpTransport,
            TRANSPORT_METADATA,
        ),
    )
    for category, name, plugin_class, metadata in definitions:
        if plugins.has_entry(category, name):
            if not replace:
                continue
            plugins.unregister(category, name)
        plugins.register(category, name, plugin_class, metadata=metadata)


__all__ = [
    "SGLangRealtimeAdapter",
    "TeleFuserLiveKitAdapter",
    "TeleFuserStructuredHttpTransport",
    "TeleFuserVlaStructuredEndpoint",
    "register_adapters",
    "register_plugins",
]
