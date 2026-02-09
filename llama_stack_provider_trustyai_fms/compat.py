"""Compatibility helpers for importing llama_stack APIs across versions.

The llama_stack APIs were moved under a separate `llama_stack_api` package
upstream. Prefer the new package layout and fall back to the legacy one so
this provider can run with both.
"""

from __future__ import annotations

# Try new llama_stack_api package first, fall back to legacy llama_stack
try:  # Current dedicated llama_stack_api package (preferred)
    from llama_stack_api.datatypes import (
        Api,
        ProviderSpec,
        RemoteProviderSpec,
        ShieldsProtocolPrivate,
    )
    from llama_stack_api.inference import (
        OpenAIAssistantMessageParam,
        OpenAIDeveloperMessageParam,
        OpenAIMessageParam,
        OpenAISystemMessageParam,
        OpenAIToolMessageParam,
        OpenAIUserMessageParam,
        SystemMessage,
        ToolResponseMessage,
        UserMessage,
    )
    from llama_stack_api.resource import ResourceType
    from llama_stack_api.safety import (
        ModerationObject,
        ModerationObjectResults,
        RunShieldResponse,
        Safety,
        SafetyViolation,
        ShieldStore,
        ViolationLevel,
    )
    from llama_stack_api.safety.models import RunModerationRequest, RunShieldRequest
    from llama_stack_api.schema_utils import json_schema_type
    from llama_stack_api.shields import ListShieldsResponse, Shield, Shields

except ModuleNotFoundError:  # Legacy llama_stack layout
    from llama_stack.apis.datatypes import Api
    from llama_stack.apis.inference import (
        CompletionMessage,
        Message,
        SystemMessage,
        ToolResponseMessage,
        UserMessage,
    )
    from llama_stack.apis.resource import ResourceType
    from llama_stack.apis.safety import (
        RunShieldResponse,
        Safety,
        SafetyViolation,
        ShieldStore,
        ViolationLevel,
    )
    from llama_stack.apis.shields import ListShieldsResponse, Shield, Shields
    from llama_stack.providers.datatypes import (
        ProviderSpec,
        RemoteProviderSpec,
        ShieldsProtocolPrivate,
    )
    from llama_stack.schema_utils import json_schema_type

    OpenAIMessageParam = Message
    OpenAIUserMessageParam = UserMessage
    OpenAISystemMessageParam = SystemMessage
    OpenAIToolMessageParam = ToolResponseMessage
    OpenAIAssistantMessageParam = CompletionMessage
    OpenAIDeveloperMessageParam = SystemMessage  # Developer didn't exist

    try:
        from llama_stack.apis.safety import ModerationObject, ModerationObjectResults
    except ImportError:

        class ModerationObject:
            """Placeholder for legacy versions without ModerationObject"""

            pass

        class ModerationObjectResults:
            """Placeholder for legacy versions without ModerationObjectResults"""

            pass

    # Legacy versions don't have request models - create placeholders
    from pydantic import BaseModel

    class RunShieldRequest(BaseModel):
        """Legacy placeholder for RunShieldRequest"""

        shield_id: str
        messages: list

    class RunModerationRequest(BaseModel):
        """Legacy placeholder for RunModerationRequest"""

        input: str | list[str]
        model: str | None = None


__all__ = [
    "Api",
    "ListShieldsResponse",
    "ModerationObject",
    "ModerationObjectResults",
    "OpenAIAssistantMessageParam",
    "OpenAIDeveloperMessageParam",
    "OpenAIMessageParam",
    "OpenAISystemMessageParam",
    "OpenAIToolMessageParam",
    "OpenAIUserMessageParam",
    "ProviderSpec",
    "RemoteProviderSpec",
    "ResourceType",
    "RunModerationRequest",
    "RunShieldRequest",
    "RunShieldResponse",
    "Safety",
    "SafetyViolation",
    "Shield",
    "Shields",
    "ShieldsProtocolPrivate",
    "ShieldStore",
    "SystemMessage",
    "ToolResponseMessage",
    "UserMessage",
    "ViolationLevel",
    "json_schema_type",
]
