from unittest.mock import AsyncMock, MagicMock

import pytest

from llama_stack_provider_trustyai_fms.compat import (
    OpenAIUserMessageParam,
    RunModerationRequest,
)


@pytest.mark.asyncio
async def test_run_moderation_flagged():
    from llama_stack_provider_trustyai_fms.detectors.base import DetectorProvider

    provider = DetectorProvider(detectors={})
    provider._get_shield_id_from_model = AsyncMock(return_value="test_shield")
    provider._convert_input_to_messages = MagicMock(
        return_value=[
            OpenAIUserMessageParam(content="bad message", role="user"),
            OpenAIUserMessageParam(content="good message", role="user"),
        ]
    )

    # Simulate shield_response with one flagged and one not
    class FakeViolation:
        violation_level = "error"
        user_message = "violation"
        metadata = {
            "results": [
                {
                    "message_index": 0,
                    "detection_type": "LABEL_1",
                    "score": 0.99,
                    "status": "violation",
                },
                {
                    "message_index": 1,
                    "detection_type": None,
                    "score": None,
                    "status": "pass",
                },
            ]
        }

    class FakeShieldResponse:
        violation = FakeViolation()

    provider.run_shield = AsyncMock(return_value=FakeShieldResponse())

    result = await provider.run_moderation(
        RunModerationRequest(input=["bad message", "good message"], model="test_model")
    )
    assert len(result.results) == 2
    assert result.results[0].flagged is True
    assert result.results[1].flagged is False
    assert result.results[0].user_message == "bad message"
    assert result.results[1].user_message == "good message"


@pytest.mark.asyncio
async def test_run_moderation_error():
    from llama_stack_provider_trustyai_fms.detectors.base import DetectorProvider

    provider = DetectorProvider(detectors={})
    provider._get_shield_id_from_model = AsyncMock(side_effect=Exception("fail"))
    provider._convert_input_to_messages = MagicMock(
        return_value=[OpenAIUserMessageParam(content="msg", role="user")]
    )

    result = await provider.run_moderation(
        RunModerationRequest(input=["msg"], model="test_model")
    )
    assert len(result.results) == 1
    assert result.results[0].flagged is False
    assert "fail" in result.results[0].metadata["error"]


@pytest.mark.asyncio
async def test_run_moderation_empty_input():
    from llama_stack_provider_trustyai_fms.detectors.base import DetectorProvider

    provider = DetectorProvider(detectors={})
    provider._get_shield_id_from_model = AsyncMock(return_value="test_shield")
    provider._convert_input_to_messages = MagicMock(return_value=[])
    provider.run_shield = AsyncMock()
    result = await provider.run_moderation(
        RunModerationRequest(input=[], model="test_model")
    )
    assert len(result.results) == 0


@pytest.mark.asyncio
async def test_run_moderation_single_string_input():
    from llama_stack_provider_trustyai_fms.detectors.base import DetectorProvider

    provider = DetectorProvider(detectors={})
    provider._get_shield_id_from_model = AsyncMock(return_value="test_shield")
    provider._convert_input_to_messages = MagicMock(
        return_value=[OpenAIUserMessageParam(content="one message", role="user")]
    )
    provider.run_shield = AsyncMock(
        return_value=MagicMock(
            violation=MagicMock(
                metadata={
                    "results": [
                        {
                            "message_index": 0,
                            "detection_type": None,
                            "score": None,
                            "status": "pass",
                        }
                    ]
                }
            )
        )
    )
    result = await provider.run_moderation(
        RunModerationRequest(input="one message", model="test_model")
    )
    assert len(result.results) == 1
    assert result.results[0].user_message == "one message"
