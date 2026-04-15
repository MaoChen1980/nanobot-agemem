"""Tests for RouterBus: register_route, start_router, _route_loop, consume_default_inbound."""

from __future__ import annotations

import asyncio

import pytest

from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.router import RouterBus


@pytest.mark.asyncio
async def test_register_route_returns_queue():
    bus = RouterBus()

    def predicate(msg: InboundMessage) -> bool:
        return msg.content.startswith("/test")

    q = bus.register_route("test_consumer", predicate)
    assert isinstance(q, asyncio.Queue)


@pytest.mark.asyncio
async def test_unregister_route():
    bus = RouterBus()

    def predicate(msg: InboundMessage) -> bool:
        return True

    bus.register_route("consumer1", predicate)
    bus.unregister_route("consumer1")
    # Should not raise


@pytest.mark.asyncio
async def test_routed_message_goes_to_consumer_queue():
    bus = RouterBus()

    routed_queue = bus.register_route("test", lambda m: m.content == "go")

    await bus.start_router()
    try:
        await bus.publish_inbound(InboundMessage(
            channel="cli",
            sender_id="user",
            chat_id="c1",
            content="go",
        ))

        # Give router a chance to dispatch
        await asyncio.sleep(0.1)

        msg = await asyncio.wait_for(routed_queue.get(), timeout=2.0)
        assert msg.content == "go"
    finally:
        await bus.stop_router()


@pytest.mark.asyncio
async def test_unmatched_message_goes_to_default_queue():
    bus = RouterBus()

    bus.register_route("test", lambda m: m.content == "routed")

    await bus.start_router()
    try:
        await bus.publish_inbound(InboundMessage(
            channel="cli",
            sender_id="user",
            chat_id="c1",
            content="not routed",
        ))

        await asyncio.sleep(0.1)

        msg = await asyncio.wait_for(bus.consume_default_inbound(), timeout=2.0)
        assert msg.content == "not routed"
    finally:
        await bus.stop_router()


@pytest.mark.asyncio
async def test_multiple_routes_predicate_checking():
    bus = RouterBus()

    queue_a = bus.register_route("a", lambda m: m.content.startswith("a"))
    queue_b = bus.register_route("b", lambda m: m.content.startswith("b"))

    await bus.start_router()
    try:
        await bus.publish_inbound(InboundMessage(
            channel="cli", sender_id="user", chat_id="c1", content="a message"
        ))
        await asyncio.sleep(0.1)

        msg_a = await asyncio.wait_for(queue_a.get(), timeout=2.0)
        assert msg_a.content == "a message"

        # queue_b should be empty
        assert queue_b.empty()
    finally:
        await bus.stop_router()


@pytest.mark.asyncio
async def test_outbound_still_works():
    bus = RouterBus()
    await bus.start_router()
    try:
        await bus.publish_outbound(OutboundMessage(
            channel="cli",
            chat_id="c1",
            content="response",
        ))

        msg = await asyncio.wait_for(bus.consume_outbound(), timeout=2.0)
        assert msg.content == "response"
    finally:
        await bus.stop_router()


@pytest.mark.asyncio
async def test_stop_router_idempotent():
    bus = RouterBus()
    await bus.start_router()
    await bus.stop_router()
    await bus.stop_router()  # Should not raise