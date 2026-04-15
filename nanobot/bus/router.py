"""RouterBus: a MessageBus wrapper that routes messages to registered consumers."""

from __future__ import annotations

import asyncio
from typing import Callable, Awaitable

from loguru import logger

from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus


class RouterBus(MessageBus):
    """MessageBus with routing support.

    Allows multiple consumers to register handlers for specific message patterns.
    The router dispatches inbound messages to the appropriate consumer based on
    registered predicates. Unmatched messages are placed on the default inbound
    queue for AgentLoop to consume.

    Outbound messages are still fan-out to the single outbound queue.
    """

    def __init__(self):
        # Shadow MessageBus.inbound with our own router-aware queue.
        # MessageBus expects self.inbound as a Queue, so we set it directly.
        self._inbound: asyncio.Queue[InboundMessage] = asyncio.Queue()
        self._default_inbound: asyncio.Queue[InboundMessage] = asyncio.Queue()
        self.outbound: asyncio.Queue[OutboundMessage] = asyncio.Queue()
        # Consumer id -> (predicate, queue)
        self._routes: dict[str, tuple[Callable[[InboundMessage], bool], asyncio.Queue[InboundMessage]]] = {}
        self._router_task: asyncio.Task | None = None

    @property
    def inbound(self) -> asyncio.Queue[InboundMessage]:
        """Primary queue: consumed by RouterBus._route_loop() for routing."""
        return self._inbound

    async def consume_default_inbound(self) -> InboundMessage:
        """Consume from the default queue — for AgentLoop reading non-routed messages."""
        return await self._default_inbound.get()

    def register_route(
        self,
        consumer_id: str,
        predicate: Callable[[InboundMessage], bool],
    ) -> asyncio.Queue[InboundMessage]:
        """Register a consumer route.

        Args:
            consumer_id: unique identifier for this consumer
            predicate: returns True if this consumer handles the message

        Returns:
            A dedicated inbound queue for this consumer.
            Consumer should continuously read from this queue.
        """
        queue: asyncio.Queue[InboundMessage] = asyncio.Queue()
        self._routes[consumer_id] = (predicate, queue)
        logger.debug("Registered route for consumer: {}", consumer_id)
        return queue

    def unregister_route(self, consumer_id: str) -> None:
        """Remove a registered route."""
        self._routes.pop(consumer_id, None)
        logger.debug("Unregistered route: {}", consumer_id)

    async def start_router(self) -> None:
        """Start the router loop (non-blocking)."""
        if self._router_task is not None:
            return
        self._router_task = asyncio.create_task(self._route_loop())
        logger.info("RouterBus started")

    async def stop_router(self) -> None:
        """Stop the router loop."""
        if self._router_task is not None:
            self._router_task.cancel()
            self._router_task = None
            logger.info("RouterBus stopped")

    async def _route_loop(self) -> None:
        """Continuously consume from inbound and dispatch to registered routes."""
        while True:
            try:
                msg = await self.inbound.get()
            except asyncio.CancelledError:
                break
            dispatched = False
            for consumer_id, (predicate, queue) in self._routes.items():
                try:
                    if predicate(msg):
                        queue.put_nowait(msg)
                        dispatched = True
                except Exception as e:
                    logger.warning("Route predicate error for {}: {}", consumer_id, e)
            if not dispatched:
                # No route matched — put on default queue for AgentLoop
                self._default_inbound.put_nowait(msg)
