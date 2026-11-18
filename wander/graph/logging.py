import inspect
import logging
from functools import wraps
from time import perf_counter
from typing import Any, Callable


logger = logging.getLogger("wander.graph")


def log_node(name: str, node: Callable[..., Any]) -> Callable[..., Any]:
    if inspect.iscoroutinefunction(node):
        @wraps(node)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            started_at = perf_counter()
            logger.info("Node started: %s", name)
            try:
                result = await node(*args, **kwargs)
            except Exception:
                logger.exception("Node failed: %s", name)
                raise
            logger.info(
                "Node completed: %s (%.3fs)",
                name,
                perf_counter() - started_at,
            )
            return result

        return async_wrapper

    @wraps(node)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        started_at = perf_counter()
        logger.info("Node started: %s", name)
        try:
            result = node(*args, **kwargs)
        except Exception:
            logger.exception("Node failed: %s", name)
            raise
        logger.info(
            "Node completed: %s (%.3fs)",
            name,
            perf_counter() - started_at,
        )
        return result

    return wrapper
