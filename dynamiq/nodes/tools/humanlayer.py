from functools import wraps
from typing import Callable, TypeVar

from humanlayer import ApprovalMethod, ContactChannel, HumanLayer, UserDeniedError

hl = HumanLayer()
T = TypeVar("T")
R = TypeVar("R")


def require_approval(
    contact_channel: ContactChannel | None = None,
):

    def decorator(fn):  # type: ignore
        if hl.approval_method is ApprovalMethod.CLI:
            return _approve_cli(fn)
        return hl._approve_with_backend(fn, contact_channel)

    return decorator


def _approve_cli(fn: Callable[[T], R]) -> Callable[[T], R | str]:
    """
    NOTE we convert a callable[[T], R] to a Callable [[T], R | str]

    this is safe to do for most LLM use cases. It will blow up
    a normal function.

    If we can guarantee the function calling framework
    is properly handling exceptions, then we can
    just raise and let the framework handle the stringification
    of what went wrong.

    Because some frameworks dont handle exceptions well, were stuck with the hack for now
    """

    @wraps(fn)
    def wrapper(*args, **kwargs) -> R | str:  # type: ignore
        print(
            f"""Agent {hl.run_id} wants to call

{fn.__name__}({str(kwargs)})

{"" if not args else " with args: " + str(args)}"""
        )
        feedback = input("Hit ENTER to proceed, or provide feedback to the agent to deny: \n\n")
        if feedback not in {
            None,
            "",
        }:
            return str(UserDeniedError(f"User denied {fn.__name__} with feedback: {feedback}"))
        try:
            print(args)
            return fn(*args, **kwargs)
        except Exception as e:
            return f"Error running {fn.__name__}: {e}"

    wrapper.__doc__ = fn.__doc__
    return wrapper
