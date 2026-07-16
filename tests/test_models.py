import pytest
from pydantic import ValidationError

from support_qa.models import Message, Role, Ticket


def test_ticket_requires_messages():
    with pytest.raises(ValidationError):
        Ticket(id="x", messages=[])


def test_transcript_render():
    ticket = Ticket(
        id="x",
        messages=[
            Message(role=Role.CUSTOMER, text="hello"),
            Message(role=Role.AGENT, text="hi there"),
        ],
    )
    assert ticket.transcript() == "CUSTOMER: hello\nAGENT: hi there"
