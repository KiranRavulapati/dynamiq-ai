import pytest

from dynamiq import Workflow
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.connections import connections
from dynamiq.flows import Flow
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.operators import Map
from dynamiq.prompts import Message, Prompt
from dynamiq.runnables import RunnableConfig, RunnableResult, RunnableStatus


def get_map_workflow(
    model: str,
    connection: connections.OpenAI,
):
    openai_node = OpenAI(
        name="OpenAI",
        model=model,
        connection=connection,
        prompt=Prompt(
            messages=[
                Message(
                    role="user",
                    content="What is LLM?",
                ),
            ],
        ),
        temperature=0.1,
    )
    wf_map = Workflow(
        flow=Flow(
            nodes=[Map(node=openai_node)],
        ),
    )

    return wf_map


@pytest.mark.parametrize(
    ("inputs", "outputs"),
    [
        (
            [{"test": "OK"}, {"test": "BAD"}],
            [{"content": "mocked_response", "tool_calls": None}, {"content": "mocked_response", "tool_calls": None}],
        ),
    ],
)
def test_workflow_with_map_node(inputs, outputs):
    model = "gpt-3.5-turbo"
    connection = connections.OpenAI(
        api_key="api_key",
    )
    wf_map_node = get_map_workflow(model, connection)
    input_data = {"inputs": inputs}
    tracing = TracingCallbackHandler()
    response = wf_map_node.run(input_data=input_data, config=RunnableConfig(callbacks=[tracing]))

    expected_result = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input_data,
        output=outputs,
    ).to_dict()

    expected_output = {wf_map_node.flow.nodes[0].id: expected_result}
    assert response == RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input_data,
        output=expected_output,
    )
