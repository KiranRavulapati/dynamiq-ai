from dynamiq.nodes.tools.function_tool import FunctionTool, function_tool
from humanlayer import HumanLayer
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.runnables import RunnableConfig
from dynamiq.utils import JsonWorkflowEncoder
import json
from examples.llm_setup import setup_llm 
from dynamiq import Workflow
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.flows import Flow
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.connections import E2B as E2BConnection
from dynamiq.nodes.tools.e2b_sandbox import E2BInterpreterTool
from dynamiq.runnables import RunnableConfig
hl = HumanLayer()

AGENT_ROLE = "Professional mathematician"
AGENT_GOAL = "Answer on questions"
INPUT_QUESTION = "What is (5 + 10) * 3."
INPUT_QUESTION1 = "print('test') to console"

def run_simple_workflow(test_type = 1) -> tuple[str, dict]:
    """
    Execute a workflow using the OpenAI agent to process a predefined question. Uses HumanLayer.

    Returns:
        tuple[str, dict]: The generated content by the agent and the trace logs.

    Raises:
        Exception: Captures and prints any errors during workflow execution.
    """
    llm = setup_llm()

    e2b_tool = E2BInterpreterTool(
        connection=E2BConnection()
    )
    @function_tool
    @hl.require_approval() # Add HumanLayer support
    def python_tool(code: str):
        f"""{e2b_tool.description}"""
        return e2b_tool.execute({'python': code})
    
    tool_python = python_tool() 

    class AddNumbersTool(FunctionTool[int]):
        name: str = "Add Numbers Tool"
        description: str = "A tool that adds two numbers together."

        @hl.require_approval() # Add HumanLayer support
        def run_tool(self, a: int, b: int) -> int:
            """Add two numbers together."""
            return a + b

    @function_tool
    @hl.require_approval() # Add HumanLayer support
    def multiply_numbers(a: int, b: int) -> int:
        """Multiply two numbers together."""
        return a * b

    multiply_tool = multiply_numbers()
    add_tool = AddNumbersTool()

    if test_type == 0:
        tools=[add_tool, multiply_tool]
        input_question = INPUT_QUESTION
    elif test_type == 1:
        tools=[tool_python]
        input_question = INPUT_QUESTION1

    agent = ReActAgent(
        name=" Agent",
        llm=llm,
        role=AGENT_ROLE,
        goal=AGENT_GOAL,
        id="agent",
        tools=tools
    )
    tracing = TracingCallbackHandler()
    wf = Workflow(flow=Flow(nodes=[agent]))

    try:
        result = wf.run(
            input_data={"input": input_question},
            config=RunnableConfig(callbacks=[tracing]),
        )
        json.dumps(
            {"runs": [run.to_dict() for run in tracing.runs.values()]},
            cls=JsonWorkflowEncoder,
        )

        return result.output[agent.id]["output"]["content"], tracing.runs
    except Exception as e:
        print(f"An error occurred: {e}")
        return "", {}


if __name__ == "__main__":
    result, _ = run_simple_workflow()
    print(result)