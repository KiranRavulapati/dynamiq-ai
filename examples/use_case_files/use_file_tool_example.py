import json

from dynamiq import Workflow
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.connections import E2B
from dynamiq.flows import Flow
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.tools.e2b_sandbox import E2BInterpreterTool
from dynamiq.runnables import RunnableConfig
from dynamiq.utils import JsonWorkflowEncoder
from examples.llm_setup import setup_llm

INPUT_PROMPT = "Summarize the text and try to evaluate it"
FILE_PATH = ".data/sample-essay-1.pdf"
AGENT_ROLE = (
    "A helpful and general-purpose AI assistant with strong language, Python, "
    "and Linux command-line skills. The goal is to provide concise answers to the user. "
    "Additionally, try to generate code to solve tasks, then run it accurately. "
    "Before answering, create a plan for solving the task. You can search for any API, "
    "and use any free, open-source API that doesn't require authorization."
    "You can install any packages for loading PDFS, such as PyMuPDF, PyPDF2, or pdfplumber."
    "and for the other file extensions as well, if you need to open them try to search and then install."
    "Also if you are working with binary files, try to understand the file format and then read the file."
)


def run_workflow(
    agent: ReActAgent,
    input_prompt: str,
    input_files: list,
) -> tuple[str, dict]:
    """
    Execute a workflow using the ReAct agent to process a predefined query.

    Returns:
        tuple[str, dict]: The generated content by the agent and the trace logs.

    Raises:
        Exception: Captures and prints any errors during workflow execution.
    """
    tracing = TracingCallbackHandler()
    wf = Workflow(flow=Flow(nodes=[agent]))

    try:
        result = wf.run(
            input_data={"input": input_prompt, "files": input_files},
            config=RunnableConfig(callbacks=[tracing]),
        )
        # Verify that traces can be serialized to JSON
        json.dumps(
            {"runs": [run.to_dict() for run in tracing.runs.values()]},
            cls=JsonWorkflowEncoder,
        )

        return result.output[agent.id]["output"]["content"], tracing.runs
    except Exception as e:
        print(f"An error occurred: {e}")
        return "", {}


csv_bytes = open(FILE_PATH, "rb").read()

python_tool = E2BInterpreterTool(connection=E2B())

llm = setup_llm()


agent = ReActAgent(
    name="Agent",
    id="Agent",
    llm=llm,
    role=AGENT_ROLE,
    tools=[python_tool],
)

output, traces = run_workflow(
    agent=agent,
    input_prompt=INPUT_PROMPT,
    input_files=[csv_bytes],
)
print("Agent Output:", output)
