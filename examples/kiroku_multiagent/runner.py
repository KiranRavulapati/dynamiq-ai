from typing import Any

import yaml

from dynamiq.nodes.agents.orchestrators.graph import END, GraphOrchestrator
from dynamiq.nodes.agents.orchestrators.graph_manager import GraphAgentManager
from dynamiq.nodes.tools.human_feedback import HumanFeedbackTool
from dynamiq.runnables import RunnableResult
from examples.llm_setup import setup_llm

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from examples.kiroku_multiagent.states import internet_search_state, suggest_title_review_state, suggest_title_state

nodes = {"internet_search": internet_search_state}


def is_title_review_complete(context: dict[str, Any]):
    return True if context.get("messages") else False


def create_orchestrator(configuration) -> GraphOrchestrator:
    """
    Creates orchestrator

    Returns:
        GraphOrchestrator: The configured orchestrator.
    """
    llm = setup_llm(model_provider="gpt", model_name="gpt-4o-mini", temperature=0)

    suggest_title = configuration.get("suggest_title", False)

    orchestrator = GraphOrchestrator(
        name="Graph orchestrator",
        manager=GraphAgentManager(llm=llm),
        context=configuration,
        initial_state="suggest_title" if suggest_title else "internet_search",
    )

    if suggest_title:
        orchestrator.add_node("suggest_title", [suggest_title_state])
        orchestrator.add_node("suggest_title_review", [suggest_title_review_state])

        orchestrator.add_edge("suggest_title", "suggest_title_review")
        orchestrator.add_conditional_edge(
            "suggest_title_review", ["suggest_title", "internet_search"], is_title_review_complete
        )

    orchestrator.add_edge("internet_search", END)

    return orchestrator


def parse_configuration(filename):
    stream = open(filename)
    state = yaml.safe_load(stream, Loader=Loader)
    return state


def run_orchestrator(configuration_file) -> RunnableResult:
    """Runs orchestrator"""

    configuration_context = parse_configuration(configuration_file)

    orchestrator = create_orchestrator(configuration_context)

    result = orchestrator.run(
        input_data={
            "input": f"Write {configuration_context["type_of_document"]} about {configuration_context["title"]}"
        },
        config=None,
    )

    feedback_tool = HumanFeedbackTool()

    while True:
        user_feedback = feedback_tool.run(input_data={"input": result}).output["content"]
        if user_feedback == "EXIT":
            break

        orchestrator.context["update_instruction"] = user_feedback
        result = orchestrator.run(input_data={"input": f"Update {configuration_context["type_of_document"]}"})

    return result


if __name__ == "__main__":
    configuration_file = "./examples/kiroku_multiagent/test.yaml"
    result = run_orchestrator(configuration_file)
    print("Result:")
    print(result)
