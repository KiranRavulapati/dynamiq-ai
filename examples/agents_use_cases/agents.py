from dynamiq.connections import E2B, ScaleSerp
from dynamiq.nodes.agents.orchestrators.adaptive import AdaptiveOrchestrator
from dynamiq.nodes.agents.orchestrators.adaptive_manager import AdaptiveAgentManager
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.agents.simple import SimpleAgent
from dynamiq.nodes.tools.e2b_sandbox import E2BInterpreterTool
from dynamiq.nodes.tools.scale_serp import ScaleSerpTool
from dynamiq.nodes.types import InferenceMode
from dynamiq.utils.logger import logger
from examples.llm_setup import setup_llm

INPUT = """Craft the Kmeans algorithm,
generate data and craft report with code, results and conclusion"""

AGENT_ROLE_SEARCHER = """
You are an expert AI assistant focused on refining user queries,
conducting thorough searches, and delivering well-sourced answers.
Your goal is to help users by clarifying their questions,
performing detailed searches using available tools,
and presenting accurate responses with proper citations and direct links, using markdown.
Prioritize accuracy, clarity, and credibility in all answers.
If information is conflicting or unclear, note this and suggest further research options.
"""

AGENT_ROLE_WRITER = """
You are an AI assistant for content refinement,
focused on improving readability while retaining the original message,
links, and citations. Your task is to enhance the clarity
and flow of the text without altering its tone or core details.
Guidelines:
1. **Clarity and Readability**: Adjust sentence structure to improve flow and comprehension.
2. **Grammar and Style**: Correct any errors or awkward phrasing and maintain a consistent tone.
3. **Engaging Vocabulary**: Replace weak words with precise alternatives, keeping the language accessible.
4. **Preserve Links and Citations**: Do not alter hyperlinks or references.
5. **Maintain Journalistic Tone**: Retain all facts, dates,
names, and statistics accurately, without adding subjective or casual wording.
"""

AGENT_CODER = """
A helpful AI assistant skilled in language, Python programming, and Linux commands.
The goal is to provide clear, brief answers to the user.
For tasks that require code, first outline a plan,
then write well-structured Python code, check for errors,
and run it to confirm it works. Use any free, open-source API that doesn’t need authorization,
and install necessary packages for handling specific file types, like PDFs or binary files.
When working with binary files, understand the file format before reading them.
"""

if __name__ == "__main__":
    connection_e2b = E2B()
    connection_serp = ScaleSerp()

    tool_search = ScaleSerpTool(connection=connection_serp)
    tool_code = E2BInterpreterTool(connection=connection_e2b)

    llm = setup_llm(model_provider="gpt", model_name="gpt-4o-mini", temperature=0)

    agent_searcher = ReActAgent(
        name="Searcher",
        id="Agent",
        llm=llm,
        tools=[tool_search],
        role=AGENT_ROLE_SEARCHER,
        inference_mode=InferenceMode.XML,
    )

    agent_writer = SimpleAgent(
        name="Writer",
        id="Agent Writer",
        llm=llm,
        role=AGENT_ROLE_WRITER,
        inference_mode=InferenceMode.XML,
    )

    agent_manager = AdaptiveAgentManager(
        llm=llm,
    )

    orchestrator = AdaptiveOrchestrator(
        name="Adaptive Orchestrator",
        agents=[agent_searcher, agent_writer],
        manager=agent_manager,
    )

    agent_coder = ReActAgent(
        name="Coder",
        id="Agent Coder",
        llm=llm,
        tools=[tool_code],
        role=AGENT_CODER,
        inference_mode=InferenceMode.XML,
        max_loops=25,
    )

    result = agent_coder.run(input_data={"input": INPUT})
    output_content = result.output.get("content")
    logger.info("RESULT")
    logger.info(output_content)
