import os

from dynamiq.callbacks.streaming import StreamingIteratorCallbackHandler
from dynamiq.connections import E2B, HttpApiKey, ScaleSerp
from dynamiq.memory import Memory
from dynamiq.memory.backend.in_memory import InMemory
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.tools.e2b_sandbox import E2BInterpreterTool
from dynamiq.nodes.tools.mailgun import MailGunTool
from dynamiq.nodes.tools.scale_serp import ScaleSerpTool
from dynamiq.runnables import RunnableConfig
from dynamiq.types.streaming import StreamingConfig, StreamingMode
from examples.llm_setup import setup_llm


def setup_agent(agent_role: str, streaming_enabled: bool, streaming_mode: str, streaming_tokens: bool) -> ReActAgent:
    """
    Initializes an AI agent with a specified role and streaming configuration.
    """
    llm = setup_llm()
    memory = Memory(backend=InMemory())

    # Convert string mode to StreamingMode enum
    mode = StreamingMode.FINAL if streaming_mode == "Answer" else StreamingMode.ALL

    # Create streaming config
    streaming_config = StreamingConfig(enabled=streaming_enabled, mode=mode, by_tokens=streaming_tokens)

    # Create tool instances
    tool_email = MailGunTool(
        connection=HttpApiKey(api_key=os.getenv("MAILGUN_API_KEY"), url="https://api.mailgun.net/v3"),
        domain_name=os.getenv("MAILGUN_DOMAIN"),
    )
    tool_search = ScaleSerpTool(connection=ScaleSerp())
    tool_code = E2BInterpreterTool(connection=E2B())

    # Create and return agent
    agent = ReActAgent(
        name="Agent",
        llm=llm,
        role=agent_role,
        id="agent",
        memory=memory,
        tools=[tool_code, tool_search, tool_email],
        streaming=streaming_config,
    )
    return agent


def generate_agent_response(agent: ReActAgent, user_input: str):
    """
    Processes the user input using the agent. Supports both streaming and non-streaming responses.
    """
    try:
        if agent.streaming.enabled:
            streaming_handler = StreamingIteratorCallbackHandler()
            config = RunnableConfig(callbacks=[streaming_handler])

            # Start agent execution
            agent.run(input_data={"input": user_input}, config=config)

            for chunk in streaming_handler:
                if chunk and isinstance(chunk.data, dict):
                    content = chunk.data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    if content:
                        yield content
        else:
            result = agent.run({"input": user_input})
            if result and hasattr(result, "output"):
                yield result.output.get("content", "No response generated")
            else:
                yield "Error: No valid response received"

    except Exception as e:
        print(f"Error generating response: {str(e)}")
        yield f"Error: {str(e)}"
