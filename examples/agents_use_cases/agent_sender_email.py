from dynamiq.connections import E2B, HttpApiKey
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.tools.e2b_sandbox import E2BInterpreterTool
from dynamiq.nodes.tools.mailgun import MailGunTool
from examples.llm_setup import setup_llm

# Create connection
connection = HttpApiKey(api_key="", url="https://api.mailgun.net/v3")

# Create tool instance
mailgun_tool = MailGunTool(connection=connection, domain_name="")
connection_e2b = E2B()

tool_code = E2BInterpreterTool(connection=connection_e2b)


llm = setup_llm()


agent = ReActAgent(
    name="AI Agent",
    llm=llm,
    tools=[mailgun_tool, tool_code],
    role="is to help user with various tasks, goal is to provide best of possible answers to user queries",  # noqa: E501
)

result = agent.run(
    input_data={
        "input": (
            "Generate the 10 random numbers and send report does their"
            "sum is magic to `olbychos@gmail.com` and oleksii@getdynamiq.ai"
        )
    },
    config=None,
)
print(result.output)
