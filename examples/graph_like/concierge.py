import json

from dotenv import load_dotenv

from dynamiq import Workflow
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.flows import Flow
from dynamiq.nodes.agents.orchestrators import AdaptiveOrchestrator
from dynamiq.nodes.agents.orchestrators.adaptive_manager import AdaptiveAgentManager
from dynamiq.nodes.agents.react import ReActAgent

from dynamiq.runnables import RunnableConfig
from dynamiq.utils import JsonWorkflowEncoder
from dynamiq.utils.logger import logger
from examples.llm_setup import setup_llm
from dynamiq.nodes.tools.function_tool import function_tool
from dynamiq.nodes.tools.human_feedback import HumanFeedbackTool
from dynamiq.nodes.agents.base import Agent


class CustomToolAgent(Agent):
    def execute(
        self, input_data, config: RunnableConfig | None = None, **kwargs
    ):
        """
        Executes the agent with the given input data.
        """
        logger.debug(f"Agent {self.name} - {self.id}: started with input {input_data}")
        self.reset_run_state()
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        tool_result = self.tools[0].run(
            input_data=input_data,
            config=config,
            **kwargs,
        )

        execution_result = {
            "content": tool_result,
            "intermediate_steps": self._intermediate_steps,
        }

        if self.streaming.enabled:
            self.run_on_node_execute_stream(
                config.callbacks, execution_result, **kwargs
            )

        logger.debug(f"Agent {self.name} - {self.id}: finished with result {tool_result}")
        return execution_result


# Load environment variables
load_dotenv()


def create_workflow() -> Workflow:
    """
    Create the workflow with all necessary agents and tools.

    Returns:
        Workflow: The configured workflow.
    """

    llm = setup_llm()

    # Stock Lookup Agent
    @function_tool
    def lookup_stock_price(stock_symbol: str) -> str:
        """Useful for looking up a stock price."""
        logger.info(f"Looking up stock price for {stock_symbol}")
        return f"Symbol {stock_symbol} is currently trading at $100.00"

    @function_tool
    def search_for_stock_symbol(str: str) -> str:
        """Useful for searching for a stock symbol given a free-form company name."""
        logger.info("Searching for stock symbol")
        return str.upper()

    @function_tool
    def done() -> None:
        """When you have returned a stock price, call this tool."""
        return "Stock lookup is complete"

    stock_lookup_agent = ReActAgent(
        name="stock_lookup_agent",
        role="""When you have authenticated, call the tool "done" to signal that you are done.
        If the user asks to do anything other than authenticate,
         call the tool "done" to signal some other agent should help.""",
        goal="Provide actions according to role",  # noqa: E501
        llm=llm,
        tools=[lookup_stock_price(), search_for_stock_symbol(), done()],
    )

    # Auth Agent
    @function_tool
    def store_username(username: str) -> None:
        """Adds the username to the user state."""
        print("Recording username")
        return "Username was recorded."

    @function_tool
    def login(password: str) -> None:
        """Given a password, logs in and stores a session token in the user state."""
        logger.info(f"Logging in with password {password}")
        return f"Sucessfully logged in with password {password}"

    @function_tool
    def is_authenticated() -> bool:
        """Checks if the user has a session token."""
        print("Checking if authenticated")
        return "User is authenticated"

    @function_tool
    def done() -> None:
        """When you complete your task, call this tool."""
        logger.info("Authentication is complete")
        return "Authentication is complete"

    auth_agent = ReActAgent(
        name="auth_agent",
        role="""You are a helpful assistant that is authenticating a user.
        Your task is to get a valid session token stored in the user state.
        To do this, the user must supply you with a username and a valid password. You can ask them to supply these.
        If the user supplies a username and password, call the tool "login" to log them in.""",
        goal="Provide actions according to role",  # noqa: E501
        llm=llm,
        tools=[store_username(), login(), is_authenticated(), done()],
    )

    # Account Balance Agent
    @function_tool
    def get_account_id(account_name: str) -> str:
        """Useful for looking up an account ID."""
        print(f"Looking up account ID for {account_name}")
        account_id = "1234567890"
        return f"Account id is {account_id}"

    @function_tool
    def get_account_balance(account_id: str) -> str:
        """Useful for looking up an account balance."""
        logger.info(f"Looking up account balance for {account_id}")
        return f"Account {account_id} has a balance of $1000"

    @function_tool
    def is_authenticated() -> bool:
        """Checks if the user has a session token."""
        logger.info("Account balance agent is checking if authenticated")
        return "User is authentificated"

    @function_tool
    def done() -> None:
        """When you complete your task, call this tool."""
        logger.info("Account balance lookup is complete")
        return "Account balance lookup is complete"

    account_balance_agent = ReActAgent(
        name="account_balance_agent",
        role="""
        You are a helpful assistant that is looking up account balances.
        The user may not know the account ID of the account they're interested in,
        so you can help them look it up by the name of the account.
        The user can only do this if they are authenticated, which you can check with the is_authenticated tool.
        If they aren't authenticated, tell them to authenticate
        If they're trying to transfer money, they have to check their account balance first, which you can help with.
        Once you have supplied an account balance, you must call the tool "done" to signal that you are done.
        If the user asks to do anything other than look up an account balance,
        call the tool "done" to signal some other agent should help.
         """,
        goal="Provide actions according to role",  # noqa: E501
        llm=llm,
        tools=[get_account_id(), get_account_balance(), is_authenticated(), done()],
    )

    # Transfer Money Agent
    @function_tool
    def transfer_money(from_account_id: str, to_account_id: str, amount: int) -> None:
        """Useful for transferring money between accounts."""
        logger.info(f"Transferring {amount} from {from_account_id} account {to_account_id}")
        return f"Transferred {amount} to account {to_account_id}"

    @function_tool
    def balance_sufficient(account_id: str, amount: int) -> bool:
        """Useful for checking if an account has enough money to transfer."""
        logger.info("Checking if balance is sufficient")
        return "There is enough money."

    @function_tool
    def has_balance() -> bool:
        """Useful for checking if an account has a balance."""
        logger.info("Checking if account has a balance")
        return "Account has enough balance"

    @function_tool
    def is_authenticated() -> bool:
        """Checks if the user has a session token."""
        logger.info("Transfer money agent is checking if authenticated")
        return "User has a session token."

    @function_tool
    def done() -> None:
        """When you complete your task, call this tool."""
        logger.info("Money transfer is complete")
        return "Money transfer is complete"

    transfer_money_agent = ReActAgent(
        name="transfer_money_agent",
        role="""
        You are a helpful assistant that transfers money between accounts.
        The user can only do this if they are authenticated, which you can check with the is_authenticated tool.
        If they aren't authenticated, tell them to authenticate first.
        The user must also have looked up their account balance already, which you can check with the has_balance tool.
        If they haven't already, tell them to look up their account balance first.
        Once you have transferred the money, you can call the tool "done" to signal that you are done.
        If the user asks to do anything other than transfer money, call the tool "done" to signal some othe
         """,
        goal="Provide actions according to role",  # noqa: E501
        llm=llm,
        tools=[transfer_money(), balance_sufficient(), has_balance(), is_authenticated(), done()],
    )

    # Transfer Money Agent
    human_feedback_tool = HumanFeedbackTool()
    concierge_agent = CustomToolAgent(
        name="concierge_agent",
        role="""
            You are a helpful assistant that is helping a user navigate a financial system.
            Your job is to ask the user questions to and return this in Answer. (make only one loop)

            Introduce user with the available things they can do.
            That includes
            * looking up a stock price
            * authenticating the user
            * checking an account balance (requires authentication first)
            * transferring money between accounts (requires authentication and checking an account balance first)

            Make up some answer.

         """,
        goal="Provide actions according to role",  # noqa: E501
        tools=[human_feedback_tool],
        llm=llm)

    llm = llm
    agent_manager = AdaptiveAgentManager(llm=llm)

    # Create agent orchestrator
    adaptive_orchestrator = AdaptiveOrchestrator(
        manager=agent_manager,
        agents=[stock_lookup_agent, auth_agent, account_balance_agent, transfer_money_agent, concierge_agent],
        final_summarizer=True,
    )

    return Workflow(
        flow=Flow(nodes=[adaptive_orchestrator]),
    )


def run_planner() -> tuple[str, dict]:
    # Create workflow
    workflow = create_workflow()

    user_prompt = """
    Hello
    """  # noqa: E501

    # Run workflow
    tracing = TracingCallbackHandler()
    try:
        result = workflow.run(
            input_data={"input": user_prompt},
            config=RunnableConfig(callbacks=[tracing]),
        )

        # Dump traces
        _ = json.dumps(
            {"runs": [run.to_dict() for run in tracing.runs.values()]},
            cls=JsonWorkflowEncoder,
        )

        logger.info("Workflow completed successfully")

        # Print and save result
        output = result.output[workflow.flow.nodes[0].id]['output']['content']
        print(output)

        return output, tracing.runs

    except Exception as e:
        logger.error(f"An error occurred during workflow execution: {str(e)}")
        return "", {}


if __name__ == "__main__":
    run_planner()
