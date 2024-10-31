from dynamiq.connections.connections import Firecrawl
from dynamiq.nodes.tools.firecrawl import FirecrawlTool

if __name__ == "__main__":

    connection = Firecrawl()

    tool = FirecrawlTool(connection=connection, url="https://example.com")

    input_data = {}

    result_within_node = tool.run(input_data)

    tool = FirecrawlTool(connection=connection)

    input_data = {
        "url": "https://example.com",
    }
    result_with_input = tool.run(input_data)

    print("Result within node:")
    print(result_within_node)
    print("Result with input:")
    print(result_with_input)
