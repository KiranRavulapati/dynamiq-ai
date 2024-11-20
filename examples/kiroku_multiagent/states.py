import json
import re
from typing import Any

from dynamiq.connections import ZenRows
from dynamiq.nodes.tools.zenrows import ZenRowsTool
from dynamiq.prompts import Message, Prompt
from examples.kiroku_multiagent.prompts import INTERNET_SEARCH_PROMPT, TASK_TEMPLATE, TITLE_PROMPT
from examples.llm_setup import setup_llm

llm = setup_llm()
tool_scrapper = ZenRowsTool(connection=ZenRows())


def inference_model(messages):
    """
    Inference model
    """
    llm_result = llm.run(
        input_data={},
        prompt=Prompt(
            messages=messages,
        ),
    ).output["content"]

    return llm_result


def suggest_title_review_state(context: dict[str, Any]):
    """State that suggests user to review proposed titles"""

    messages = [Message(**message) for message in context.get("messages", [])]

    instruction = context.get("update_instruction")
    if instruction:
        message = Message(role="user", content=instruction)
    else:
        message = Message(role="user", content="Just return the final title without any additional information")

    messages = messages.append(message)
    llm_result = inference_model(messages)

    if not instruction:
        messages = []

    return {"result": llm_result, "messages": messages}


def suggest_title_state(context: dict[str, Any]):
    """State that suggests title for a paper."""

    messages = context.get("messages", [])

    if not messages:
        messages = [
            Message(
                role="system",
                content=TITLE_PROMPT.format(
                    area_of_paper=context["area_of_paper"], title=context["title"], hypothesis=context["hypothesis"]
                ),
            ),
            Message(
                role="user",
                content="Write the original title first. Then,"
                "generate 10 thought provoking titles that "
                "instigates reader's curiosity based on the given information",
            ),
        ]

    llm_result = inference_model(messages)

    return {"result": llm_result, "messages": messages}


def internet_search_state(context: dict[str, Any]):
    messsages = context.get("messages", [])
    messsages += [
        Message(
            role="system",
            content=INTERNET_SEARCH_PROMPT.format(number_of_queries=context["number_of_queries"])
            + " You must only output the response in a plain list of queries "
            "in the JSON format '{ \"queries\": list[str] }' and no other text. "
            "You MUST only cite references that are in the references "
            "section. ",
        ),
        Message(
            role="user",
            content=TASK_TEMPLATE.format(
                title=context.get("title", "(No title)"),
                type_of_document=context.get("type_of_document", "(No type_of_document)"),
                area_of_paper=context.get("area_of_paper", "(No area_of_paper)"),
                sections=context.get("sections", "(No sections)"),
                instruction=context.get("instruction", "(No instruction)"),
                hypothesis=context.get("hypothesis", "(No hypothesis)"),
                results=context.get("results", "(No results)"),
                references="\n".join(context.get("references", ["(No references)"])),
            ),
        ),
    ]

    generated_queries = json.loads(inference_model(messsages))["queries"]

    for ref in context["references"]:
        search_match = re.search(r"http.*(\s|$)", ref)
        if search_match:
            l, r = search_match.span()
            http_ref = ref[l:r]
            generated_queries.append(http_ref)

    content = context.get("content", [])
    for query in generated_queries:
        search_result = tool_scrapper.run(input_data={"url": query})

        for result in search_result:
            text = f"link: {result['url']}, " f"content: {result['content']}"

            content.append(text)

    return {
        "result": "Gathered information.",
        "content": content,
    }
