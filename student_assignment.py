import json
import traceback

from model_configurations import get_model_configuration

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)


class Anniversary(BaseModel):
    date: str = Field(description="Anniversary date")
    name: str = Field(description="Anniversary name")


class AnniversaryResponse(BaseModel):
    Result: List[Anniversary] = Field(description="List of Anniversary")


def generate_hw01(question):
    llm = get_llm()
    # pydantic_parser = PydanticOutputParser(pydantic_object=AnniversaryResponse)
    # format_instructions = pydantic_parser.get_format_instructions()
    #
    # sys_message = SystemMessage(
    #     content=[
    #         {"type": "text", "text": format_instructions},
    #     ]
    # )

    message = HumanMessage(
        content=[
            {"type": "text", "text": question},
        ]
    )

    llm = llm.with_structured_output(AnniversaryResponse)
    response = llm.invoke([message])
    return response.json()


def generate_hw02(question):
    pass


def generate_hw03(question2, question3):
    pass


def generate_hw04(question):
    pass


def demo(question):
    llm = AzureChatOpenAI(
        model=gpt_config['model_name'],
        deployment_name=gpt_config['deployment_name'],
        openai_api_key=gpt_config['api_key'],
        openai_api_version=gpt_config['api_version'],
        azure_endpoint=gpt_config['api_base'],
        temperature=gpt_config['temperature']
    )
    message = HumanMessage(
        content=[
            {"type": "text", "text": question},
        ]
    )
    response = llm.invoke([message])

    return response


def get_llm():
    return AzureChatOpenAI(
        model=gpt_config['model_name'],
        deployment_name=gpt_config['deployment_name'],
        openai_api_key=gpt_config['api_key'],
        openai_api_version=gpt_config['api_version'],
        azure_endpoint=gpt_config['api_base'],
        temperature=gpt_config['temperature']
    )


def main():
    response = generate_hw01('2024年台灣10月紀念日有哪些?')
    print(response)


if __name__ == '__main__':
    main()
