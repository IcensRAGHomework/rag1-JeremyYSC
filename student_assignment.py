import requests

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from model_configurations import get_calendarific_api_key
from model_configurations import get_model_configuration
from pydantic import BaseModel, Field
from typing import List

gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)

api_key = get_calendarific_api_key()
base_url = "https://calendarific.com/api/v2/holidays"


class Anniversary(BaseModel):
    date: str = Field(description="Anniversary date")
    name: str = Field(description="Anniversary name")


class AnniversaryResponse(BaseModel):
    Result: List[Anniversary] = Field(description="List of Anniversary")


def generate_hw01(question):
    llm = get_llm()

    message = HumanMessage(
        content=[
            {"type": "text", "text": question},
        ]
    )

    llm = llm.with_structured_output(AnniversaryResponse)
    response = llm.invoke([message])
    return response.json()


def generate_hw02(question):
    llm = get_llm()
    tools = [fetch_holidays]
    llm = llm.bind_tools(tools)

    message = HumanMessage(
        content=[
            {"type": "text", "text": question},
        ]
    )

    chain = llm | fetch_holidays_from_ai_msg
    return chain.invoke([message])


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


@tool
def fetch_holidays(country: str, year: str, month: str):
    """Get the anniversary of a specific country in a certain year and month"""
    params = {
        "api_key": api_key,
        "country": country,
        "year": year,
        "month": month,
    }

    response = requests.get(base_url, params=params)
    anniversary_response = AnniversaryResponse(Result=[])

    if response.status_code == 200:
        holidays = response.json().get("response", {}).get("holidays", [])

        for holiday in holidays:
            anniversary_response.Result.append(Anniversary(date=holiday["date"]["iso"], name=holiday["name"]))

        return anniversary_response.model_dump_json()
    return anniversary_response.model_dump_json()


def fetch_holidays_from_ai_msg(msg):
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        for tool_call in msg.tool_calls:
            if tool_call["name"].lower() == "fetch_holidays":
                return fetch_holidays.invoke(tool_call["args"])

    return msg


def main():
    response = generate_hw01('2024年台灣10月紀念日有哪些?')
    print(response)
    print(type(response))

    response = generate_hw02('2024年台灣10月紀念日有哪些?')
    print(response)
    print(type(response))


if __name__ == '__main__':
    main()
