import base64
import json
import requests

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from langchain_community.document_loaders import UnstructuredImageLoader
from langchain.prompts import PromptTemplate
from mimetypes import guess_type
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


class CheckResponse(BaseModel):
    add: bool = Field(description="If need to add a new anniversary")
    reason: str = Field(description="Why or why not add an anniversary")


class CheckResponseList(BaseModel):
    Result: CheckResponse = Field(description="Result of CheckResponse")


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
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    memory = ChatMessageHistory()
    memory.add_user_message(question2)
    memory.add_ai_message(generate_hw02(question2))

    def get_session_history(session_id):
        return memory

    llm = get_llm()
    llm = llm.with_structured_output(CheckResponseList)
    chain = prompt | llm

    chat_history = RunnableWithMessageHistory(
        chain,
        get_session_history=get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    response = chat_history.invoke({"input": question3}, config={"configurable": {"session_id": "0"}})
    return response.json()


def generate_hw04(question):
    path = 'baseball.png'
    page3_encoded = image_to_data_url(path)
    llm = get_llm()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {
                "type": "text",
                "text": f'請回答 {question},以 JSON 格式輸出，格式如下: {{\"Result\": {{\"score\": \"答案\"}}}}'
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": page3_encoded
                }
            }
        ]}
    ]

    chain = llm
    response = chain.invoke(messages)
    parser = JsonOutputParser()
    parsed_result = parser.parse(response.content)
    parsed_result["Result"]["score"] = int(parsed_result["Result"]["score"])
    return json.dumps(parsed_result, ensure_ascii=False)

    # image_path = 'baseball.png'
    # data_url = image_to_data_url(image_path)
    #
    # prompt_template = ChatPromptTemplate.from_messages([
    #     ("system", "You are an expert in analyzing images and extracting data."),
    #     ("human", "{question}\nImage data: {image_data}")
    # ])
    #
    # prompt = prompt_template.format_prompt(
    #     question=question,
    #     image_data=data_url
    # )
    #
    # llm = get_llm()
    # return llm.invoke(prompt)


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


def image_to_data_url(image_path):
    with open(image_path, "rb") as img_file:
        img_data = img_file.read()
        return "data:image/jpeg;base64," + base64.b64encode(img_data).decode('utf-8')


def main():
    question1 = '2024年台灣10月紀念日有哪些?'
    # response = generate_hw01(question1)
    # print(response)
    # print(type(response))

    question2 = '2024年台灣10月紀念日有哪些?'
    # response2 = generate_hw02(question2)
    # print(response2)
    # print(type(response2))

    question3 = '根據先前的節日清單，這個節日{"date": "10-31", "name": "蔣公誕辰紀念日"}是否有在該月份清單？'
    # response3 = generate_hw03(question2, question3)
    # print(response3)
    # print(type(response3))

    question4 = '請問中華台北的積分是多少'
    response4 = generate_hw04(question4)
    # print(response4)
    # print(type(response4))


if __name__ == '__main__':
    main()
