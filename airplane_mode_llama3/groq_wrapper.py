from enum import Enum
from dotenv import load_dotenv
from groq import Groq
import os

from .utils import Message

load_dotenv()


class GroqModels(Enum):
    LLAMA3_1_8B_INSTANT = "llama-3.1-8b-instant"
    LLAMA3_3_70B_VERSATILE = "llama-3.3-70b-versatile"
    GROQ_8B_TOOL_USE = "llama3-groq-8b-8192-tool-use-preview"
    GROQ_70B_TOOL_USE = "llama3-groq-70b-8192-tool-use-preview"
    GEMMA2_9B = "gemma2-9b-it"


class GroqWrapper:
    def __init__(self, model: GroqModels) -> None:
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model = model.value

    def chat_completion(
        self,
        messages: list[Message],
        temperature: float = 1.0,
        max_tokens: int | None = 1024,
    ) -> Message:
        messages = [
            {"role": message.role, "content": message.content} for message in messages
        ]
        response = (
            self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            .choices[0]
            .message
        )
        return Message(role=response.role, content=response.content)


if __name__ == "__main__":
    groq_wrapper = GroqWrapper(GroqModels.LLAMA3_3_70B_VERSATILE)

    print("Enter 'q' or 'exit' to quit/exit.")
    messages = []
    while True:
        prompt = input(">>> ")
        if prompt == "q" or prompt == "exit":
            break
        messages.append(Message(role="user", content=prompt))
        response = groq_wrapper.chat_completion(messages=messages)
        messages.append(response)
        print(response.content)
