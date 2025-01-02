from enum import Enum

from model import Model
from tokenizer import Tokenizer
from utils import Message


# These are models I've tested, but feel free to add other Llama3-based huggingface models.
class Llama3Models(Enum):
    LLAMA3_2_1B = "meta-llama/Llama-3.2-1B-Instruct"
    LLAMA3_2_1B_Q4 = "mlx-community/Llama-3.2-1B-Instruct-4bit"
    LLAMA3_2_3B = "meta-llama/Llama-3.2-3B-Instruct"
    LLAMA3_2_3B_Q4 = "mlx-community/Llama-3.2-3B-Instruct-4bit"
    LLAMA3_1_8B = "meta-llama/Llama-3.1-8B-Instruct"
    LLAMA3_1_8B_Q4 = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
    GROQ_8B_TOOL_USE = "Groq/Llama-3-Groq-8B-Tool-Use"
    LLAMA3_1_70B_Q4 = "mlx-community/Meta-Llama-3.1-70B-Instruct-4bit"
    LLAMA3_3_70B_Q4 = "mlx-community/Meta-Llama-3.3-70B-Instruct-4bit"


class Llama3:
    def __init__(self, llama3_model: Llama3Models) -> None:
        self.model = Model.from_huggingface(hf_repo_id=llama3_model.value)
        self.tokenizer = Tokenizer()

    def chat_completion(
        self,
        messages: list[Message],
        temperature: float = 0.8,
        max_tokens: int | None = None,
    ) -> Message:
        tokens = self.tokenizer.apply_chat_template(
            messages=messages, tokenize=True, add_generation_prompt=True
        )
        response_tokens = []
        for token in self.model.generate(tokens=tokens, temperature=temperature):
            response_tokens.append(token)
            if token == self.tokenizer.eos_token_id:
                break
            elif max_tokens and len(response_tokens) >= max_tokens:
                response_tokens.append(self.tokenizer.eos_token_id)
                break
        content = self.tokenizer.decode(response_tokens[:-1])  # Don't include eos_token
        return Message(role="assistant", content=content)


if __name__ == "__main__":
    llama3 = Llama3(Llama3Models.LLAMA3_2_1B)

    print("Enter 'q' or 'exit' to quit/exit.")
    messages = []
    while True:
        prompt = input(">>> ")
        if prompt == "q" or prompt == "exit":
            break
        messages.append(Message(role="user", content=prompt))
        response = llama3.chat_completion(messages=messages)
        messages.append(response)
        print(response.content)
