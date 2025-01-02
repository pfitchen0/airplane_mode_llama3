import mesop as me

from airplane_mode_llama3 import GroqModels, GroqWrapper, Llama3Models, Llama3, Message, chat


_GROQ_MODEL = GroqModels.LLAMA3_3_70B_VERSATILE
_LOCAL_MODEL = Llama3Models.LLAMA3_2_1B
_TEMPERATURE = 0.8
_MAX_TOKENS = None

groq_model = GroqWrapper(_GROQ_MODEL)
local_model = Llama3(_LOCAL_MODEL)


def on_load(e: me.LoadEvent):
    me.set_theme_mode("system")


@me.page(on_load=on_load)
def page():
    chat(chat_completion)


def chat_completion(messages: list[Message], use_groq: bool) -> Message:
    if use_groq:
        return groq_model.chat_completion(
            messages=messages, temperature=_TEMPERATURE, max_tokens=_MAX_TOKENS
        )
    else:
        return local_model.chat_completion(
            messages=messages, temperature=_TEMPERATURE, max_tokens=_MAX_TOKENS
        )
