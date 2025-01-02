from typing import Callable
import mesop as me

from .utils import Message, Role

_USER_BUBBLE_COLOR = me.theme_var("secondary-container")


@me.stateclass
class State:
    user_input: str
    messages: list[Message]
    on_airplane_mode: bool


def chat(chat_completion: Callable[[list[Message], Message], str]):
    state = me.state(State)

    # Event Handlers.

    def on_blur(e: me.InputBlurEvent):
        state = me.state(State)
        state.user_input = e.value

    def on_click_submit(e: me.ClickEvent):
        state = me.state(State)
        if not state.user_input:
            return
        user_input = state.user_input
        state.user_input = ""
        yield

        messages = state.messages
        if messages is None:
            messages = []
        messages.append(Message(role="user", content=user_input))
        me.scroll_into_view(key="scroll-to")
        yield

        response = chat_completion(state.messages, use_groq=not state.on_airplane_mode)
        messages.append(response)
        state.messages = messages
        me.focus_component(key=f"input-{len(state.messages)}")
        yield

    def on_click_toggle_airplane_mode(e: me.ClickEvent):
        state = me.state(State)
        state.on_airplane_mode = not state.on_airplane_mode

    def on_click_toggle_theme(e: me.ClickEvent):
        if me.theme_brightness() == "light":
            me.set_theme_mode("dark")
        else:
            me.set_theme_mode("light")

    # Style Helpers.

    def chat_bubble_postion(role: Role) -> me.Style:
        align_items = "end" if role == "user" else "start"
        return me.Style(
            display="flex",
            flex_direction="column",
            align_items=align_items,
        )

    def chat_bubble_style(role: Role) -> me.Style:
        if role != "user":
            return None
        return me.Style(
            width="70%",
            background=_USER_BUBBLE_COLOR,
            border_radius=16,
            padding=me.Padding(left=16, top=8, bottom=8),
            flex_grow=True,
            flex_direction="column",
        )

    def chat_input_native_textarea_style():
        font_color = "white" if me.theme_brightness() == "dark" else "black"
        return me.Style(
            padding=me.Padding(top=8, left=16, bottom=8),
            outline="none",
            width="100%",
            border=me.Border.all(me.BorderSide(style="none")),
            background=_USER_BUBBLE_COLOR,
            color=font_color,
        )

    # Components.

    # Toggle groq llama3 vs local llama3.
    with me.content_button(
        type="icon",
        style=me.Style(position="absolute", left=8, top=8),
        on_click=on_click_toggle_airplane_mode,
    ):
        me.icon("wifi_off" if state.on_airplane_mode else "wifi_on")

    # Toggle light/dark mode.
    with me.content_button(
        type="icon",
        style=me.Style(position="absolute", right=8, top=8),
        on_click=on_click_toggle_theme,
    ):
        me.icon("light_mode" if me.theme_brightness() == "dark" else "dark_mode")

    # Chat bounding box.
    with me.box(
        style=me.Style(
            height="100%",
            margin=me.Margin.symmetric(vertical=0, horizontal="auto"),
            width="min(1024px, 100%)",
            padding=me.Padding.all(20),
        )
    ):
        # Chat history.
        with me.box(
            style=me.Style(
                flex_grow=1,
                overflow_y="scroll",
                padding=me.Padding.all(20),
                margin=me.Margin(bottom=20),
            )
        ):
            for message in state.messages:
                with me.box(style=chat_bubble_postion(message.role)):
                    with me.box(style=chat_bubble_style(message.role)):
                        if message.role == "user":
                            # Hack to display newlines properly.
                            text = message.content.replace("\n", "<br>")
                            me.html(text)
                        else:
                            me.markdown(message.content)

        # Chat input box.
        with me.box(
            style=me.Style(
                border_radius=16,
                padding=me.Padding.all(8),
                background=_USER_BUBBLE_COLOR,
                display="flex",
                width="100%",
            )
        ):
            with me.box(style=me.Style(flex_grow=1)):
                me.native_textarea(
                    value=state.user_input,
                    placeholder="Enter a prompt",
                    on_blur=on_blur,
                    style=chat_input_native_textarea_style(),
                )
            with me.content_button(type="icon", on_click=on_click_submit):
                me.icon("send")

        # Empty box to add some margin below the chat input box as the page extends.
        with me.box(style=me.Style(height=50)):
            pass
