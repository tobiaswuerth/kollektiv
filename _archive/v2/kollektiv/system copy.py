from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt
from langchain_core.messages import ToolMessage


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]
    name: str
    birthday: str


# Note that because we are generating a ToolMessage for a state update, we
# generally require the ID of the corresponding tool call. We can use
# LangChain's InjectedToolCallId to signal that this argument should not
# be revealed to the model in the tool's schema.
@tool(description="Requests human assistance")
def human_assistance(name: str, birthday: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> str:
    """Request assistance from a human."""
    human_response = interrupt(
        {
            "question": "Is this correct?",
            "name": name,
            "birthday": birthday,
        },
    )
    # If the information is correct, update the state as-is.
    if human_response.get("correct", "").lower().startswith("y"):
        verified_name = name
        verified_birthday = birthday
        response = "Correct"
    # Otherwise, receive information from the human reviewer.
    else:
        verified_name = human_response.get("name", name)
        verified_birthday = human_response.get("birthday", birthday)
        response = f"Made a correction: {human_response}"

    # This time we explicitly update the state with a ToolMessage inside
    # the tool.
    state_update = {
        "name": verified_name,
        "birthday": verified_birthday,
        "messages": [ToolMessage(response, tool_call_id=tool_call_id)],
    }
    # We return a Command object in the tool to update our state.
    return Command(update=state_update)


def visualize_graph(graph):
    try:
        graph.get_graph().draw_mermaid_png(output_file_path="output/graph.png")
        from PIL import Image

        img = Image.open("output/graph.png")
        img.show()
    except Exception as e:
        print("Error displaying graph:", e)


class System:
    def __init__(self):
        self.search = DuckDuckGoSearchRun()
        self.tools = [
            self.search,
            human_assistance,
        ]

        # self.llm = init_chat_model("ollama:mistral-nemo")
        self.llm = init_chat_model("ollama:mistral-small3.1:latest")
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        self.memory = MemorySaver()

    def run(self):

        def chatbot(state: State):
            message = self.llm_with_tools.invoke(state["messages"])
            assert len(message.tool_calls) <= 1
            return {"messages": [message]}

        # The argument is the function or object that will be called whenever
        # the node is used.
        graph_builder = StateGraph(State)

        graph_builder.add_node("chatbot", chatbot)
        graph_builder.add_edge(START, "chatbot")

        tool_node = ToolNode(tools=self.tools)
        graph_builder.add_node("tools", tool_node)

        graph_builder.add_conditional_edges(
            "chatbot",
            tools_condition,
        )
        graph_builder.add_edge("tools", "chatbot")  # retour
        graph = graph_builder.compile(
            checkpointer=self.memory,
        )

        # visualize_graph(graph)

        config = {"configurable": {"thread_id": "1"}}

        def stream_graph_updates(user_input: str):
            events = graph.stream(
                {"messages": [{"role": "user", "content": user_input}]},
                config,
                stream_mode="values",
            )
            for event in events:
                if "messages" in event:
                    event["messages"][-1].pretty_print()
                else:
                    print(event)


        user_input = (
            "Lookup when LangGraph was released. "
            "When you have the answer, pass your best guess to the human_assistance tool for review. "
            "Do not ask for confirmation and do use the tools required."
        )
        stream_graph_updates(user_input)

        # while True:
        #     try:

        #         user_input = input("User: ")
        #         if user_input.lower() in ["quit", "exit", "q"]:
        #             print("Goodbye!")
        #             break

        #         stream_graph_updates(user_input)
        #     except Exception as e:
        #         print(f"Error: {e}")
        #         break
