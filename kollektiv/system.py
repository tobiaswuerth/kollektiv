from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


@tool(description="Requests human assistance")
def human_assistance(query: str) -> str:
    human_response = interrupt({"query": query})
    return human_response["data"]


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
        self.tools = [self.search, human_assistance]

        self.llm = init_chat_model("ollama:mistral-nemo")
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        self.memory = MemorySaver()

    def run(self):

        def chatbot(state: State):
            return {"messages": [self.llm_with_tools.invoke(state["messages"])]}

        # The argument is the function or object that will be called whenever
        # the node is used.
        graph_builder = StateGraph(State)

        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_node("chatbot", chatbot)

        tool_node = ToolNode(tools=self.tools)
        graph_builder.add_node("tools", tool_node)

        graph_builder.add_conditional_edges(
            "chatbot",
            tools_condition,
        )
        graph_builder.add_edge("tools", "chatbot")  # retour
        graph = graph_builder.compile(checkpointer=self.memory)

        # visualize_graph(graph)

        config = {"configurable": {"thread_id": "1"}}

        def stream_graph_updates(user_input: str):
            events = graph.stream(
                {"messages": [{"role": "user", "content": user_input}]},
                config,
                stream_mode="values",
            )
            for event in events:
                event["messages"][-1].pretty_print()

        while True:
            try:
                user_input = input("User: ")
                if user_input.lower() in ["quit", "exit", "q"]:
                    print("Goodbye!")
                    break

                stream_graph_updates(user_input)
            except Exception as e:
                print(f"Error: {e}")
                break
