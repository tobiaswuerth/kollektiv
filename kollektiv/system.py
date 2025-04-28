from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import ToolMessage
import json


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]

class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            print(f"Tool call: {tool_call}")
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}


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
        self.llm = init_chat_model("ollama:mistral-nemo")
        self.search = DuckDuckGoSearchRun()
        self.llm_with_search = self.llm.bind_tools([self.search])

    def run(self):

        def chatbot(state: State):
            return {"messages": [self.llm_with_search.invoke(state["messages"])]}

        # The first argument is the unique node name
        # The second argument is the function or object that will be called whenever
        # the node is used.
        graph_builder = StateGraph(State)


        graph_builder.add_node("chatbot", chatbot)
        tool_node = BasicToolNode(tools=[self.search])
        graph_builder.add_node("tools", tool_node)

        def route_tools(
            state: State,
        ):
            """
            Use in the conditional_edge to route to the ToolNode if the last message
            has tool calls. Otherwise, route to the end.
            """
            if isinstance(state, list):
                ai_message = state[-1]
            elif messages := state.get("messages", []):
                ai_message = messages[-1]
            else:
                raise ValueError(f"No messages found in input state to tool_edge: {state}")
            if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
                return "tools"
            return END

        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_conditional_edges(
            "chatbot",
            route_tools,
            # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
            # It defaults to the identity function, but if you
            # want to use a node named something else apart from "tools",
            # You can update the value of the dictionary to something else
            # e.g., "tools": "my_tools"
            {"tools": "tools", END: END},
        )
        graph_builder.add_edge("tools", "chatbot")
        graph = graph_builder.compile()

        # visualize_graph(graph)

        def stream_graph_updates(user_input: str):
            for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
                for value in event.values():
                    print(value['messages'][-1].__class__.__name__, ":", value['messages'][-1].content)
                    print()

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