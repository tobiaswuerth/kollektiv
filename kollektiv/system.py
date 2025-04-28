from typing import Annotated, Literal
from pydantic import BaseModel, Field
from typing_extensions import TypedDict 

from langchain.chat_models import init_chat_model
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import InjectedToolCallId, tool
from langchain_core.messages import ToolMessage, HumanMessage, SystemMessage, AIMessage
from langchain.output_parsers import RetryOutputParser

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt

from .nodes import DecisionNode, InfoNode, ChatNode


class CreateNode(BaseModel):
    attached_to: int = Field(description="The ID of the node to attach to")
    node_title: str = Field(description="The title of the node")


class DeleteNode(BaseModel):
    node_id: int = Field(description="The ID of the node to delete")


class UpdateNode(BaseModel):
    node_id: int = Field(description="The ID of the node to update")
    new_title: str = Field(description="The new title of the node")


class Node(BaseModel):
    id: int = Field(
        default_factory=lambda self: id(self), description="The ID of the node"
    )
    title: str = Field(description="The title of the node")
    children: list["Node"] = Field(
        default_factory=list, description="The children of the node"
    )


problem_tree = Node(
    title="Problem",
)


class State(TypedDict):
    messages: Annotated[list, add_messages]


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
        ]

        # self.llm = init_chat_model("ollama:mistral-nemo")
        self.llm = init_chat_model("ollama:mistral-small3.1:latest")
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        self.memory = MemorySaver()
        self.config = {"configurable": {"thread_id": "1"}}
        self.printed_messages = 0

    def stream_graph_updates(self, graph: CompiledStateGraph, user_input: str):
        events = graph.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            self.config,
            stream_mode="values",
        )
        for event in events:
            if "messages" in event:
                for i, message in enumerate(event["messages"]):
                    if i < self.printed_messages:
                        continue
                    self.printed_messages += 1
                    message.pretty_print()
            else:
                print(event)


    def build_graph(self):
        builder = StateGraph(State)

        # Phase 1: Research Phase

        builder.add_edge(START, "instructions")
        builder.add_node("instructions", ChatNode(self.llm_with_tools))
        builder.add_edge("instructions", "websearch")
        builder.add_node("websearch", ToolNode(tools=[self.search]))
        builder.add_edge("websearch", "websearch_info")
        builder.add_node("websearch_info", InfoNode((
            "Now that you have the result, would you like to search the internet for more information "
            "or are you confident you have enough information to start breaking down the problem?"
            "If you decide on needs_more_research, you will be able to search the internet again. "
            "If you decide on finish_research_phase, you will be able to start breaking down the problem. "
        )))
        builder.add_edge("websearch_info", "decide_more_research")

        n_dmr = DecisionNode(self.llm,
            decision_routes={
                "needs_more_research": "websearch",
                "finish_research_phase": END,
            }
        )
        builder.add_node("decide_more_research", n_dmr)
        builder.add_conditional_edges("decide_more_research", n_dmr.f_cond_paths)

        # Phase 2: Breaking down the problem
        # todo

        # Finalize
        return builder.compile(
            checkpointer=self.memory,
        )

    def run(self):
        graph = self.build_graph()
        visualize_graph(graph)
        raise
        user_input = (
            "You will be given a goal and your task is to make a plan to achieve that goal. "
            "Your will break the problem down into smaller parts. "
            "Each part may be divided into smaller parts, "
            "until each element consists only of the smallest possible action. "
            ""
            "Your goal is: Write a story. "
            "The story must have 10 chapters and each chapter consist of around 1000 words. "
            "The story must be a fantasy sci-fi story with a novel plot in a post-apocalyptic world. "
            "The required output are 10 individual markdown files, one for each chapter. "
            "The files must be named chapter_1.md, chapter_2.md, etc. "
            "The story must further be consistent, coherent and following a logical structure. "
            "Generally in this interaction, keep your answers short and concise. "
            "Your first task is to search the internet for potential approaches on how people go about it. "
            "Use the search engine to find relevant information. "
        )
        self.stream_graph_updates(graph, user_input)
