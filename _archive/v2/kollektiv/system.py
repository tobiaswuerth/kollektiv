from typing import Annotated
from typing_extensions import TypedDict

from langchain.chat_models import init_chat_model
from langchain_community.tools import DuckDuckGoSearchRun

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

from .nodes import DecisionNode, InfoNode, ChatNode, TypedValidatorNode
from .structure import Node, ProblemStructurizer, CreateRoot, CreateNode, DeleteNode, UpdateNode


class State(TypedDict):
    messages: Annotated[list, add_messages]


def visualize_graph(graph:CompiledStateGraph):
    try:
        graph.get_graph().draw_png(
            output_file_path="output/graph.png",
        )
        from PIL import Image
        img = Image.open("output/graph.png")
        img.show()
    except Exception as e:
        print("Error displaying graph:", e)


class System:
    def __init__(self, goal: str):
        self.goal = goal

        self.structurizer = ProblemStructurizer()
        self.search = DuckDuckGoSearchRun()

        # self.llm = init_chat_model("ollama:mistral-nemo")
        self.llm = init_chat_model("ollama:mistral-small3.1:latest")

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
            for i, message in enumerate(event["messages"]):
                if i < self.printed_messages:
                    continue
                self.printed_messages += 1
                message.pretty_print()

    def build_graph_research_and_root(self):
        builder = StateGraph(State)

        # Phase 1: Research Phase

        builder.add_edge(START, "instructions")
        builder.add_node("instructions", ChatNode(self.llm.bind_tools([self.search])))
        builder.add_edge("instructions", "websearch")
        builder.add_node("websearch", ToolNode(tools=[self.search]))
        builder.add_edge("websearch", "websearch_info")
        builder.add_node("websearch_info", InfoNode(
                (
                    "Now that you have the result, would you like to search the internet for more information "
                    "or are you confident you have enough information to start breaking down the problem?"
                    "If you decide on needs_more_research, you will be able to search the internet again. "
                    "If you decide on finish_research_phase, you will be able to start breaking down the problem. "
                )
            ),
        )
        builder.add_edge("websearch_info", "decide_more_research")

        n_dmr = DecisionNode(
            self.llm,
            decision_routes={
                "needs_more_research": "instructions",
                "finish_research_phase": "structure_intro",
            },
        )
        builder.add_node("decide_more_research", n_dmr)
        builder.add_conditional_edges("decide_more_research", n_dmr.f_cond_paths)
        
        # Phase 2: Breaking down the problem
        builder.add_node("structure_intro", InfoNode((
            "Now that you have enough information, you can start breaking down the problem. "
            "You will be able to create a root node and then create child nodes. "
            "I will guide you through the whole process. "
            "You will only have to provide the information requested in small bits. "
            "The final result will be a tree structure, where each node represents a part of the problem. "
            "We start now with the root node. Please use the provided tool to create the root node. "
        )))
        builder.add_edge("structure_intro", "ask_for_root")
        builder.add_node("ask_for_root", ChatNode(self.llm.bind_tools([self.structurizer.create_root])))
        builder.add_edge("ask_for_root", "create_root")
        builder.add_node("create_root", ToolNode(tools=[self.structurizer.create_root]))
        builder.add_edge("create_root", END)

        # Finalize
        return builder.compile(
            checkpointer=self.memory,
        )

    def build_graph_tree_layer1(self):
        builder = StateGraph(State)

        builder.add_edge(START, "info_row")
        builder.add_node("info_row", InfoNode(
                (
                    "Now that you have the root node, you can start creating child nodes. "
                    "You will be able to create a sequence of nodes and attach them to the active node. "
                    "I will guide you through the whole process. "
                    "You will only have to provide the information requested in small bits. "
                    "The final result will be a tree structure, where each node represents a part of the problem. "
                    "We start now with the first layer of child nodes. Please use the provided tool to create the child nodes. "
                    "Now, think, and use the tool to break down the active node into 3-5 child nodes. "
                )
            ),
        )
        
        # assume active node is root
        builder.add_edge("info_row", "get_create_node_row")
        builder.add_node("get_create_node_row", ChatNode(self.llm.bind_tools([self.structurizer.create_node_row])))
        builder.add_edge("get_create_node_row", "create_node_row")
        builder.add_node("create_node_row", ToolNode(tools=[self.structurizer.create_node_row]))
        builder.add_edge("create_node_row", END)
        
        # Finalize
        return builder.compile(
            checkpointer=self.memory,
        )
    
    def build_graph_tree_layer2(self):
        builder = StateGraph(State)

        nodes = self.structurizer.active_node.children
        starts = [START] + [f"create_node_row_{i}" for i in range(len(nodes)-1)]

        for ni, (node, start) in enumerate(zip(nodes, starts)):
            builder.add_edge(start, f'activate_node_{ni}')
            builder.add_node(f'activate_node_{ni}', self.structurizer.get_activation_node(node.id))
            builder.add_edge(f'activate_node_{ni}', f'info_row_{ni}')
            builder.add_node(f'info_row_{ni}', InfoNode(
                (
                    f"Now that you have the node {node.title} active, you can start creating child nodes. "
                    f"You will be able to create a sequence of nodes and attach them to the active node. "
                    "Now, think, and use the tool to break down the active node into 3-5 child nodes. "
                )
            ))
            builder.add_edge(f'info_row_{ni}', f'get_create_node_row_{ni}')
            builder.add_node(f'get_create_node_row_{ni}', ChatNode(self.llm.bind_tools([self.structurizer.create_node_row])))
            builder.add_edge(f'get_create_node_row_{ni}', f'create_node_row_{ni}')
            builder.add_node(f'create_node_row_{ni}', ToolNode(tools=[self.structurizer.create_node_row]))
        
        builder.add_edge(f'create_node_row_{len(nodes)-1}', END)
        
        # Finalize
        return builder.compile(
            checkpointer=self.memory,
        )
    
    def run(self):
        graph = self.build_graph_research_and_root()
        visualize_graph(graph)
        # raise
        user_input = (
            "You will be given a goal and your task is to make a plan to achieve that goal. "
            "Your will break the problem down into smaller parts. "
            "Each part may be divided into smaller parts, "
            "until each element consists only of the smallest possible action.\n"
            f"Your goal is: {self.goal}\n"
            "Generally in this interaction, keep your answers short and concise. "
            "Your first task is to search the internet for potential approaches on how people go about it. "
            "Use the search engine to find relevant information. "
        )
        self.stream_graph_updates(graph, user_input)

        # Phase 2: Breaking down the problem
        graph = self.build_graph_tree_layer1()
        visualize_graph(graph)
        # raise
        user_input = (
            "Now that you have enough information, you can start breaking down the problem. "
            "We will go through each layer of the tree and you break it down into smaller parts. "
        )
        self.stream_graph_updates(graph, user_input)

        # Phase 3: Breaking down the problem further
        graph = self.build_graph_tree_layer2()
        visualize_graph(graph)
        # raise
        user_input = (
            "We will go through each layer of the tree and you break it down into smaller parts. "
        )
        self.stream_graph_updates(graph, user_input)

        print('done')
