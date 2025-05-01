from .llm import LLMClient, Message, UserMessage
from .node import Node, NodeModel, NodeListModel


assistant_priming = """
# IDENTITY and PURPOSE

You are an AI assistant specialized in task decomposition and recursive outlining.
Your primary role is to take complex tasks, projects, or ideas and break them down into smaller, more manageable components.
You excel at identifying the core purpose of any given task and systematically creating hierarchical outlines that capture all essential elements.
Your expertise lies in recursively analyzing each component, ensuring that every aspect is broken down to its simplest, actionable form.
Whether it's an article that needs structuring or an application that requires development planning, you approach each task with the same methodical precision.
You are adept at recognizing when a subtask has reached a level of simplicity that requires no further breakdown, ensuring that the final outline is comprehensive yet practical.
Take a step back and think step-by-step about how to achieve the best possible results by following the steps below.

# STEPS

- Identify the main task or project presented by the user
- Determine the overall purpose or goal of the task
- Create a high-level outline of the main components or sections needed to complete the task
- For each main component or section:
  - Identify its specific purpose
  - Break it down into smaller subtasks or subsections
  - Continue this process recursively until each subtask is simple enough to not require further breakdown
- Review the entire outline to ensure completeness and logical flow
- Present the finalized recursive outline to the user
"""


class System:

    def __init__(self, goal: str):
        self.goal = goal

        self.llm = LLMClient(model_name="deepseek-r1:14b")
        # self.llm = LLMClient(model_name="deepseek-r1:32b")
        # self.llm = LLMClient(model_name="mistral-nemo:latest")
        # self.llm = LLMClient(model_name="mistral-small3.1:latest")
        # self.llm = LLMClient(model_name="llama3.3:latest")
        # self.llm = LLMClient()

    def run(self):
        history: list[Message] = [
            UserMessage(assistant_priming).print(),
            UserMessage(f"This is my goal:\n{self.goal}").print(),
        ]

        rootM, _ = self.llm.chat(
            message="Create the root node for the task decomposition tree",
            message_history=history,
            format=NodeModel,
        )
        root: Node = rootM.to_node()

        l1_nodes = self.produce_layer([root], history)
        l2_nodes = self.produce_layer(l1_nodes, history)
        # l3_nodes = self.produce_layer(l2_nodes, history)

        print("-" * 50)
        print("Final tree:")
        print(root.to_json())

    def produce_layer(self, nodes: list[Node], history: list[Message]) -> list[Node]:
        root: Node = nodes[0].root

        history_base = history.copy()
        history_base.append(
            UserMessage((
                    "You have been working on planning out the steps needed to achieve the goal.\n"
                    "The full tree is currently like this:\n\n```\n"
                    f"{root.to_json()}\n```\n\n"
                    "Important: The person executing those tasks has only access to the following tools:\n"
                    "1. list_files\n"
                    "2. read_file\n"
                    "3. write_file\n"
                    "4. delete_file\n"
                    "Your breakdown must be structured such that the person knows exactly what to do even without prior knowledge.\n"
                    "This means that each step ideally has a clear input, action and output.\n"
                    "On the higher levels of the tree this might not be possible, but on the lower levels it is a must."
            )).print()
        )

        new_nodes: list[Node] = []
        for node in nodes:
            history_branch = history_base.copy()
            history_branch.append(
                UserMessage((
                    "You have been working on the following node:\n\n"
                    f"{node.to_json()}\n\n"
                    "Create a new set of child nodes for this node.\n"
                    "The new nodes should be unique given the whole of the tree and fit its parent."
                )).print()
            )

            nodeListM, history_branch = self.llm.chat(
                message=(
                    "Create a new set of child nodes for the node you are currently working on.\n"
                    "The new nodes should be unique given the whole of the tree and fit its parent."
                ),
                message_history=history_branch,
                format=NodeListModel,
            )

            nodeList: list[Node] = nodeListM.to_nodes()
            node.add_children(nodeList)
            new_nodes.extend(nodeList)

        return new_nodes
