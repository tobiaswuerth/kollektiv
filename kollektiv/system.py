from .llm import LLMClient, Message
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
        self.llm = LLMClient(model_name="mistral-small3.1:latest")

    def run(self):
        history: list[Message] = [
            Message("system", assistant_priming).print(),
            Message("user", f"This is my goal:\n{self.goal}").print(),
        ]

        ###### Create Root node ######
        rootM, _ = self.llm.chat(
            message="Create the root node for the task decomposition tree",
            message_history=history,
            format=NodeModel,
        )
        root: Node = rootM.to_node()

        l1_nodes = self.produce_layer([root], history)
        l2_nodes = self.produce_layer(l1_nodes, history)
        l3_nodes = self.produce_layer(l2_nodes, history)

        print("-" * 50)
        print("Final tree:")
        print(root.to_json())

    def produce_layer(self, nodes: list[Node], history: list[Message]) -> list[Node]:
        root: Node = nodes[0].root
        history = history.copy()
        history.append(Message("system", f"The tree is now:\n{root.to_json()}").print())

        new_nodes: list[Node] = []
        for node in nodes:
            history_branch = history.copy()
            history_branch.append(
                Message(
                    "system",
                    f"@Assistant: You are now working on this node: (only node + ancestor-line is shown) \n{node.to_json()}",
                ).print()
            )

            nodeListM, history_branch = self.llm.chat(
                message="Your next step is to create child nodes indicating sub-steps of the active node you are working on",
                message_history=history_branch,
                format=NodeListModel,
            )

            nodeList: list[Node] = nodeListM.to_nodes()
            node.add_children(nodeList)
            new_nodes.extend(nodeList)

        return new_nodes
