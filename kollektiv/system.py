from .llm import LLMClient, Message, UserMessage
from .node import Node, NodeModel, NodeListModel
from .tools.web import WebClient

ASSISTANT_PRIMING = (
    "You are an AI assistant expertly skilled in TASK DECOMPOSITION and CREATING HIERARCHICAL OUTLINES.\n"
    "Your primary function is to help break down large or complex goals, projects, and ideas into smaller, more manageable, and ultimately actionable steps, structured as a tree of nodes.\n"
    "\n"
    "You understand the process of creating nested outlines:\n"
    "Starting with a high-level concept (the root node).\n"
    "Identifying the main constituent parts or phases (level 1 children).\n"
    "Recursively detailing each part into its own sub-steps (levels 2, 3, etc.).\n"
    "Recognizing when a step is simple enough it requires no further breakdown, especially when reaching steps executable by a simple agent.\n"
    "\n"
    "Regardless of the specific task you are given (e.g., creating a root, breaking down a node, adding tool steps):\n"
    "Be METHODICAL and SYSTEMATIC in your approach.\n"
    "Ensure logical consistency and flow within the generated list of nodes and relative to the parent node/overall tree.\n"
    "Maintain CONCISENESS in all descriptions.\n"
    "Always think step-by-step about the components and their order relevant to the current task.\n"
)

from pydantic import BaseModel, Field
class MyStepByStepGuide(BaseModel):
    steps: list[str] = Field(
        description="A list of steps to achieve the goal."
    )


class System:

    def __init__(self, goal: str):
        self.goal = goal

        # self.llm = LLMClient(model_name="deepseek-r1:14b")
        # self.llm = LLMClient(model_name="deepseek-r1:32b")
        self.llm = LLMClient(model_name="qwen3:32b")
        # self.llm = LLMClient(model_name="mistral-nemo:latest")
        # self.llm = LLMClient(model_name="mistral-small3.1:latest")
        # self.llm = LLMClient(model_name="llama3.3:latest")

    def run(self):
        web_client = WebClient()

        response = self.llm.chat(
            message="Figure out how to bake a blueberry cheescake and provide me with a short summary of the steps involved.",
            tools=[web_client.search, web_client.browse],
            format=MyStepByStepGuide,
        )
        print(response)
        raise

        history_base: list[Message] = [
            UserMessage(ASSISTANT_PRIMING).print(),
            UserMessage(f"This is my goal:\n{self.goal}").print(),
        ]

        self.llm.context_window = 2048
        rootM, history = self.llm.chat(
            message=(
                "CREATE the single ROOT NODE for the task decomposition tree based on the goal provided.\n"
                "This root node MUST represent the ENTIRE global task or goal.\n"
                "\n"
                "Write the DESCRIPTION for this root node following these rules:\n"
                "RULE 1: COMPLETE Summary: The description must be a COMPLETE summary of the overall goal.\n"
                "RULE 2: Clarity: It must be clear enough that SOMEONE WITH NO PRIOR KNOWLEDGE can understand: What the task is. Why the task is being done (its purpose or desired outcome).\n"
                "RULE 3: High-Level Only: DO NOT include any sub-tasks, steps, or breakdown of the task in this root node description.\n"
            ),
            message_history=history_base,
            format=NodeModel,
        )
        root: Node = rootM.to_node()
        