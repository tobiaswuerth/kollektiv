from .llm import LLMClient, Message, UserMessage
from .node import Node, NodeModel, NodeListModel

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
            history=history_base,
            format=NodeModel,
        )
        root: Node = rootM.to_node()

        # create main tasks
        self.llm.context_window = 4096
        nodeListM, _ = self.llm.chat(
            message=(
                "You have created the main goal node (the root):\n"
                f"```json\n{root.to_json()}\n```\n"
                "\n"
                "Your next task is to identify and list the MAIN, FIRST-LEVEL steps or components needed to directly achieve this goal.\n"
                "These will be the DIRECT children nodes of the root node.\n"
                "\n"
                "Create a LIST of these main task nodes. For each task node in the list, follow these specific instructions:\n"
                "RULE 1: INDEPENDENT: Each task must represent a distinct, stand-alone part of the overall goal.\n"
                "RULE 2: SEPARATE: Tasks should be clearly different from each other with no overlap.\n"
                "RULE 3: RELEVANT: Each task must directly fit under and contribute to the root node's goal.\n"
                "RULE 4: CONSISTENT DETAIL: All tasks in THIS list should have a similar, high-level of detail.\n"
                "RULE 5: HIGH-LEVEL ONLY: FOCUS ONLY on the major parts or phases. DO NOT break these main tasks down into smaller sub-steps yet.\n"
                "RULE 6: LOGICAL ORDER: Arrange the tasks in the sequence they would typically be performed to progress towards the root goal.\n"
                "RULE 7: NO DUPLICATION: The generated tasks must NOT duplicate tasks conceptually already covered by the root or by siblings in this list.\n"
            ),
            history=history,
            format=NodeListModel,
        )
        nodeList: list[Node] = nodeListM.to_nodes()
        root.add_children(nodeList)

        # break down main tasks
        self.llm.context_window = 16384
        new_nodes = []
        for node in nodeList:
            nodeListM, _ = self.llm.chat(
                message=(
                    "Here is the current structure of the task decomposition tree built so far:\n"
                    f"```json\n{root.to_json()}\n```\n"
                    "\n"
                    "You need to continue breaking down the overall goal by detailing one of the intermediate tasks.\n"
                    "Your specific focus is on THIS parent node, which needs further breakdown into its immediate steps:\n"
                    f"```json\n{node.to_json()}\n```\n"
                    "\n"
                    "Based on the ENTIRE tree context provided ABOVE and focusing specifically on the parent node shown HERE, create a LIST of the IMMEDIATE child nodes (sub-tasks) for THIS specific parent node.\n"
                    "These child nodes should represent the distinct steps or components necessary to complete the specific parent task.\n"
                    "\n"
                    "Follow these strict guidelines for the list of child nodes you generate:\n"
                    "RULE 1: BELONG to Parent: Each child node MUST be a direct step or part required to complete the specific parent task shown above.\n"
                    "RULE 2: DISTINCT & SEPARATE: Ensure each child node is a unique, independent item. They must be clearly separate from each other.\n"
                    "RULE 3: CONSISTENT DETAIL: All child nodes in THIS list should have a similar level of detail. This detail level should be the logical NEXT step down from the parent, but not yet the final tool-level steps.\n"
                    "RULE 4: NO DUPLICATION: The generated tasks must NOT duplicate tasks already present elsewhere in the ENTIRE tree structure provided.\n"
                    "RULE 5: LOGICAL ORDER: Arrange the child nodes in the most logical sequence for performing the specific parent task.\n"
                    "RULE 6: SUFFICIENT BREAKDOWN: Break down the parent task into its necessary sub-steps. If the parent task is already simple enough that its immediate children would be tool-executable steps, generate those CONCEPTUAL steps here, even if not yet strictly tool calls.\n"
                ),
                history=history_base,
                format=NodeListModel,
            )
            nodeList: list[Node] = nodeListM.to_nodes()
            node.add_children(nodeList)
            new_nodes.extend(nodeList)

        # start introducing the tools
        self.llm.context_window = 65536
        for node in new_nodes:
            nodeListM, _ = self.llm.chat(
                message=(
                    "Here is the current structure of the task decomposition tree built so far:\n"
                    f"```json\n{root.to_json()}\n```\n"
                    "\n"
                    "You have broken down the goal significantly. Now, you must break down the tasks into steps executable by a simple agent with tools.\n"
                    "Your specific focus is on THIS parent node, which requires breakdown into tool-based steps:\n"
                    f"```json\n{node.to_json()}\n```\n"
                    "\n"
                    "The agent executing these tasks is extremely simple and has access to ONLY these tools:\n"
                    "list_files: List all files in the output directory.\n"
                    "write_file: Write content to a file in the output directory. IMPORTANT: This tool OVERWRITES the file if it already exists.\n"
                    "delete_file: Delete a file in the output directory.\n"
                    "\n"
                    "The agent has NO MEMORY or prior knowledge of the task, the tree, or files, EXCEPT what is explicitly stated in the current task description it is executing.\n"
                    "Each child node you create for the parent task above MUST be a standalone, precise, actionable instruction for the agent using the available tools. Think of each node as one command or a single, clear mental step for the agent.\n"
                    "Create a LIST of these highly specific, tool-based child task nodes for the parent node above.\n"
                    "\n"
                    "Follow these STRICT rules for the list of child nodes:\n"
                    "RULE 1: ACTIONABLE STEP: Each node description must describe a single, clear action the agent can take, often involving a tool call. Start the description with an imperative verb.\n"
                    "RULE 2: USE AVAILABLE TOOLS: ONLY reference the tools listed above (list_files, write_file, delete_file). DO NOT invent or mention other tools or require capabilities the agent doesn't have.\n"
                    "RULE 3: EXTREME SPECIFICITY: Use EXTREMELY specific, unique, and descriptive filenames (e.g., initial_plan.md, section_1_draft.txt, final_report.md). DO NOT use generic terms like 'the file', 'input', 'output', 'previous result', or 'review files'. Filenames MUST be literal strings the agent can use.\n"
                    "RULE 4: EXPLICIT REFERENCES: If a step requires information from a file created in a PREVIOUS step (within the parent task's sequence or from a completed sibling/parent in the overall tree), the FULL, SPECIFIC filename MUST be explicitly mentioned in the CURRENT task description. DO NOT assume the agent remembers filenames or content from previous steps.\n"
                    "RULE 5: INDEPENDENT INSTRUCTION: Each child node MUST contain all information needed for the agent to understand AND ATTEMPT TO EXECUTE that specific step, assuming no memory of previous steps other than what's referenced by explicit filenames.\n"
                    "RULE 6: CLEAR BOUNDARIES: Ensure steps are separate and sequential operations.\n"
                    "RULE 7: UNIFORM DETAIL: All child nodes in THIS list MUST be at the same, LOWEST level of detail, ready for direct execution by the simple agent.\n"
                    "RULE 8: NO DUPLICATION: The generated tasks MUST NOT duplicate steps already present elsewhere in the ENTIRE tree structure provided.\n"
                    "RULE 9: LOGICAL SEQUENCE: Order the child nodes step-by-step to precisely complete the specific parent task from start to finish using the available tools.\n"
                    "RULE 10: TERMINAL NODES: The nodes generated at this level are the final steps and should not require further breakdown. DO NOT suggest sub-children for these nodes.\n"
                    "\n"
                    "Given the parent task, the available tools, and the agent's limitations, generate the list of these highly specific, tool-based child task nodes for the specified parent.\n"
                ),
                history=history_base,
                format=NodeListModel,
            )
            nodeList: list[Node] = nodeListM.to_nodes()
            node.add_children(nodeList)

        print("-" * 50)
        print("Final tree:")
        print(root.to_json())
