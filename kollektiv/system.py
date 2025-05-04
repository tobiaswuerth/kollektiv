from .llm import LLMClient, Message, UserMessage, SystemMessage
from .llm import WebClient, Storage
from .response_models import ProblemDeconstructionTree, ProjectPlan

ASSISTANT_PRIMING = (
    "You are a versatile and highly capable AI assistant. Your primary goal is to understand and respond to user requests accurately, efficiently, and helpfully.\n\n"
    "Core Instructions:\n"
    "1. Accuracy and Factuality: Prioritize providing information that is accurate and fact-based. If information is uncertain or speculative, clearly indicate this. Avoid making up information.\n"
    "2. Instruction Adherence: Follow the user's instructions precisely. Pay close attention to the specific task requested, any constraints given, and the desired output.\n"
    "3. Conciseness and Clarity: Be clear and concise in your responses. Avoid unnecessary jargon or overly verbose explanations, but ensure the answer is complete and understandable.\n"
    "4. Systematic Processing: Approach tasks methodologically. For complex requests, break them down into logical steps. Think step-by-step to ensure a coherent and well-structured response.\n"
    "5. Format Compliance: Strictly adhere to any specified output formats (e.g., JSON, markdown, specific structuring, code blocks). If no format is specified, use clear and standard formatting.\n"
    "6. Language: Respond in clear and correct English, unless specifically instructed to use another language.\n"
    "7. Objectivity: Maintain a neutral and objective tone, unless the task specifically requires a different persona or style (e.g., creative writing).\n"
    "8. Helpfulness: Always aim to be helpful and provide relevant information or task execution that directly addresses the user's needs.\n\n"
    "Execute the user's request based on these principles.\n"
    "You will be guided in the sense that the System will tell you which tool to use and when."
)


class System:

    def __init__(self, goal: str):
        self.goal = goal
        self.llm = LLMClient(model_name="qwen3:32b")

    def run(self):
        debug = False
        self.llm.debug = debug

        history: list[Message] = [
            SystemMessage(ASSISTANT_PRIMING).print(not debug),
            UserMessage(f"This is my goal:\n{self.goal}").print(not debug),
        ]

        # # 1. Do research
        # self.llm.context_window = 8192
        # response, _ = self.llm.chat(
        #     message=(
        #         "Your task in this step is to figure out in principle how one tackles a project like this.\n"
        #         "You will do the following steps:\n"
        #         "1. Search for helpful resources on how to break the problem down into phases.\n"
        #         "2. Browse one of those resources to get in-depth information.\n"
        #         "3. Browse a second of those resources to diversify the information.\n"
        #         "4. Finally, respond with your reflections and the overall summary. "
        #         "Include all information that you think is relevant to the project. "
        #         "Assume the person executing the task might not have the tools available to research on their own."
        #     ),
        #     history=history,
        #     tools=[
        #         WebClient.web_search,
        #         WebClient.web_browse,
        #         WebClient.web_browse,
        #     ],
        # )
        # Storage.write_file("research.txt", response.strip())

        # # 2. Structure project into phases
        self.llm.context_window = 4096
        history.append(Storage.read_file("research.txt").print(not debug))
        # tree, _ = self.llm.chat(
        #     message=(
        #         "In the previous step you successfully figured out in principle how one tackles a project like this.\n"
        #         "Your task now is to do the following:\n"
        #         "1. Reflect on what suitable project phases are to achieve the goal.\n"
        #         "2. Finally, respond in the requested format and create the ProblemDeconstructionTree."
        #     ),
        #     history=history,
        #     format=ProblemDeconstructionTree,
        # )

        # Storage.write_file(
        #     "project_structure.json",
        #     tree.model_dump_json(indent=2),
        # )

        # 3. Create project plan with deliverables
        history.append(Storage.read_file("project_structure.json").print(not debug))
        plan, _ = self.llm.chat(
            message=(
                "In the previous step you successfully created a project structure with phases of how to tackle the project.\n"
                "Your task now is to do the following:\n"
                "1. Think through each phase and note down the following:\n"
                "   - What files are expected to be produced as part of this phase? (valid extensions are [.txt, .md, .json])\n"
                "   - What files are required as input for this phase?\n"
                "2. Finally, respond in the requested format and create the ProjectPlan."
            ),
            history=history,
            format=ProjectPlan,
        )
        Storage.write_file(
            "project_plan.json",
            plan.model_dump_json(indent=2),
        )
