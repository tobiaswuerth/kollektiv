import os

from .llm import LLMClient, Message, UserMessage, SystemMessage
from .llm import WebClient, Storage

from .utils import save_pydantic_json, load_pydantic_json

from .models.models_phase2_phases import Project
from .models.models_phase3_deliverables import ProjectWithDeliverables
from .models.models_phase4_deliverable_tasks import ProjectWithTasks, TaskList

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

        # Phase 1. Do research
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
        history.append(Storage.read_file("research.txt").print(not debug))

        # Phase 2. Structure project into phases
        # self.llm.context_window = 4096
        # project, _ = self.llm.chat(
        #     message=(
        #         "In the previous step you successfully figured out in principle how one tackles a project like this.\n"
        #         "Your task now is to do the following:\n"
        #         "1. Reflect on what suitable project phases are to achieve the goal.\n"
        #         "2. Finally, respond in the requested format and create the Project hierarchy."
        #     ),
        #     history=history,
        #     format=Project,
        # )
        # save_pydantic_json(project, "project_structure.json")
        history.append(Storage.read_file("project_structure.json").print(not debug))

        # Phase 3. Create project plan with deliverables
        # self.llm.context_window = 6144
        # plan, _ = self.llm.chat(
        #     message=(
        #         "In the previous step you successfully created a project structure with phases of how to tackle the project.\n"
        #         "\n"
        #         "Your task now is to do the following:\n"
        #         "1. Think through each phase and note down the following:\n"
        #         "   - What files are expected to be produced as part of this phase?\n"
        #         "   - What files are required as input for this phase?\n"
        #         "2. Finally, respond in the requested format and create the ProjectPlan.\n"
        #         "\n"
        #         "Note: The already existing files (research.txt, project_structure.json) are also valid input files "
        #         "and must also be considered as input files if required.\n"
        #         "\n"
        #         "IMPORTANT: Choose the file names carefully. "
        #         "The file names should be descriptive and indicate the content of the file. Avoid re-using file names. "
        #         "For example, if in the first phase you create a file called `foo.txt`, "
        #         "do not use the same name in the second phase to modify or overwrite its contents, "
        #         "rather consider a suffix like `foo_draft.txt`, `foo_edited.txt`, or `foo_final.txt` etc."
        #     ),
        #     history=history,
        #     format=ProjectWithDeliverables,
        # )
        # save_pydantic_json(plan, "project_plan.json")
        history.append(Storage.read_file("project_plan.json").print(not debug))

        # Phase 4. Go through each phase and analyze the todos
        # self.llm.context_window = 6144
        # plan = load_pydantic_json("project_plan.json", ProjectWithDeliverables)
        # plan = ProjectWithTasks.from_plan(plan)
        # for phase in plan.project_phases:
        #     taskListM, _ = self.llm.chat(
        #         message=(
        #             f"In the previous steps you successfully created a project plan with phases and deliverables.\n"
        #             f"We will go through each phase individually now upon my instruction to analyze the todos.\n"
        #             "\n"
        #             f"For this iteration, you are working on the phase '{phase.phase_name}'.\n"
        #             f"```json\n{phase.model_dump_json(indent=2)}\n```\n"
        #             "\n"
        #             f"Your task now is to do the following:\n"
        #             f"1. Think about what steps need to be performed to complete the phase '{phase.phase_name}'.\n"
        #             f"2. Given the available input files of this phase overall, which of these files are required for the individual steps?\n"
        #             f"3. Given the expected output files of this phase overall, which of these files are produced by the individual steps?\n"
        #             f"4. Finally, respond in the requested format to populate the TaskList.\n"
        #             "\n"
        #             "Note: The agent performing these tasks are stateless, if you a step produces information that is required for a later step, "
        #             "you need to include this information in the output of the step by writing it to that file.\n"
        #             "Only files that are an expected deliverable of this phase can be produced in this phase. "
        #             "However, the goal is to create atomic actions that are as small as possible, "
        #             "therefore you may produce intermediate files that are not part of the expected deliverables. "
        #             "These intermediate files are only required for the next step and will be deleted after the phase is over. "
        #             "For example, you may create a TMP_foo_draft.txt file that is only required for the next step to produce the final foo.txt file.\n"
        #             "Split the tasks into such small atomic actions that each task produces exactly one file. "
        #             "Each task may have multiple required input files though.\n"
        #             "\n"
        #             "IMPORTANT: Before you respond, verify that: The tasks include all required input files, otherwise the information is not available and overall coherence is lost."
        #         ),
        #         history=history,
        #         format=TaskList,
        #     )
        #     phase.tasks = taskListM.tasks
        #     save_pydantic_json(phase, "project_plan_with_tasks.json")

        history.append(Storage.read_file("project_plan_with_tasks.json").print(not debug))
        plan = load_pydantic_json("project_plan_with_tasks.json", ProjectWithTasks)
        
        from .utils import generate_project_plan_graph
        generate_project_plan_graph(
            json_file_path="output/project_plan_with_tasks.json",
            output_png_path="output/project_plan_with_tasks.png",
        )
        print('ok')