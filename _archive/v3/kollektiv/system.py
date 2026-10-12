from tqdm import tqdm

from .llm import LLMClient, Message, UserMessage, SystemMessage
from .llm import Judge, EvaluationResult
from .llm import WebClient, Storage

from .utils import save_pydantic_json, load_pydantic_json, generate_project_plan_graph

from .models.models_phase2_phases import Project
from .models.models_phase3_deliverables import ProjectWithDeliverables
from .models.models_phase4_deliverable_tasks import ProjectWithTasks, TaskList
from .models.models_phase5_perform import ResultEvaluation


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

FILE_RESEARCH = "research.txt"
FILE_PROJECT_STRUCTURE = "project_structure.json"
FILE_PROJECT_PLAN = "project_plan.json"
FILE_PROJECT_PLAN_WITH_TASKS = "project_plan_with_tasks.json"


class System:

    def __init__(self, goal: str):
        self.goal = goal
        self.llm = LLMClient(model_name="qwen3:32b")
        self.judge = Judge(LLMClient(model_name="qwen3:32b"))

    def run(self):
        debug = False
        self.llm.debug = debug
        self.llm.context_window_dynamic = True

        history_base: list[Message] = [
            SystemMessage(ASSISTANT_PRIMING).print(not debug),
            UserMessage(f"This is my goal:\n{self.goal}").print(not debug),
        ]

        self.run_phase1_research(debug, history_base)
        self.run_phase2_phases(debug, history_base)
        self.run_phase3_deliverables(debug, history_base)
        self.run_phase4_tasks(debug, history_base)
        self.generate_phase4_graph()
        self.run_phase5_perform(debug, history_base)

    def run_phase1_research(self, debug, history):
        if FILE_RESEARCH in Storage.list_files():
            print(
                f"[DEBUG 1] File {FILE_RESEARCH} already exists. Skipping research phase."
            )
            return

        response, _ = self.llm.chat(
            message=(
                "Your task in this step is to figure out in principle how one tackles a project like this.\n"
                "You will be guided through the following steps:\n"
                "1. Search for helpful resources on how to break the problem down into phases.\n"
                "2. Browse one of those resources to get in-depth information.\n"
                "3. Browse a second of those resources to diversify the information.\n"
                "4. Finally, respond with your reflections and the overall summary. "
                "Include all information that you think is relevant to the project. "
                "Assume the person executing the task might not have the tools available to research on their own.\n"
                "IMPORTANT: Focus on the 'how to structure' part, and not on specific details already.\n"
                "Also note, you MUST use the tool provided!"
            ),
            history=history,
            tools=[
                WebClient.web_search,
                WebClient.web_browse,
            ],
            tools_forced_sequence=True,
        )
        Storage.write_file(FILE_RESEARCH, response.strip())

    def run_phase2_phases(self, debug, history):
        if FILE_PROJECT_STRUCTURE in Storage.list_files():
            print(
                f"[DEBUG 2] File {FILE_PROJECT_STRUCTURE} already exists. Skipping project structure phase."
            )
            return

        history = history.copy()
        history.append(Storage.read_file(FILE_RESEARCH).print(not debug))

        project, history = self.llm.chat_reflect_improve(
            message=(
                "In the previous step you successfully figured out in principle how one tackles a project like this.\n"
                "Your task now is to do the following:\n"
                "1. Reflect on what suitable project phases are to achieve the goal.\n"
                "2. Finally, respond in the requested format and create the Project hierarchy."
            ),
            history=history,
            format=Project,
            judge=self.judge,
        )

        save_pydantic_json(project, FILE_PROJECT_STRUCTURE)

    def run_phase3_deliverables(self, debug, history):
        if FILE_PROJECT_PLAN in Storage.list_files():
            print(
                f"[DEBUG 3] File {FILE_PROJECT_PLAN} already exists. Skipping project plan phase."
            )
            return

        history = history.copy()
        history.extend(
            [
                Storage.read_file(FILE_RESEARCH).print(not debug),
                Storage.read_file(FILE_PROJECT_STRUCTURE).print(not debug),
            ]
        )

        plan, _ = self.llm.chat_reflect_improve(
            message=(
                "In the previous step you successfully created a project structure with phases of how to tackle the project.\n"
                "\n"
                "Your task now is to do the following:\n"
                "1. Think through each phase and note down the following:\n"
                "   - What files are expected to be produced as part of this phase?\n"
                "   - What files are required as input for this phase?\n"
                "2. Finally, respond in the requested format and create the ProjectPlan.\n"
                "\n"
                "Note: The already existing files (research.txt, project_structure.json) are also valid input files "
                "and must also be considered as input files if required.\n"
                "\n"
                "IMPORTANT: Choose the file names carefully. "
                "The file names should be descriptive and indicate the content of the file. Avoid re-using file names. "
                "For example, if in the first phase you create a file called `foo.txt`, "
                "do not use the same name in the second phase to modify or overwrite its contents, "
                "rather consider a suffix like `foo_draft.txt`, `foo_edited.txt`, or `foo_final.txt` etc."
            ),
            history=history,
            format=ProjectWithDeliverables,
            judge=self.judge,
        )
        save_pydantic_json(plan, FILE_PROJECT_PLAN)

    def run_phase4_tasks(self, debug, history):
        if FILE_PROJECT_PLAN_WITH_TASKS in Storage.list_files():
            print(
                f"[DEBUG 4] File {FILE_PROJECT_PLAN_WITH_TASKS} already exists. Skipping project plan with tasks phase."
            )
            return

        history = history.copy()
        history.extend(
            [
                Storage.read_file(FILE_RESEARCH).print(not debug),
                Storage.read_file(FILE_PROJECT_PLAN).print(not debug),
            ]
        )

        plan = load_pydantic_json(FILE_PROJECT_PLAN, ProjectWithDeliverables)
        plan = ProjectWithTasks.from_plan(plan)
        for phase in plan.project_phases:
            taskListM, _ = self.llm.chat_reflect_improve(
                message=(
                    f"In the previous steps you successfully created a project plan with phases and deliverables.\n"
                    f"We will go through each phase individually now upon my instruction to analyze the todos.\n"
                    "\n"
                    f"For this iteration, you are working on the phase '{phase.phase_name}'.\n"
                    f"```json\n{phase.model_dump_json(indent=2)}\n```\n"
                    "\n"
                    f"Your task now is to do the following:\n"
                    f"1. Think about what steps need to be performed to complete the phase '{phase.phase_name}'.\n"
                    f"2. Given the available input files of this phase overall, which of these files are required for the individual steps?\n"
                    f"3. Given the expected output files of this phase overall, which of these files are produced by the individual steps?\n"
                    f"4. Finally, respond in the requested format to populate the TaskList.\n"
                    "\n"
                    "Note: The agent performing these tasks are stateless, if you a step produces information that is required for a later step, "
                    "you need to include this information in the output of the step by writing it to that file.\n"
                    "Only files that are an expected deliverable of this phase can be produced in this phase. "
                    "However, the goal is to create atomic actions that are as small as possible, "
                    "therefore you may produce intermediate files that are not part of the expected deliverables. "
                    "These intermediate files are only required for the next step and will be deleted after the phase is over. "
                    "For example, you may create a TMP_foo_draft.txt file that is only required for the next step to produce the final foo.txt file.\n"
                    "Split the tasks into such small atomic actions that each task produces exactly one file. "
                    "Each task may have multiple required input files though.\n"
                    "\n"
                    "IMPORTANT: Before you respond, verify that: The tasks include all required input files, otherwise the information is not available and overall coherence is lost."
                ),
                history=history,
                format=TaskList,
                judge=self.judge,
            )
            phase.tasks = taskListM.tasks
            save_pydantic_json(plan, FILE_PROJECT_PLAN_WITH_TASKS)

    def generate_phase4_graph(self):
        output_filename = FILE_PROJECT_PLAN_WITH_TASKS + ".png"
        if output_filename in Storage.list_files():
            print(
                f"[DEBUG 4] File {output_filename} already exists. Skipping graph generation."
            )
            return

        import os

        generate_project_plan_graph(
            json_file_path=os.path.join(
                Storage.directory, FILE_PROJECT_PLAN_WITH_TASKS
            ),
            output_png_path=os.path.join(Storage.directory, output_filename),
        )

    def run_phase5_perform(self, debug, history: list[Message]):
        plan: ProjectWithTasks = load_pydantic_json(
            FILE_PROJECT_PLAN_WITH_TASKS, ProjectWithTasks
        )

        history = history.copy()
        history.append(Storage.read_file(FILE_PROJECT_PLAN).print(not debug))

        tasks_completed = []
        for phase in tqdm(plan.project_phases, desc="Phase"):
            for task in tqdm(phase.tasks, desc=f"Task in phase '{phase.phase_name}'"):
                # for recovery
                files = Storage.list_files()
                if task.deliverable_file.file_name in files:
                    tasks_completed.append(task)
                    print(
                        f"[DEBUG 5] File {task.deliverable_file.file_name} already exists. Skipping task '{task.task_name}'."
                    )
                    continue

                history_ = history.copy()
                if tasks_completed:
                    texts = "\n".join(
                        [
                            f"- ✓ {t.task_name} ({t.deliverable_file.file_name})"
                            for t in tasks_completed
                        ]
                    )
                    history_.append(
                        SystemMessage(
                            f"You previously completed the following tasks:\n{texts}"
                        ).print(not debug)
                    )

                if task.required_inputs:
                    history_.append(
                        SystemMessage(
                            "You will require the following files for your next task:"
                        ).print(not debug)
                    )
                    for input_file in task.required_inputs:
                        history_.append(Storage.read_file(input_file).print(not debug))

                while True:
                    resultEvalM, history_ = self.llm.chat(
                        message=(
                            "You are currently in the phase:\n"
                            f"- Phase Name: {phase.phase_name}\n"
                            f"- Phase Description: {phase.description}\n"
                            "\n"
                            "You are currently working on the task:\n"
                            f"- Task Name: {task.task_name}\n"
                            f"- Task Description: {task.description}\n"
                            "\n"
                            "You are required to produce the following file:\n"
                            f"- Deliverable File Name: {task.deliverable_file.file_name}\n"
                            f"- Deliverable File Description: {task.deliverable_file.description}\n"
                            "\n"
                            "Your task now is to:\n"
                            f"1. Think about what information must be included in the file '{task.deliverable_file.file_name}'.\n"
                            f"2. Write the content of the file '{task.deliverable_file.file_name}' in the requested format using the tool API.\n"
                            f"3. Judge whether you need to overwrite your output or if you want to continue with the next task.\n"
                            "\n"
                            "Note: On rare occasions, if you deem it necessary, you may also overwrite the content of an existing file. "
                            "However, this should be avoided if possible and is only a valid strategy if new insights came to light "
                            "that could not have been anticipated before and require a retroactive change of certain project artifacts "
                            "to guarantee overall coherence, completeness, correctness, consistency and compatibility.\n"
                            "The system will only move to the next task once the requested file is produced. "
                        ),
                        history=history_,
                        tools=[
                            Storage.read_file,
                            Storage.write_file,
                            Storage.count_words,
                        ],
                        format=ResultEvaluation,
                    )
                    files = Storage.list_files()
                    if task.deliverable_file.file_name in files:
                        if resultEvalM.continue_with_next_task:
                            break
                        continue

                    if resultEvalM.continue_with_next_task:
                        history_.append(
                            SystemMessage(
                                f"Task '{task.task_name}' is not completed yet. "
                                f"Please make sure to produce the required file '{task.deliverable_file.file_name}' before continuing."
                            ).print(not debug)
                        )

                tasks_completed.append(task)
