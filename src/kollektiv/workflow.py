"""Dynamic agents on a fixed graph: discuss, plan, execute, review, reflect, revise."""

from collections.abc import Callable
from typing import Any

from langchain_core.messages import (
    HumanMessage,
    ToolMessage,
    messages_from_dict,
    messages_to_dict,
)
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Send

from .config import Settings
from .control import RunControl
from .llm import Models, json_text
from .models import (
    Contribution,
    DiscussionOutput,
    DiscussionState,
    GoalReview,
    GraphState,
    Plan,
    StaffingState,
    Task,
    TaskReview,
    WorkerOutput,
    WorkerState,
)
from .tools import Toolkit, record, visible_messages
from .workspace import Workspace


class Workflow:
    def __init__(
        self,
        settings: Settings,
        workspace: Workspace,
        emit: Callable[..., dict],
        models: Any = None,
        control: RunControl | None = None,
        delegation=None,
        child_control: RunControl | None = None,
    ):
        self.settings = settings
        self.workspace = workspace
        self.emit = emit
        self.models = models or Models(settings, emit)
        self.control = control
        self.delegation = delegation
        self.child_control = child_control

    def model_call(self, function, *args):
        if self.delegation is None:
            return function(*args)
        with self.delegation.model_gate:
            if self.child_control and self.child_control.snapshot()["stopped"]:
                from .control import RunStopped

                raise RunStopped("Sub-agent retired at a model boundary")
            return function(*args)

    def toolkit(self, agent, output=None, *, task_id=None, input_files=None, write_key=None):
        return Toolkit(
            self.settings,
            self.workspace,
            agent,
            output,
            self.emit,
            input_files=input_files,
            delegation=self.delegation,
            task_id=task_id,
            write_key=write_key,
        )

    def controlled(self, node: str, function: Callable) -> Callable:
        def invoke(state: dict) -> dict:
            task = state.get("task", {})
            fields = {"task": task.get("id"), "agent": state.get("agent_name") or task.get("owner")}
            if self.delegation and fields["agent"]:
                self.delegation.check_active(fields["agent"])
            cancel = (
                (lambda: self.child_control.snapshot()["stopped"]) if self.child_control else None
            )
            token = self.control.enter(node, cancel=cancel, **fields) if self.control else None
            self.emit("node_started", node=node, **fields)
            try:
                result = function(state)
                if self.delegation and fields["agent"]:
                    self.delegation.check_active(fields["agent"])
                self.emit("node_finished", node=node, **fields)
                return result
            except BaseException as exc:
                self.emit("node_failed", node=node, error=str(exc), **fields)
                raise
            finally:
                if self.control and token:
                    self.control.leave(token)

        return invoke

    def event(self, kind: str, **fields) -> dict:
        event = self.emit(kind, **fields)
        return {event["id"]: event}

    def team_context(self, state: dict, agent: str) -> str:
        roster = [a.model_dump(exclude_none=True) for a in self.settings.team]
        context = {
            "team": roster,
            "shared_memory": list(state.get("notes", {}).values())[-20:],
            "your_inbox": visible_messages(
                {
                    **state.get("messages", {}),
                    **(self.delegation.messages_for(agent) if self.delegation else {}),
                },
                agent,
            ),
            "allowed_message_recipients": self.settings.allowed_recipients(agent),
            "team_findings": list(state.get("contributions", {}).values())[-len(roster) :],
        }
        if self.delegation:
            context["subagents"] = self.delegation.context(agent)
            _, level = self.delegation.actor(agent)
            family = self.delegation.allowed_recipients(agent)
            context["allowed_message_recipients"] = (
                family
                if level > 1
                else list(dict.fromkeys([*context["allowed_message_recipients"], *family]))
            )
        return json_text(context)

    def discussion(self, state: DiscussionState) -> dict:
        name = state["agent_name"]
        recipients = self.settings.allowed_recipients(name)
        message_instruction = (
            "Send at least one useful message using an allowed_message_recipients entry. "
            if recipients
            else "No outgoing message channel is available. Return an empty messages list. "
        )
        prompt = (
            f"Shared goal:\n{state['goal']}\nDiscussion round {state['discussion_round']}.\n"
            f"{self.team_context(state, name)}\n"
            "Contribute from your own expertise. Discuss a concrete approach, useful conventions, "
            "risks to completion and how teammates should collaborate. Respond to your inbox. "
            f"{message_instruction}Keep findings focused, about 150 words. "
            "If no web tools are used, label outside facts as assumptions; never invent sources."
        )
        toolkit = self.toolkit(name, task_id=f"discussion:{state['discussion_round']}:{name}")
        research = []
        if self.settings.web_enabled and "web_search" in self.settings.agent(name).tools:
            tools = toolkit.build()
            history = [
                self.models.system(name),
                HumanMessage(
                    content=(
                        prompt
                        + "\nResearch relevant facts using your tools before offering recommendations."
                    )
                ),
            ]
            for _ in range(min(4, self.settings.max_steps)):
                reply = self.model_call(self.models.act, name, history, tools)
                history.append(reply)
                if not reply.tool_calls:
                    break
                response = ToolNode(tools, handle_tool_errors=True).invoke({"messages": [reply]})
                history.extend(response["messages"])
                research.extend(m.content for m in response["messages"])
            prompt += "\nResearch observations:\n" + "\n".join(research)[-24000:]

        def validate(contribution: Contribution) -> None:
            for draft in contribution.messages:
                self.settings.message_recipients(name, draft.recipient)
            if recipients and not contribution.messages:
                raise ValueError("Include at least one actionable message to a teammate")

        result = self.model_call(self.models.structured, name, Contribution, prompt, validate)
        messages = dict(toolkit.messages)
        for draft in result.messages:
            for recipient in self.settings.message_recipients(name, draft.recipient):
                message = record(name, draft.content, recipient=recipient)
                messages[message["id"]] = message
                self.emit("message", **message)
        key = f"{state['discussion_round']}:{name}"
        note = record(name, result.findings, round=state["discussion_round"])
        return {
            "contributions": {
                key: {
                    "agent": name,
                    "round": state["discussion_round"],
                    "findings": result.findings,
                }
            },
            "messages": messages,
            "notes": {**toolkit.notes, key: note},
            "events": self.event("discussion", agent=name, round=state["discussion_round"]),
        }

    def send_discussion(self, state: GraphState) -> list[Send] | str:
        if state["discussion_round"] > self.settings.discussion_rounds:
            return "plan"
        return [
            Send(
                "discuss",
                {
                    "goal": state["goal"],
                    "run_id": state["run_id"],
                    "agent_name": a.name,
                    "discussion_round": state["discussion_round"],
                    "contributions": state.get("contributions", {}),
                    "messages": state.get("messages", {}),
                    "notes": state.get("notes", {}),
                },
            )
            for a in self.settings.team
        ]

    def validate_plan(self, plan: Plan) -> None:
        if len(plan.tasks) > self.settings.max_tasks:
            raise ValueError(f"Plan exceeds max_tasks={self.settings.max_tasks}")
        owners = set(self.settings.task_owners())
        if any(t.owner not in owners for t in plan.tasks):
            raise ValueError(f"Task owners must be one of the workers: {sorted(owners)}")
        if any(
            t.output.casefold() == "delegates" or t.output.casefold().startswith("delegates/")
            for t in plan.tasks
        ):
            raise ValueError("The delegates/ namespace is reserved for dynamically hired agents")

    def plan(self, state: GraphState) -> dict:
        prompt = (
            f"USER GOAL (preserve every requirement):\n{state['goal']}\n"
            f"TEAM DISCUSSION:\n{self.team_context(state, self.settings.coordinator)}\n"
            "Create the complete project plan with phases, deliverables and atomic tasks. "
            f"Use at most {self.settings.max_tasks} tasks; avoid unnecessary intermediate files. "
            "Each task produces exactly one file and has measurable acceptance criteria. "
            "Review, approval, and final validation happen automatically in the workflow; "
            "DO NOT create tasks for reviews, approvals, signals or final validation reports. "
            "Plan actual user deliverables and only indispensable intermediate documents. "
            "Use ordinary file extensions such as .md, .txt, .py, .html, .js, .json. "
            "Assign tasks to eligible authors or publishers with write_file or assemble_files access. "
            "An orchestrator with assemble_files can own integration tasks, hire specialist authors, "
            "and publish their approved direct-child artifacts into each assigned output. "
            "Coordinator and reviewer may be the same configured agent; do not invent extra team members. "
            "Use unique short task IDs and unique relative output paths. "
            "Preserve exact filenames requested by the user. "
            "Dependencies must contain task IDs, not filenames, and form a DAG. "
            "Include dependencies for any files a worker must read, especially to maintain consistency. "
            "The complete task list must cover the user's entire goal. "
            "A coherent story needs a shared outline followed by chapters using it. "
            "A software project needs actual runnable source files and usage instructions."
            "\nFINAL PLANNING RULE: List file-production tasks only. Review and approval are "
            "already performed automatically, so never assign review-only tasks. For example, "
            "a goal requesting design.md and story.md needs a design.md writing task and a "
            "story.md writing task depending on it. Additional planning documents are useful "
            "only when their contents are indispensable for a complex deliverable."
        )
        plan = self.model_call(
            self.models.structured, self.settings.coordinator, Plan, prompt, self.validate_plan
        )
        tasks = {
            t.id: {**t.model_dump(), "status": "pending", "attempts": 0, "feedback": ""}
            for t in plan.tasks
        }
        return {
            "plan": plan.model_dump(),
            "tasks": tasks,
            "status": "working",
            "events": self.event("plan_created", tasks=len(tasks)),
        }

    def worker_prompt(self, state: WorkerState) -> str:
        task = state["task"]
        scope = (
            f"Your assigned output is exactly {task['output']}. Complete this task only. "
            "Other agents handle other deliverables; an intermediate document must satisfy "
            "its own task criteria rather than contain the final project output.\n"
            if task.get("output")
            else "You have a read-only assignment and no assigned file. Use your permitted tools "
            "to complete the assignment and finish with your complete findings as a text response.\n"
        )
        writing = (
            "Read the dependency files listed in the task before writing. "
            "Use only your available tools. If write_file is available, use it to produce the "
            "complete assigned artifact, including all content. "
            "write_file takes only content; its output path is already bound to your task. "
            "If assemble_files is available, use it to join approved dependency files or verified "
            "completed direct-child outputs in the required order. An orchestrator without "
            "write_file hires authors, reads their actual outputs, and publishes them with "
            "assemble_files. It writes your "
            "assigned output directly without making you repeat the entire text. "
            "Do not replace a successfully assembled artifact with a shortened write_file call. "
            "Read any existing version before making corrections. Use check_file when helpful. "
            "You can only write your assigned output. After writing it, finish with a brief summary. "
            if task.get("output")
            else "Read actual inputs when relevant. Report evidence and uncertainty clearly. "
        )
        return (
            f"YOUR TASK:\n{json_text(task)}\n"
            f"{scope}"
            f"PROJECT GOAL FOR CONTEXT:\n{state['goal']}\nPLAN:\n{state['plan_summary']}\n"
            f"{self.team_context(state, task['owner'])}\n"
            f"REVIEW FEEDBACK:\n{state.get('feedback') or 'First attempt.'}\n"
            f"{writing}"
            "Send useful decisions or questions to teammates and save reusable facts with share_note. "
            "If hiring is available, specify the exact child name, role, instructions, skills and "
            "tools plus its assignment. Hires have their own output under delegates/name/ or can "
            "return text findings with output omitted. Status changes arrive automatically in your "
            "inbox; monitor_agents shows current progress, instruct_agent redirects a direct child, "
            "and fire retires it while preserving completed work. Level-3 agents cannot hire or fire. "
            "Wait for actual results and inspect them before integration; never claim a hire's "
            "work is done just because it was launched. Retire finished helpers when no longer needed. "
            "The reviewer will inspect the actual result against every acceptance criterion."
        )

    @staticmethod
    def final_response(state: WorkerState) -> str:
        if not state.get("reply"):
            return ""
        reply = messages_from_dict([state["reply"]])[0]
        return reply.content if isinstance(reply.content, str) else json_text(reply.content)

    def act(self, state: WorkerState) -> dict:
        task = state["task"]
        revision = self.delegation.snapshot()["revision"] if self.delegation else 0
        observed_children = self.child_statuses(task["owner"])
        toolkit = self.toolkit(
            task["owner"],
            task["output"],
            task_id=task["id"],
            input_files=task.get("input_files", []),
        )
        tools = toolkit.build()
        history = messages_from_dict(state.get("history", []))
        prompt = self.worker_prompt(state)
        messages = [self.models.system(task["owner"]), HumanMessage(content=prompt), *history]
        reply = self.model_call(self.models.act, task["owner"], messages, tools)
        return {
            "history": messages_to_dict([reply]),
            "reply": messages_to_dict([reply])[0],
            "steps": state.get("steps", 0) + 1,
            "finished": not (reply.tool_calls or reply.invalid_tool_calls),
            "delegation_revision": revision,
            "observed_children": observed_children,
            "events": self.event(
                "worker_step",
                agent=task["owner"],
                task=task["id"],
                attempt=state["attempt"],
                step=state.get("steps", 0) + 1,
            ),
        }

    def after_act(self, state: WorkerState) -> str:
        if state["finished"]:
            owner = state["task"]["owner"]
            changed = state.get("observed_children", {}) != self.child_statuses(owner)
            if self.has_children_working(owner) or (
                changed and state["steps"] < self.settings.max_steps
            ):
                return "await_children"
            return "review"
        return "tools"

    def child_statuses(self, agent: str) -> dict[str, str]:
        if not self.delegation:
            return {}
        return {
            child["name"]: child["status"]
            for child in self.delegation.snapshot()["agents"].values()
            if child["parent"] == agent
        }

    def has_children_working(self, agent: str) -> bool:
        if not self.delegation:
            return False
        return any(
            child["parent"] == agent and child["status"] not in ("completed", "failed", "retired")
            for child in self.delegation.snapshot()["agents"].values()
        )

    def await_children(self, state: WorkerState) -> dict:
        owner = state["task"]["owner"]
        self.delegation.raise_if_interrupted(owner)
        self.delegation.wait_for_child_change(owner, state.get("delegation_revision", 0), 0.25)
        revision = self.delegation.snapshot()["revision"]
        return {
            "delegation_revision": revision,
            "delegation_changed": revision != state.get("delegation_revision", 0),
            "finished": False,
        }

    def after_await_children(self, state: WorkerState) -> str:
        working = self.has_children_working(state["task"]["owner"])
        if state["steps"] >= self.settings.max_steps:
            return "await_children" if working else "review"
        if working and not state.get("delegation_changed"):
            return "await_children"
        return "act"

    def staff_act(self, state: StaffingState) -> dict:
        name = state["agent_name"]
        revision = self.delegation.snapshot()["revision"]
        observed_children = self.child_statuses(name)
        toolkit = self.toolkit(name, task_id=f"discussion:{state['discussion_round']}:{name}")
        prompt = (
            f"SHARED GOAL:\n{state['goal']}\n{self.team_context(state, name)}\n"
            "Prepare your contribution. Use hire only when a focused specialist is needed, "
            "specifying its exact name, role, instructions, skills, tools and assignment. "
            "Original agents are level 1; their children are level 2; grandchildren are level 3 "
            "and cannot hire or fire. Inspect automatic status messages and monitor_agents; "
            "use instruct_agent to correct your child and fire to retire it when its work is done. "
            "Hires write separate files under delegates/name/; the original team integrates "
            "those files into planned deliverables. Do not hire just to repeat another role. "
            "There is no output assigned during this discussion. Finish with a concise recommendation."
        )
        reply = self.model_call(
            self.models.act,
            name,
            [
                self.models.system(name),
                HumanMessage(content=prompt),
                *messages_from_dict(state.get("history", [])),
            ],
            toolkit.build(),
        )
        return {
            "history": messages_to_dict([reply]),
            "reply": messages_to_dict([reply])[0],
            "steps": state.get("steps", 0) + 1,
            "finished": not (reply.tool_calls or reply.invalid_tool_calls),
            "delegation_revision": revision,
            "observed_children": observed_children,
        }

    def staff_tools(self, state: StaffingState) -> dict:
        name = state["agent_name"]
        toolkit = self.toolkit(name, task_id=f"discussion:{state['discussion_round']}:{name}")
        reply = messages_from_dict([state["reply"]])[0]
        response = ToolNode(toolkit.build(), handle_tool_errors=True).invoke({"messages": [reply]})
        for message in response["messages"]:
            self.emit(
                "tool_result",
                agent=name,
                tool=getattr(message, "name", None),
                status=getattr(message, "status", None),
                content=str(message.content)[:16000],
            )
        return {
            "history": messages_to_dict(response["messages"]),
            "messages": {**state.get("messages", {}), **toolkit.messages},
            "notes": {**state.get("notes", {}), **toolkit.notes},
        }

    def after_staff(self, state: StaffingState) -> str:
        if state.get("finished") or state.get("steps", 0) >= self.settings.max_steps:
            changed = state.get("observed_children", {}) != self.child_statuses(state["agent_name"])
            return (
                "wait"
                if self.has_children_working(state["agent_name"])
                or (changed and state.get("steps", 0) < self.settings.max_steps)
                else "contribute"
            )
        return "tools"

    def after_staff_tools(self, state: StaffingState) -> str:
        if state.get("steps", 0) >= self.settings.max_steps:
            return "wait" if self.has_children_working(state["agent_name"]) else "contribute"
        return "act"

    def staff_wait(self, state: StaffingState) -> dict:
        self.delegation.raise_if_interrupted(state["agent_name"])
        self.delegation.wait_for_child_change(
            state["agent_name"], state.get("delegation_revision", 0), 0.25
        )
        revision = self.delegation.snapshot()["revision"]
        return {
            "delegation_revision": revision,
            "delegation_changed": revision != state.get("delegation_revision", 0),
        }

    def after_staff_wait(self, state: StaffingState) -> str:
        if self.has_children_working(state["agent_name"]) and (
            not state.get("delegation_changed") or state.get("steps", 0) >= self.settings.max_steps
        ):
            return "wait"
        return "act" if state.get("steps", 0) < self.settings.max_steps else "contribute"

    def build_discussion(self):
        if not self.delegation:
            return self.controlled("discuss", self.discussion)
        graph = StateGraph(StaffingState, output_schema=DiscussionOutput)
        graph.add_node("act", self.controlled("staff.act", self.staff_act))
        graph.add_node("tools", self.controlled("staff.tools", self.staff_tools))
        graph.add_node("wait", self.controlled("staff.wait", self.staff_wait))
        graph.add_node("contribute", self.controlled("discuss", self.discussion))
        graph.add_conditional_edges(
            START,
            lambda s: (
                "act"
                if any(
                    tool in self.settings.agent(s["agent_name"]).tools
                    for tool in ("hire", "fire", "instruct_agent")
                )
                else "contribute"
            ),
            ["act", "contribute"],
        )
        graph.add_conditional_edges("act", self.after_staff, ["tools", "wait", "contribute"])
        graph.add_conditional_edges("tools", self.after_staff_tools, ["act", "wait", "contribute"])
        graph.add_conditional_edges("wait", self.after_staff_wait, ["act", "wait", "contribute"])
        graph.add_edge("contribute", END)
        return graph.compile()

    def execute(self, state: WorkerState) -> dict:
        task = state["task"]
        reply = messages_from_dict([state["reply"]])[0]
        toolkit = self.toolkit(
            task["owner"],
            task["output"],
            task_id=task["id"],
            input_files=task.get("input_files", []),
            write_key=next(
                (
                    call["id"]
                    for call in reply.tool_calls
                    if call["name"] in ("write_file", "assemble_files")
                ),
                None,
            ),
        )
        if reply.invalid_tool_calls and not reply.tool_calls:
            messages = [
                HumanMessage(
                    content=(
                        "Your tool arguments were invalid. Try again with valid JSON arguments. "
                        f"Errors: {json_text(reply.invalid_tool_calls)}"
                    )
                )
            ]
        elif sum(c["name"] in ("write_file", "assemble_files") for c in reply.tool_calls) > 1:
            messages = [
                ToolMessage(
                    content="Use only one output-writing call (write_file or assemble_files) per step.",
                    tool_call_id=c["id"],
                    name=c["name"],
                    status="error",
                )
                for c in reply.tool_calls
            ]
        else:
            response = ToolNode(toolkit.build(), handle_tool_errors=True).invoke(
                {"messages": [reply]}
            )
            messages = response["messages"]
        for message in messages:
            self.emit(
                "tool_result",
                agent=task["owner"],
                task=task["id"],
                tool=getattr(message, "name", None),
                status=getattr(message, "status", None),
                content=str(message.content)[:16000],
            )
        return {
            "history": messages_to_dict(messages),
            "messages": toolkit.messages,
            "notes": toolkit.notes,
        }

    def after_tools(self, state: WorkerState) -> str:
        if state["steps"] >= self.settings.max_steps and self.has_children_working(
            state["task"]["owner"]
        ):
            return "await_children"
        return "review" if state["steps"] >= self.settings.max_steps else "act"

    def review_task(self, state: WorkerState) -> dict:
        task = state["task"]
        if not task.get("output"):
            response = self.final_response(state)
            if not response.strip():
                review = TaskReview(
                    decision="revise",
                    feedback="Return complete findings in your final text response.",
                    reflection="Read-only workers must return their actual findings.",
                )
            else:
                if len(response) > self.settings.context_chars:
                    raise ValueError(
                        "Read-only findings exceed context_chars; increase the review context budget"
                    )
                prompt = (
                    f"TASK:\n{json_text(task)}\nACTUAL FINAL RESPONSE:\n{response}\n"
                    f"ACTUAL INPUT FILES:\n{self.workspace.context(task.get('input_files', []), self.settings.context_chars)}\n"
                    f"PROJECT GOAL FOR CONTEXT:\n{state['goal']}\n"
                    "Independently check every acceptance criterion against the actual response. "
                    "Reject incomplete findings, unsupported claims or missing requirements. "
                    "This is a read-only assignment; a separate output file is not required."
                )
                review = self.model_call(
                    self.models.structured, self.settings.reviewer, TaskReview, prompt
                )
            return self.review_result(state, review)
        errors = []
        try:
            info = self.workspace.inspect(task["output"])
            errors.extend(info["problems"])
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
        if errors:
            review = TaskReview(
                decision="revise",
                feedback="; ".join(errors),
                reflection="Verify artifact existence and syntax before reporting done.",
            )
        else:
            paths = [*task.get("input_files", []), task["output"]]
            prompt = (
                f"TASK:\n{json_text(task)}\n"
                f"Review ONLY the assigned file {task['output']} against this task's criteria. "
                "Other files are completed by other tasks. Intermediate documents do not need "
                "to be the final user deliverable or use its filename.\n"
                f"ACTUAL FILES:\n{self.workspace.context(paths, self.settings.context_chars)}\n"
                f"DETERMINISTIC CHECKS:\n{json_text(info)}\n"
                f"PROJECT GOAL FOR CONTEXT:\n{state['goal']}\n"
                + (
                    "The reviewer and author are the same configured agent: perform a fresh-context "
                    "self-review without claiming an independent reviewer. Inspect the assigned output "
                    "against EACH acceptance criterion, "
                    if task["owner"] == self.settings.reviewer
                    else "Independently inspect the assigned output against EACH acceptance criterion, "
                )
                + "its dependency inputs, and the goal. Approve only a complete, usable deliverable. "
                "Reject placeholders, missing required sections, contradictions and unmet word counts. "
                "Code syntax checks do not prove runtime correctness; do not claim code execution. "
                "If rejecting, explain concrete corrections for the owner. "
                "Record a concise lesson or design decision for future team work."
            )
            review = self.model_call(
                self.models.structured, self.settings.reviewer, TaskReview, prompt
            )
        return self.review_result(state, review)

    def review_result(self, state: WorkerState, review: TaskReview) -> dict:
        task = state["task"]
        message = record(
            self.settings.reviewer,
            review.feedback,
            recipient=task["owner"],
            task=task["id"],
            system=True,
        )
        self.emit("message", **message)
        return {
            "review": review.model_dump(),
            "feedback": review.feedback,
            "messages": {message["id"]: message},
            "events": self.event(
                "task_review",
                task=task["id"],
                agent=self.settings.reviewer,
                accepted=review.accepted,
                feedback=review.feedback,
                attempt=state["attempt"],
            ),
        }

    def after_review(self, state: WorkerState) -> str:
        if state["review"]["accepted"] or state["attempt"] >= self.settings.max_task_attempts:
            return "finish"
        return "retry"

    def retry(self, state: WorkerState) -> dict:
        # Resolve the previous tool call sequence before beginning a fresh attempt.
        return {
            "attempt": state["attempt"] + 1,
            "steps": 0,
            "finished": False,
            "history": messages_to_dict(
                [
                    HumanMessage(
                        content=(
                            "Independent review rejected the artifact. Correct it using the feedback:\n"
                            + state["feedback"]
                        )
                    )
                ]
            ),
        }

    def finish_worker(self, state: WorkerState) -> dict:
        task = state["task"]
        review = state["review"]
        artifact = None
        if task.get("output"):
            try:
                artifact = self.workspace.inspect(task["output"])
            except (OSError, ValueError):
                pass
        result = {
            "task_id": task["id"],
            "status": "completed" if review["accepted"] else "failed",
            "attempts": state["attempt"],
            "feedback": review["feedback"],
            "artifact": artifact,
            "summary": self.final_response(state),
        }
        note = record(task["owner"], review["reflection"], task=task["id"])
        message = record(
            task["owner"],
            f"{task['title']}: {result['status']}. {review['feedback']}",
            recipient="all",
            task=task["id"],
            system=True,
        )
        self.emit("message", **message)
        return {
            "results": {task["id"]: result},
            "notes": {note["id"]: note},
            "messages": {message["id"]: message},
            "events": self.event("task_finished", task=task["id"], status=result["status"]),
        }

    def build_worker(self, checkpointer=None, *, full_state=False):
        worker = StateGraph(WorkerState, output_schema=WorkerState if full_state else WorkerOutput)
        worker.add_node("act", self.controlled("worker.act", self.act))
        worker.add_node("tools", self.controlled("worker.tools", self.execute))
        worker.add_node("review", self.controlled("worker.review", self.review_task))
        worker.add_node("retry", self.controlled("worker.retry", self.retry))
        worker.add_node("finish", self.controlled("worker.finish", self.finish_worker))
        worker.add_node(
            "await_children", self.controlled("worker.await_children", self.await_children)
        )
        worker.add_edge(START, "act")
        worker.add_conditional_edges("act", self.after_act, ["tools", "review", "await_children"])
        worker.add_conditional_edges("tools", self.after_tools, ["act", "review", "await_children"])
        worker.add_conditional_edges(
            "await_children", self.after_await_children, ["act", "review", "await_children"]
        )
        worker.add_conditional_edges("review", self.after_review, ["retry", "finish"])
        worker.add_edge("retry", "act")
        worker.add_edge("finish", END)
        return worker.compile(checkpointer=checkpointer)

    def collect(self, state: GraphState) -> dict:
        tasks = {k: dict(v) for k, v in state["tasks"].items()}
        for task_id in state["ready"]:
            result = state["results"][task_id]
            tasks[task_id].update(
                status=result["status"],
                attempts=result["attempts"],
                feedback=result["feedback"],
                artifact=result["artifact"],
            )
        return {"tasks": tasks}

    def schedule(self, state: GraphState) -> dict:
        tasks = {k: dict(v) for k, v in state["tasks"].items()}
        completed = {k for k, t in tasks.items() if t["status"] == "completed"}
        ready = [
            k
            for k, t in tasks.items()
            if t["status"] == "pending" and set(t["dependencies"]) <= completed
        ]
        return {"ready": ready, "tasks": tasks}

    def send_workers(self, state: GraphState) -> list[Send] | str:
        if not state["ready"]:
            if all(t["status"] == "completed" for t in state["tasks"].values()):
                return "goal_review"
            return "finalize"
        workers = []
        for task_id in state["ready"]:
            task = dict(state["tasks"][task_id])
            task["input_files"] = [state["tasks"][d]["output"] for d in task["dependencies"]]
            workers.append(
                Send(
                    "worker",
                    {
                        "goal": state["goal"],
                        "run_id": state["run_id"],
                        "task": task,
                        "plan_summary": state["plan"]["summary"],
                        "attempt": task["attempts"] + 1,
                        "steps": 0,
                        "feedback": task["feedback"],
                        "messages": state.get("messages", {}),
                        "notes": state.get("notes", {}),
                        "history": [],
                        "events": {},
                        "results": {},
                    },
                )
            )
        return workers

    def review_goal(self, state: GraphState) -> dict:
        tasks = state["tasks"]
        # Verify that approvals still describe the actual files, including after a resume.
        changed = []
        for task_id, task in tasks.items():
            try:
                current = self.workspace.inspect(task["output"])
                if current["problems"] or current["sha256"] != task["artifact"]["sha256"]:
                    changed.append(task_id)
            except (OSError, ValueError):
                changed.append(task_id)
        if changed:
            review = GoalReview(
                decision="revise",
                feedback="Accepted artifacts changed or disappeared.",
                revise_tasks=changed,
            )
        else:
            prompt = (
                f"ORIGINAL USER GOAL:\n{state['goal']}\n"
                f"PLAN:\n{json_text(state['plan'])}\n"
                f"TASK REVIEWS:\n{json_text(tasks)}\n"
                f"DELEGATED WORK:\n{json_text(self.delegated_review_context())}\n"
                f"ACTUAL DELIVERABLES:\n{self.workspace.context([t['output'] for t in tasks.values()] + self.delegate_outputs(), self.settings.context_chars)}\n"
                "Determine whether the WHOLE user goal has been achieved. Audit each explicit "
                "requirement using these actual files, including cross-file coherence. "
                "Task approvals alone do not establish goal completion. "
                "If changes are needed, identify exact existing task IDs to revise. "
                "If required deliverables are missing from the plan, supply additional atomic tasks "
                "with unique IDs and paths, valid owners, acceptance criteria, and dependencies. "
                "Do not invent tests or code execution. Explain the evidence and practical limitations."
            )

            def validate(review: GoalReview) -> None:
                if any(k not in tasks for k in review.revise_tasks):
                    raise ValueError("revise_tasks must reference existing task IDs")
                if review.achieved and (review.revise_tasks or review.additional_tasks):
                    raise ValueError("An achieved goal cannot still require tasks")
                if review.additional_tasks:
                    merged = Plan.model_validate(
                        {
                            **state["plan"],
                            "tasks": [
                                *state["plan"]["tasks"],
                                *[t.model_dump() for t in review.additional_tasks],
                            ],
                        }
                    )
                    self.validate_plan(merged)
                if not review.achieved and not (review.revise_tasks or review.additional_tasks):
                    raise ValueError("Identify actionable task revisions or missing deliverables")

            review = self.model_call(
                self.models.structured, self.settings.reviewer, GoalReview, prompt, validate
            )
        return {
            "goal_review": review.model_dump(),
            "events": self.event("goal_review", **review.model_dump()),
        }

    def delegate_outputs(self) -> list[str]:
        if not self.delegation:
            return []
        return [
            job["output"]
            for job in self.delegation.snapshot()["agents"].values()
            if job.get("artifact")
        ]

    def delegated_review_context(self) -> list[dict]:
        if not self.delegation:
            return []
        fields = (
            "name",
            "parent",
            "level",
            "status",
            "role",
            "assignment",
            "tools",
            "output",
            "summary",
            "error",
            "artifact",
            "result",
        )
        return [
            {key: job.get(key) for key in fields}
            for job in self.delegation.snapshot()["agents"].values()
        ]

    def after_goal_review(self, state: GraphState) -> str:
        if state["goal_review"]["achieved"]:
            return "finalize"
        if state["goal_revision"] >= self.settings.max_goal_revisions:
            return "finalize"
        return "revise"

    def revise(self, state: GraphState) -> dict:
        review = state["goal_review"]
        tasks = {k: dict(v) for k, v in state["tasks"].items()}
        reopen = set(review["revise_tasks"])
        # Rebuild descendants so they cannot retain stale assumptions about changed inputs.
        while True:
            descendants = {k for k, t in tasks.items() if set(t["dependencies"]) & reopen}
            if descendants <= reopen:
                break
            reopen |= descendants
        for task_id in reopen:
            tasks[task_id].update(
                status="pending",
                attempts=0,
                feedback=review["feedback"]
                + "\nPrevious review: "
                + tasks[task_id].get("feedback", ""),
            )
        for task in review["additional_tasks"]:
            tasks[task["id"]] = {
                **task,
                "status": "pending",
                "attempts": 0,
                "feedback": review["feedback"],
            }
        plan = {
            **state["plan"],
            "tasks": [
                Task.model_validate({k: t[k] for k in Task.model_fields}).model_dump()
                for t in tasks.values()
            ],
        }
        message = record(self.settings.reviewer, review["feedback"], recipient="all", system=True)
        self.emit("message", **message)
        return {
            "tasks": tasks,
            "plan": plan,
            "goal_revision": state["goal_revision"] + 1,
            "messages": {message["id"]: message},
            "events": self.event(
                "goal_revision",
                revision=state["goal_revision"] + 1,
                reopened=sorted(reopen),
                added=len(review["additional_tasks"]),
            ),
        }

    def finalize(self, state: GraphState) -> dict:
        complete = bool(state.get("goal_review", {}).get("achieved")) and all(
            t["status"] == "completed" for t in state["tasks"].values()
        )
        if complete:
            summary = state["goal_review"]["feedback"]
        elif all(t["status"] == "completed" for t in state["tasks"].values()):
            summary = "Goal review still requires changes: " + state["goal_review"]["feedback"]
        else:
            failed = [
                f"{t['id']}: {t['feedback']}"
                for t in state["tasks"].values()
                if t["status"] == "failed"
            ]
            summary = "Task attempt limit reached. " + "; ".join(failed)
        return {
            "status": "completed" if complete else "incomplete",
            "final_summary": summary,
            "events": self.event("run_finished", status="completed" if complete else "incomplete"),
        }

    def build(self, checkpointer):
        graph = StateGraph(GraphState)
        graph.add_node(
            "begin",
            self.controlled("begin", lambda _: {"discussion_round": 1, "status": "discussing"}),
        )
        graph.add_node("discuss", self.build_discussion())
        graph.add_node(
            "discussion_join",
            self.controlled(
                "discussion_join", lambda s: {"discussion_round": s["discussion_round"] + 1}
            ),
        )
        graph.add_node("plan", self.controlled("plan", self.plan))
        graph.add_node("schedule", self.controlled("schedule", self.schedule))
        graph.add_node("worker", self.build_worker())
        graph.add_node("collect", self.controlled("collect", self.collect))
        graph.add_node("goal_review", self.controlled("goal_review", self.review_goal))
        graph.add_node("revise", self.controlled("revise", self.revise))
        graph.add_node("finalize", self.controlled("finalize", self.finalize))
        graph.add_edge(START, "begin")
        graph.add_conditional_edges("begin", self.send_discussion, ["discuss", "plan"])
        graph.add_edge("discuss", "discussion_join")
        graph.add_conditional_edges("discussion_join", self.send_discussion, ["discuss", "plan"])
        graph.add_edge("plan", "schedule")
        graph.add_conditional_edges(
            "schedule", self.send_workers, ["worker", "goal_review", "finalize"]
        )
        graph.add_edge("worker", "collect")
        graph.add_edge("collect", "schedule")
        graph.add_conditional_edges("goal_review", self.after_goal_review, ["revise", "finalize"])
        graph.add_edge("revise", "schedule")
        graph.add_edge("finalize", END)
        return graph.compile(checkpointer=checkpointer)
