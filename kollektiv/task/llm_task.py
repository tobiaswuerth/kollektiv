import logging
from typing import List
from dataclasses import dataclass

from kollektiv.core import Message, SystemMessage

from .step import Step, Link
from .result import Result

TEMPLATE_SYSTEM = """
# ROLE of you, the assistant:
{role}

# Process:
You will be guided through a multi-step process.
You might only recollect some of the previous steps, but this is the current state of the process:
{process_overview}

You are only responsible for the current step:
- [~] {process_step}

# Description of the current step:
{description}

{next_instructions}
""".strip()


class TaskClient:
    logger = logging.getLogger(__name__)

    def __init__(self, role: str, steps: List[Step]) -> None:
        self.model = "qwen3:32b"
        self.role: str = role.strip()
        self.links: List[Link] = Link.from_steps(steps)

    def _build_system_prompt(self, i: int, current_link: Link) -> str:
        process_overview = ""
        for n, link in enumerate(self.links):
            step = link.step
            if n < i:
                process_overview += f"- [✓] {n+1}. {step.name}\n"
            elif n == i:
                process_overview += f"- [~] {n+1}. {step.name}\n"
                process_step = f"{n+1}. {step.name}"
            else:
                process_overview += f"- [ ] {n+1}. {step.name}\n"

        return TEMPLATE_SYSTEM.format(
            role=self.role,
            process_overview=process_overview.strip(),
            process_step=process_step,
            description=current_link.step.description.strip(),
            next_instructions=(
                current_link.next.step.instructions().strip()
                if current_link.next
                else ""
            ),
        ).strip()

    def run(self, input: Message):
        history = [input]
        return self.run_step(self.links[0], history)

    def run_step(self, link: Link, history: List[Message]) -> List[Message]:
        model_input = history.copy()
        system_prompt = self._build_system_prompt(link.i, link)
        model_input.insert(0, SystemMessage(system_prompt))

        while True:
            try:
                result = link.step.execute(model_input[-1], model_input)
                link.step.on_after(result)

                if not link.next:
                    history.append(result)
                    return history

                # validate with next node
                val: Result = link.next.step.on_validate_request(result)
                if val.ok:
                    history.append(result)
                    return self.run_step(link.next, history)

                # if not validated ok, run previous step again
                model_input.append(val.content)
                continue

            except Exception as e:
                if not link.prev:
                    raise e

                # go back, try last step again
                prev_input = history.copy()
                prev_input.append(SystemMessage(f"Error: {e}"))
                new_history = link.prev.step.execute(prev_input[-1], prev_input)
                history[-1] = new_history[-1]
                model_input[-1] = new_history[-1]
                continue

        assert False, "Should not reach here"
