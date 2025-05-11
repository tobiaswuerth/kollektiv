import os
import ollama
import pydantic
import random
import logging
from typing import List, Callable, Tuple, Optional


from kollektiv.core import Message, UserMessage, SystemMessage, AssistantMessage
from kollektiv.handler import Handler, ToolHandler, FormatHandler
from kollektiv.roles import Judge, EvaluationResult


def _clean_thinking(response: str) -> str:
    if response.startswith("<think>"):
        response = response.split("</think>")[-1].strip()
    return response


class LLMClient:
    logger = logging.getLogger(__name__)
    
    def __init__(self, model: str = "mistral-nemo:latest", **chat_kwargs) -> None:
        self.model = model
        self.chat_kwargs = chat_kwargs
        self.context_window = 2048
        self.context_window_dynamic = False
        self.debug = False
        self.logger.info(f"LLMClient initialized with model: {model}...")
        self.logger.debug(f"Chat kwargs: {chat_kwargs}")

    def _get_response(self, messages: list[Message], verbose: bool) -> str:
        self.logger.info(f"Getting response from LLM model: {self.model}...")
        
        if self.debug:
            os.system("cls" if os.name == "nt" else "clear")
            for message in messages:
                message.print()

        total_word_count = sum(len(m.content.split()) for m in messages)
        context_windows = (
            self.context_window
            if not self.context_window_dynamic
            else int(total_word_count * 1.5) + 2048 # input + expected output
        )
        self.logger.debug(f"Input word count: {total_word_count} / Context window: {context_windows}")

        try:
            response = ollama.chat(
                self.model,
                messages=[m.__dict__ for m in messages],
                stream=verbose,
                options={
                    "max_tokens": 2048,
                    "num_ctx": context_windows,
                    "temperature": 0.5,
                    "top_p": 0.9,
                    "seed": random.randint(0, 2**30 - 1),
                },
                **self.chat_kwargs,
            )

            if not verbose:
                response_text = _clean_thinking(response.message.content)
                self.logger.debug(f"LLM response: {response_text}...")
                return AssistantMessage(response_text)

            AssistantMessage("")._print_title()
            chunks = []
            for chunk in response:
                chunk = chunk.message.content
                print(chunk, end="", flush=True)
                chunks.append(chunk)
            print()
            response_text = "".join(chunks)
            response_text = _clean_thinking(response_text)
            self.logger.debug(f"Streamed LLM response: {response_text}...")

            return AssistantMessage(response_text)
        except Exception as e:
            self.logger.error(f"Error getting response from LLM: {str(e)}...", exc_info=True)
            raise

    def _force_handler(
        self, history: List[Message], handler: Handler, verbose: bool
    ) -> Tuple[Message, List[Message]]:
        self.logger.info(f"Forcing handler: {handler.__class__.__name__}...")
        model_input = history.copy()

        while True:
            ai_message = self._get_response(model_input, verbose)
            self.logger.debug(f"Handler input: {ai_message.content}...")
            ok, response = handler.invoke(ai_message.content)
            
            if not ok:
                self.logger.warning(f"Handler {handler.__class__.__name__} failed, retrying...")
                model_input.append(ai_message)
                model_input.append(response.print(verbose))
                continue

            self.logger.info(f"Handler {handler.__class__.__name__} succeeded")
            history.append(ai_message)
            if issubclass(response.__class__, Message):
                history.append(response.print(verbose))
            return response, history

    def chat(
        self,
        message: str,
        history: List[Message] = [],
        format: Optional[pydantic.BaseModel] = None,
        verbose: bool = True,
        tools: Optional[List[Callable]] = None,
        tools_forced_sequence: bool = False,
    ) -> Tuple[Message, List[Message]]:
        self.logger.info(f"Starting chat with message: {message}...")
        
        history = history.copy()
        history.append(UserMessage(message).print(verbose))

        if not tools and not format:
            self.logger.info("Simple chat - no tools or format specified...")
            ai_message = self._get_response(history, verbose)
            history.append(ai_message)
            return ai_message.content, history

        # add instructions to the system message
        self.logger.debug(f"Format: {format.__class__.__name__ if format else None}, Tools: {[t.__name__ for t in tools] if tools else None}")
        if not issubclass(history[0].__class__, SystemMessage):
            self.logger.debug("Adding system message to history...")
            history.insert(0, SystemMessage())
        sm = history[0]
        
        if format:
            h_format = FormatHandler(format)
            if "# Format" not in sm.content:
                self.logger.debug(f"Adding format instructions for {format.__class__.__name__}...")
                sm.content = h_format.instructions + "\n\n" + sm.content
        if tools:
            h_tools = ToolHandler(tools)
            if "# Tools" not in sm.content:
                self.logger.debug(f"Adding tool instructions for {len(tools)} tools...")
                sm.content = h_tools.instructions + "\n\n" + sm.content
            if tools_forced_sequence:
                sm.content = sm.content.replace(
                    "\n</tool_call>", (
                        "\n</tool_call>\n\n"
                        f"You are FORCED to use the tools in EXACTLY this order: "
                        f"[ {', '.join([t.__name__ for t in tools])} ]"
                    )
                )
        sm.print(verbose)

        # execute
        if tools_forced_sequence:
            self.logger.info("Using forced tool sequence...")
            for tool in tools:
                self.logger.debug(f"Forcing tool: {tool.__name__}...")
                handler = ToolHandler([tool])
                response, history = self._force_handler(history, handler, verbose)

            if format:
                self.logger.debug(f"Forcing format: {format.__class__.__name__}...")
                response, history = self._force_handler(history, h_format, verbose)
            elif tools:
                # if tools were used but no format is provided, we need to invoke the LLm once more to get the final response
                self.logger.debug("Getting final response after tools...")
                ai_message = self._get_response(history, verbose)
                history.append(ai_message)
                response = ai_message.content

            self.logger.info("Chat with forced sequence completed")
            return response, history

        # tools may be used in any order, format is the last step
        model_input = history.copy()

        while True:
            ai_message = self._get_response(model_input, verbose)
            model_input.append(ai_message)

            if tools and h_tools.consider(ai_message.content):
                ok, response = h_tools.invoke(ai_message.content)
                model_input.append(response.print(verbose))
                if not ok:
                    continue
                history.append(ai_message)
                history.append(response)
                continue
            
            if format:
                ok, response = h_format.invoke(ai_message.content)
                if not ok:
                    model_input.append(response.print(verbose))
                    continue
                history.append(ai_message)
                return response, history

            history.append(ai_message)
            return ai_message.content, history

    def chat_reflect_improve(
        self,
        judge: Judge,
        message: str,
        history: List[Message] = [],
        format: Optional[pydantic.BaseModel] = None,
        verbose: bool = True,
        tools: Optional[List[Callable]] = None,
        tools_forced_sequence: bool = False,
        iterations: int = 2,
    ) -> Tuple[Message, List[Message]]:
        self.logger.info(f"Starting reflection-based chat with message: {message}...")
        self.logger.debug(f"Using judge: {judge.__class__.__name__}, iterations: {iterations}")

        result, history = self.chat(
            message=message,
            history=history,
            format=format,
            verbose=verbose,
            tools=tools,
            tools_forced_sequence=tools_forced_sequence,
        )

        for i in range(iterations):
            self.logger.info(f"Starting iteration {i + 1} of {iterations}...")
            print(f"[DEBUG] Iteration {i + 1} of {iterations}")
            inputs_ = "\n".join([h._get_printable() for h in history])
            
            try:
                result: EvaluationResult = judge.evaluate(inputs_)
                self.logger.debug(f"Evaluation result: {result}")
                
                history.append(
                    SystemMessage(
                        (
                            f"Your response has been evaluated:\n\n"
                            f"{result.model_dump_json(indent=2)}"
                        )
                    ).print(verbose)
                )

                result, history = self.chat(
                    message=(
                        "Please reflect on the evaluation and improve your answer accordingly."
                    ),
                    history=history,
                    format=format,
                    verbose=verbose,
                    tools=tools,
                    tools_forced_sequence=tools_forced_sequence,
                )
                self.logger.info(f"Completed iteration {i + 1}")
            except Exception as e:
                self.logger.error(f"Error during reflection iteration {i + 1}: {str(e)}...", exc_info=True)
                break

        self.logger.info("Reflection-based chat completed")
        return result, history
