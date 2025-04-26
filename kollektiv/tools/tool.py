from typing import NamedTuple, Callable, get_type_hints
import inspect
import pydantic


class ResponseStatus(NamedTuple):
    code: int
    message: str

RESPONSE_OK = ResponseStatus(200, "OK")
RESPONSE_INVALID_REQUEST = ResponseStatus(400, "Invalid request")
RESPONSE_NOT_FOUND = ResponseStatus(404, "Not found")
RESPONSE_INTERNAL_ERROR = ResponseStatus(500, "Internal error")


class Function:
    def __init__(self, name: str, description: str, func: Callable[[int, pydantic.BaseModel], pydantic.BaseModel]):
        assert name
        assert description
        assert callable(func)

        sig = inspect.signature(func)
        params = list(sig.parameters.values())
        assert len(params) == 2, "func must take exactly two argument (besides self if method): [agent_id:int, input_: pydantic.BaseModel]"
        assert params[0].annotation == int, "First argument must be of type int (agent_id)"

        hints = get_type_hints(func)
        type_input = hints.get(params[1].name)
        assert type_input and issubclass(type_input, pydantic.BaseModel), "Second argument must be of type pydantic.BaseModel (input)"
        
        type_return = hints.get("return")
        assert type_return and issubclass(type_return, pydantic.BaseModel), "Return type must be of type pydantic.BaseModel (output)"

        self.name = name
        self.description = description
        self.func_TIn = type_input
        self.func_TOut = type_return
        self.func = func



class Tool:

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.functions = {}

    def register_function(self, func: Function):
        if not isinstance(func, Function):
            raise TypeError("func must be an instance of ToolFunction")

        self.functions[func.name] = func

    def get_json_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "functions": [
                {
                    "name": func.name,
                    "description": func.description,
                    "function_input": func.func_TIn.model_json_schema(),
                    "function_output": func.func_TOut.model_json_schema(),
                }
                for func in self.functions.values()
            ],
        }
