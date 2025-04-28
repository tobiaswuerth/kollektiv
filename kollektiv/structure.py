from typing import Annotated, Type, Generic, TypeVar
from pydantic import BaseModel, Field
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command
from langchain_core.messages import ToolMessage
import json
from langchain_core.tools import BaseTool

class CreateRoot(BaseModel):
    node_title: str = Field(description="The title of the root node")


class CreateNode(BaseModel):
    attached_to: int = Field(description="The ID of the node to attach to")
    node_title: str = Field(description="The title of the node")
    execution_order: int = Field(description="The order in which the node is to be executed amongst its siblings")


class DeleteNode(BaseModel):
    node_id: int = Field(description="The ID of the node to delete")


class UpdateNode(BaseModel):
    node_id: int = Field(description="The ID of the node to update")
    new_title: str = Field(description="The new title of the node")


class Node(BaseModel):
    id: int = Field(default_factory=lambda self: id(self))
    title: str
    execution_order: int
    children: list["Node"] = Field(default_factory=list)

class ProblemStructurizer:
    def __init__(self):
        self.root = None

    def find_node(self, node_id: int, nodes: list[Node]) -> Node | None:
        for node in nodes:
            if node.id == node_id:
                return node
            found_node = self.find_node(node_id, node.children)
            if found_node:
                return found_node
        return None

    @property
    def structure_description(self) -> str:
        assert self.root is not None, "Root node does not exist"

        return (
            f"Tree looks now like this:\n"
            f"```json\n"
            f"{json.dumps(self.root, indent=2, default=lambda o: o.dict())}\n"
            f"```"
        )

    @property
    def create_root(self):
        parent = self

        class CreateRootTool(BaseTool):
            name:str = "create_root"
            description:str = "Create a root node for the structure"
            args_schema:Type[BaseModel] = CreateRoot

            def _run(self, **kwargs) -> str:
                return parent._create_root(CreateRoot(**kwargs))
            
        return CreateRootTool()

    def _create_root(self, node: CreateRoot) -> str:
        assert self.root is None, "Root node already exists"

        self.root = Node(
            title=node.node_title,
            execution_order=0,
        )

        return (
            f"Root node created successfully.\n"
            f"{self.structure_description}"
        )
