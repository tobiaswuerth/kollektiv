from typing import Type
from pydantic import BaseModel, Field
import json
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage

class CreateRoot(BaseModel):
    node_title: str = Field(description="The title of the root node")


class RowNode(BaseModel):
    node_title: str = Field(description="The title of the node")
    execution_order: int = Field(
        description="The order in which the node is to be executed amongst its siblings"
    )


class CreateNodeRow(BaseModel):
    nodes: list[RowNode] = Field(
        description=(
            "A list of nodes to be created. "
            "They will be attached to the active node."
        )
    )


class CreateNode(RowNode):
    attached_to: int = Field(description="The ID of the node to attach to")


class DeleteNode(BaseModel):
    node_id: int = Field(description="The ID of the node to delete")


class UpdateNode(BaseModel):
    node_id: int = Field(description="The ID of the node to update")
    new_title: str = Field(description="The new title of the node")
    new_execution_order: int = Field(
        description="The new order in which the node is to be executed amongst its siblings"
    )


class Node(BaseModel):
    id: int = Field(default_factory=lambda self: id(self))
    title: str
    execution_order: int
    children: list["Node"] = Field(default_factory=list)


class ProblemStructurizer:
    def __init__(self):
        self.root = None
        self.active_node = None

    def find_node(self, node_id: int, nodes: list[Node]) -> Node | None:
        for node in nodes:
            if node.id == node_id:
                return node
            found_node = self.find_node(node_id, node.children)
            if found_node:
                return found_node
        return None

    def get_activation_node(self, node_id: int):
        def _activate_node(state) -> dict:
            assert self.root is not None, "Root node does not exist"

            node = self.find_node(node_id, [self.root])
            assert node is not None, f"Node with ID {node_id} not found."

            self.active_node = node
            return {
                "messages": [
                    SystemMessage(
                        (
                            f"Node with ID {node_id} activated successfully.\n"
                            f"{self.structure_description}"
                        )
                    )
                ]
            }
        return _activate_node

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
            name: str = "create_root"
            description: str = "Create a root node for the structure"
            args_schema: Type[BaseModel] = CreateRoot

            def _run(self, **kwargs) -> str:
                node = CreateRoot(**kwargs)
                assert parent.root is None, "Root node already exists"

                parent.root = Node(
                    title=node.node_title,
                    execution_order=0,
                )
                parent.active_node = parent.root

                return (
                    f"Root node created successfully.\n"
                    f"{parent.structure_description}\n"
                    f"The root node is now the active node."
                )

        return CreateRootTool()

    @property
    def create_node_row(self):
        parent = self

        class CreateNodeRowTool(BaseTool):
            name: str = "create_node_row"
            description: str = (
                "Creates a sequence of nodes and attaches them to the active node"
            )
            args_schema: Type[BaseModel] = CreateNodeRow

            def _run(self, **kwargs) -> str:
                node = CreateNodeRow(**kwargs)
                assert parent.root is not None, "Root node does not exist"

                if parent.active_node is None:
                    return f"[ERROR] 400: No active node. Please activate one first"

                for child in node.nodes:
                    new_node = Node(
                        title=child.node_title,
                        execution_order=child.execution_order,
                    )
                    parent.active_node.children.append(new_node)

                return (
                    f"Nodes created successfully.\n" f"{parent.structure_description}"
                )

        return CreateNodeRowTool()

    @property
    def create_node(self):
        parent = self

        class CreateNodeTool(BaseTool):
            name: str = "create_node"
            description: str = "Create a child node for the structure"
            args_schema: Type[BaseModel] = CreateNode

            def _run(self, **kwargs) -> str:
                node = CreateNode(**kwargs)
                assert parent.root is not None, "Root node does not exist"

                parent_node = parent.find_node(node.attached_to, [parent.root])
                if parent_node is None:
                    return f"[ERROR] 404: Node with ID {node.attached_to} not found."

                new_node = Node(
                    title=node.node_title,
                    execution_order=node.execution_order,
                )
                parent_node.children.append(new_node)

                return f"Node created successfully.\n" f"{parent.structure_description}"

        return CreateNodeTool()

    @property
    def update_node(self):
        parent = self

        class UpdateNodeTool(BaseTool):
            name: str = "update_node"
            description: str = "Update a node in the structure"
            args_schema: Type[BaseModel] = UpdateNode

            def _run(self, **kwargs) -> str:
                node = UpdateNode(**kwargs)
                assert parent.root is not None, "Root node does not exist"

                node_to_update = parent.find_node(node.node_id, [parent.root])
                if node_to_update is None:
                    return f"[ERROR] 404: Node with ID {node.node_id} not found."

                node_to_update.title = node.new_title
                node_to_update.execution_order = node.new_execution_order

                return f"Node updated successfully.\n" f"{parent.structure_description}"

        return UpdateNodeTool()

    @property
    def delete_node(self):
        parent = self

        class DeleteNodeTool(BaseTool):
            name: str = "delete_node"
            description: str = "Delete a node from the structure"
            args_schema: Type[BaseModel] = DeleteNode

            def _run(self, **kwargs) -> str:
                node = DeleteNode(**kwargs)
                assert parent.root is not None, "Root node does not exist"

                node_to_delete = parent.find_node(node.node_id, [parent.root])
                if node_to_delete is None:
                    return f"[ERROR] 404: Node with ID {node.node_id} not found."
                if node_to_delete == parent.root:
                    return f"[ERROR] 400: Cannot delete the root node."

                parent_node = parent.find_node(node.node_id, [parent.root])
                if parent_node:
                    parent_node.children.remove(node_to_delete)

                return f"Node deleted successfully.\n" f"{parent.structure_description}"

        return DeleteNodeTool()
