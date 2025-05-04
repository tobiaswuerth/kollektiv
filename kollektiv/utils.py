from pydantic import BaseModel
import os
import json
import pygraphviz as pgv

from .llm import Storage


def save_pydantic_json(obj: BaseModel, file_name: str) -> None:
    path = os.path.join(Storage.directory, file_name)
    os.makedirs(Storage.directory, exist_ok=True)
    content = obj.model_dump_json(indent=2)
    with open(path, "w", encoding="utf-8") as file:
        file.write(content)


def load_pydantic_json(file_name: str, model: type) -> BaseModel:
    path = os.path.join(Storage.directory, file_name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File '{file_name}' not found in '{Storage.directory}'."
        )
    with open(path, "r", encoding="utf-8") as file:
        content = file.read()
    return model.model_validate_json(content, strict=True)


def generate_project_plan_graph(json_file_path: str, output_png_path: str) -> None:
    """
    Generates a Graphviz PNG visualization of the project plan from a JSON file,
    with specific styling for phases, tasks, inputs, and outputs.

    Args:
        json_file_path: Path to the project_plan_with_tasks.json file.
        output_png_path: Path where the output PNG file should be saved.
    """
    try:
        with open(json_file_path, "r", encoding="utf-8") as f:
            project_plan = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_file_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_file_path}")
        return

    # Use pygraphviz AGraph for directed graph
    dot = pgv.AGraph(
        directed=True,
        strict=False, # Important for defining nodes/edges across subgraphs
        name=project_plan.get("overarching_goal", "Project Plan"),
        rankdir='TB', # Top-to-bottom layout
        nodesep=0.15, # Reduced separation for compactness
        ranksep=0.18, # Reduced separation for compactness
        label=project_plan.get("description", ""),
        labelloc='t',
        fontsize=14, # Slightly smaller font for compactness
        fontname="Arial"
    )

    # --- Define Colors and Shapes ---
    COLOR_PHASE_BG = 'lightgrey'
    COLOR_PHASE_BORDER = 'darkgrey'
    COLOR_PHASE_INPUT = '/orrd4/3' # Phase input relay node

    COLOR_TASK_BG = 'whitesmoke'
    COLOR_TASK_BORDER = 'grey'
    COLOR_TASK_ACTION = 'lightblue'

    COLOR_TASK_INPUT = "/orrd4/2"
    COLOR_TASK_INTERMEDIATE_INPUT = '/orrd4/1'
    COLOR_TASK_OUTPUT = '/greens4/3' 
    COLOR_TASK_INTERMEDIATE_OUTPUT = '/greens4/1'

    SHAPE_RECTANGLE = 'box'
    SHAPE_ACTION = 'box' # Task action shape
    SHAPE_FILE = 'oval' # All file nodes are rectangles now
    SHAPE_RELAY = 'oval' # Relay nodes are ovals

    # --- Global Tracking ---
    # Stores the latest node ID where a file was produced (output)
    file_source_node_id = {}
    # Stores the node ID for phase input relay nodes { (phase_idx, filename): relay_node_id }
    phase_input_relay_nodes = {}
    # Keep track of nodes added globally to avoid attribute conflicts if styles differ
    globally_added_nodes = set()

    # --- Default Attributes ---
    dot.node_attr['style'] = 'filled'
    dot.node_attr['fontname'] = 'Arial'
    dot.node_attr['fontsize'] = 9 # Smaller font for compactness
    dot.node_attr['shape'] = SHAPE_RECTANGLE # Default shape

    dot.edge_attr['color'] = 'gray40'
    dot.edge_attr['arrowhead'] = 'vee'
    dot.edge_attr['penwidth'] = 1.0

    # --- Phase Iteration ---
    previous_phase_bottom_node = None  # Track last node of previous phase for vertical ordering
    for i, phase in enumerate(project_plan.get("project_phases", [])):
        phase_name = phase.get("phase_name", f"Phase {i+1}")
        phase_id = f"cluster_phase_{i}"
        phase_deliverable_files = {d.get("file_name") for d in phase.get("deliverable_files", []) if d.get("file_name")}
        phase_required_inputs = phase.get("required_inputs", [])

        # Create subgraph for the phase
        # Using 'with' ensures proper subgraph context handling in pygraphviz
        with dot.subgraph(name=phase_id, label=phase_name) as phase_graph:
            phase_graph.graph_attr['style'] = 'filled, rounded' # Rounded corners for phase
            phase_graph.graph_attr['color'] = COLOR_PHASE_BORDER
            phase_graph.graph_attr['fillcolor'] = COLOR_PHASE_BG
            phase_graph.graph_attr['labeljust'] = 'l'
            phase_graph.graph_attr['fontsize'] = 11
            phase_graph.graph_attr['fontname'] = 'Arial Bold'
            phase_graph.graph_attr['rankdir'] = 'TB'
            phase_graph.graph_attr['nodesep'] = '0.10' # More compact
            phase_graph.graph_attr['ranksep'] = '0.15' # More compact

            # Track the first and last node in this phase for vertical ordering
            phase_first_node = None
            phase_last_node = None

            # 1. Define Phase Input Relay Nodes (Purple Ovals) - Placed within Phase Subgraph
            for input_file in phase_required_inputs:
                # Unique ID for the phase-level relay node
                phase_relay_node_id = f"phase_relay_{i}_{input_file}"
                phase_input_relay_nodes[(i, input_file)] = phase_relay_node_id

                # Add relay node inside the phase subgraph (NOT task subgraph)
                phase_graph.add_node(phase_relay_node_id, label=input_file, shape=SHAPE_RELAY, fillcolor=COLOR_PHASE_INPUT)
                if not phase_first_node:
                    phase_first_node = phase_relay_node_id
                phase_last_node = phase_relay_node_id

                # Connect the actual source to the phase relay node
                source_node_id = file_source_node_id.get(input_file)
                if source_node_id:
                    # Add edge in the main graph to connect potentially across phases
                    dot.add_edge(source_node_id, phase_relay_node_id)
                else:
                    # If source not found within plan, assume external
                    if input_file not in globally_added_nodes:
                         dot.add_node(input_file, label=input_file, shape=SHAPE_FILE, fillcolor=COLOR_PHASE_INPUT)
                         globally_added_nodes.add(input_file)
                    # Connect external source directly to phase relay node
                    dot.add_edge(input_file, phase_relay_node_id)


            # 2. Process Tasks within the Phase
            for j, task in enumerate(phase.get("tasks", [])):
                task_name = task.get("task_name", f"Task {j+1}")
                task_subgraph_id = f"cluster_task_{i}_{j}"
                task_action_node_id = f"action_{i}_{j}" # Unique ID for the task action node

                # Create subgraph for the task
                with phase_graph.subgraph(name=task_subgraph_id, label=f"Task: {task_name}") as task_graph:
                    task_graph.graph_attr['style'] = 'filled'
                    task_graph.graph_attr['color'] = COLOR_TASK_BORDER
                    task_graph.graph_attr['fillcolor'] = COLOR_TASK_BG
                    task_graph.graph_attr['labeljust'] = 'l'
                    task_graph.graph_attr['fontsize'] = 10
                    task_graph.graph_attr['rankdir'] = 'TB'
                    task_graph.graph_attr['nodesep'] = 0.2
                    task_graph.graph_attr['ranksep'] = 0.3

                    # Add the task action node (Light Blue Rectangle)
                    task_graph.add_node(task_action_node_id, label=task_name, shape=SHAPE_ACTION, fillcolor=COLOR_TASK_ACTION)
                    if not phase_first_node:
                        phase_first_node = task_action_node_id
                    phase_last_node = task_action_node_id

                    # 3. Connect Task Inputs
                    task_required_inputs = task.get("required_inputs", [])
                    for input_file in task_required_inputs:
                        # Check if it's a phase input (requires phase relay -> task relay)
                        phase_relay_node_id = phase_input_relay_nodes.get((i, input_file))
                        if phase_relay_node_id:
                            # Create a task-specific relay node (Purple Oval) INSIDE the task subgraph
                            task_phase_relay_node_id = f"task_phase_relay_{i}_{j}_{input_file}"
                            task_graph.add_node(task_phase_relay_node_id, label=input_file, shape=SHAPE_RELAY, fillcolor=COLOR_TASK_INPUT)

                            # Connect Phase Relay -> Task Relay (Edge added to parent graph)
                            dot.add_edge(phase_relay_node_id, task_phase_relay_node_id)

                            # Connect Task Relay -> Task Action (Edge within task subgraph)
                            task_graph.add_edge(task_phase_relay_node_id, task_action_node_id)
                        else:
                            # Input comes from a previous task (intermediate file)
                            source_node_id = file_source_node_id.get(input_file)
                            if source_node_id:
                                # Create an intermediate relay node (Light Purple Oval) INSIDE the task subgraph
                                task_intermediate_relay_node_id = f"task_intermediate_relay_{i}_{j}_{input_file}"
                                task_graph.add_node(task_intermediate_relay_node_id, label=input_file, shape=SHAPE_RELAY, fillcolor=COLOR_TASK_INTERMEDIATE_INPUT)

                                # Connect Source Node -> Intermediate Relay (Edge added to parent graph)
                                dot.add_edge(source_node_id, task_intermediate_relay_node_id)

                                # Connect Intermediate Relay -> Task Action (Edge within task subgraph)
                                task_graph.add_edge(task_intermediate_relay_node_id, task_action_node_id)
                            else:
                                # If source still not found, treat as external/undefined globally
                                if input_file not in globally_added_nodes:
                                     dot.add_node(input_file, label=input_file, shape=SHAPE_FILE, fillcolor=COLOR_PHASE_INPUT)
                                     globally_added_nodes.add(input_file)
                                # Connect external source directly to task action (No relay for external)
                                dot.add_edge(input_file, task_action_node_id)


                    # 4. Define Task Output Node
                    task_deliverable = task.get("deliverable_file", {})
                    output_file = task_deliverable.get("file_name")

                    if output_file:
                        output_node_id = f"output_{i}_{j}_{output_file}" # Unique ID for this output instance
                        phase_last_node = output_node_id

                        # Determine color: Red if phase deliverable, Blue if intermediate
                        is_final_deliverable = output_file in phase_deliverable_files
                        output_color = COLOR_TASK_OUTPUT if is_final_deliverable else COLOR_TASK_INTERMEDIATE_OUTPUT

                        # Add output node within the task subgraph (Oval)
                        task_graph.add_node(output_node_id, label=output_file, shape=SHAPE_RELAY, fillcolor=output_color)

                        # Connect task action to its output
                        task_graph.add_edge(task_action_node_id, output_node_id)

                        # Update the global tracker for this file's source node ID
                        file_source_node_id[output_file] = output_node_id
                        globally_added_nodes.add(output_node_id) # Track that this node exists

        # After phase_graph context: enforce vertical ordering between phases
        if previous_phase_bottom_node and phase_first_node:
            # Add invisible edge to force vertical stacking of clusters
            dot.add_edge(previous_phase_bottom_node, phase_first_node, style='invis', weight=100)
        if phase_last_node:
            previous_phase_bottom_node = phase_last_node

    # --- Render the graph ---
    try:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_png_path), exist_ok=True)
        dot.draw(output_png_path, format='png', prog='dot')
        print(f"Project plan graph saved to {output_png_path}")
    except Exception as e:
        print(f"Error rendering graph with pygraphviz: {e}")
        # Attempt to save the DOT source for debugging
        try:
            dot_file_path = os.path.splitext(output_png_path)[0] + ".gv"
            dot.write(dot_file_path)
            print(f"Graphviz DOT source saved to {dot_file_path} for debugging.")
        except Exception as dot_e:
            print(f"Could not save DOT source: {dot_e}")