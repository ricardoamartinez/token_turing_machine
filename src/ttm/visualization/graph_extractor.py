import torch
import torch.fx as fx
from collections import Counter
from typing import Type, Dict, Any, List
from src.ttm.models.ttm_model import TokenTuringMachine # Import the actual TTM model
from src.ttm.utils.masking import create_causal_mask # Import mask creation utility

# Placeholder simple model for initial testing
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 20)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(20, 5)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

class GraphExtractor:
    """
    Extracts and analyzes the computational graph of a PyTorch model using torch.fx.
    """
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.traced_graph: fx.GraphModule | None = None
        self.graph_analysis: Dict[str, Any] = {}

    def extract_graph(self, concrete_args: tuple | None = None):
        """
        Performs symbolic tracing on the model to get the computational graph.
        Optionally uses concrete_args for models with control flow.
        """
        try:
            # Ensure model is in eval mode for stable graph
            self.model.eval()
            # Try tracing with concrete args if provided
            if concrete_args:
                 print(f"Attempting symbolic trace with concrete_args shapes: {tuple(arg.shape for arg in concrete_args)}")
                 # Note: symbolic_trace expects positional args matching the forward signature (excluding self)
                 self.traced_graph = fx.symbolic_trace(self.model, concrete_args=concrete_args)
            else:
                 print("Attempting symbolic trace without concrete_args...")
                 self.traced_graph = fx.symbolic_trace(self.model)

            print("Computational graph extracted successfully.")
            # print(self.traced_graph.graph) # Optional: print the graph structure
        except Exception as e:
            print(f"Error during symbolic tracing: {e}")
            self.traced_graph = None

    def analyze_graph(self) -> Dict[str, Any]:
        """
        Analyzes the extracted graph to provide insights, e.g., node types.
        """
        if not self.traced_graph:
            print("Graph not extracted yet. Call extract_graph() first.")
            return {}

        node_op_counts = Counter(node.op for node in self.traced_graph.graph.nodes)
        node_target_counts = Counter(str(node.target) for node in self.traced_graph.graph.nodes if node.op == 'call_module' or node.op == 'call_function' or node.op == 'call_method')

        self.graph_analysis = {
            "total_nodes": len(self.traced_graph.graph.nodes),
            "node_op_counts": dict(node_op_counts),
            "node_target_counts": dict(node_target_counts),
            "distinct_node_types": len(node_op_counts), # Answering the README question partially
            "distinct_targets": len(node_target_counts)
        }
        print("Graph analysis complete.")
        return self.graph_analysis

if __name__ == '__main__':
    print("Testing GraphExtractor with a simple model...")
    simple_model = SimpleModel()
    extractor = GraphExtractor(simple_model)
    extractor.extract_graph()

    if extractor.traced_graph:
        analysis = extractor.analyze_graph()
        print("\nGraph Analysis Results:")
        for key, value in analysis.items():
            print(f"- {key}: {value}")

        # Example of accessing the graph nodes
        # print("\nGraph Nodes:")
        # for node in extractor.traced_graph.graph.nodes:
        #     print(f"  Name: {node.name}, Op: {node.op}, Target: {node.target}, Args: {node.args}, Kwargs: {node.kwargs}")


    print("\nTesting GraphExtractor with TTM model...")
    try:
        # Instantiate TTM with standard config from README/paper
        ttm_config = {
            'vocab_size': 13, 'embedding_dim': 128, 'memory_size': 96, 'r': 16,
            'num_layers': 4, 'num_heads': 4, 'hidden_dim': 512, 'dropout': 0.1
        }
        ttm_model = TokenTuringMachine(**ttm_config)
        ttm_extractor = GraphExtractor(ttm_model)

        # --- Create Concrete Args ---
        # Example input tokens (batch_size=2, seq_len=10)
        # Using integers within vocab size (0-12)
        example_input_tokens = torch.randint(0, ttm_config['vocab_size'], (2, 10), dtype=torch.long)
        batch_size, seq_len = example_input_tokens.shape
        device = example_input_tokens.device
        # Get initial memory state (batch_size=2)
        initial_memory = ttm_model.initialize_memory(batch_size=batch_size).to(device)

        # Create dummy masks instead of passing None
        # Dummy key_padding_mask (all False, indicating no padding)
        key_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        # Dummy causal attention mask
        attn_mask = create_causal_mask(seq_len, device=device)
        # Represent mask_eos as a tensor (1 for True)
        mask_eos = torch.tensor(1, dtype=torch.long, device=device) # Use tensor instead of Python bool

        # Concrete args tuple matching TTM forward signature
        concrete_args = (example_input_tokens.to(device), initial_memory, attn_mask, key_padding_mask, mask_eos)
        # --- End Concrete Args ---

        ttm_extractor.extract_graph(concrete_args=concrete_args) # Attempt tracing with concrete args

        if ttm_extractor.traced_graph:
            ttm_analysis = ttm_extractor.analyze_graph()
            print("\nTTM Graph Analysis Results:")
            for key, value in ttm_analysis.items():
                print(f"- {key}: {value}")
            # Store the answer for the README
            distinct_node_types_count = ttm_analysis.get("distinct_node_types", "N/A")
            print(f"\nREADME Answer: Distinct node types identified: {distinct_node_types_count}")
        else:
             print("Skipping TTM analysis as graph extraction failed.")

    except ImportError:
        print("Could not import TokenTuringMachine. Ensure it's defined correctly in src/ttm/models/ttm_model.py.")
    except Exception as e:
         print(f"Error initializing or tracing TTM model: {e}")
         print("Note: torch.fx tracing might require example inputs for models with complex control flow or data-dependent shapes.")
