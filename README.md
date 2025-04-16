# ARC AGI 2 Project: Comprehensive Implementation Checklist 🚀🎯

This document outlines the full implementation plan, workflow, and technical checklist for the ARC AGI 2 project. Our goal is to develop a novel, modular system capable of achieving high performance on the ARC AGI 2 benchmark by integrating cutting-edge concepts like Kolmogorov-Arnold Networks (KAN 2.0), Neural Turing Machines (NTM), DreamCoder program synthesis, and an advanced interactive visualization engine. This checklist is designed for collaborative development, emphasizing robust source control, modularity, rigorous testing, and clear documentation.

**Guiding Principles:**
- **Modularity:** All major components (KAN, NTM variants, Memory Ops, DreamCoder, Visualization) should be designed as independent, hot-swappable modules with clearly defined interfaces.
- **Interpretability & Debugging:** The PyVista/IMGUI visualization engine is central, providing deep, real-time insight into every tensor and computational state via hooks, enabling interactive debugging and analysis with minimal coding post-setup.
- **Collaboration:** Strict adherence to Git best practices is crucial for managing complexity and facilitating teamwork.
- **Rigorous Evaluation:** Success is defined by achieving high performance (e.g., ≥80%) on ARC AGI 2 locally *before* submission, backed by thorough ablation studies.
- **Evolvability:** This checklist is a living document; update it as the project evolves and new insights emerge.

---

## Phase 0: Project Setup & Source Control Foundation 🔧 Git

- [ ] **Initialize Repository & Collaboration Tools**
    - [ ] Create GitHub repository.
        - **Condition:** Repository exists online.
        - **Answer:** What is the URL of the remote repository? `_________________________`
        - **Git:** `git init`, `git remote add origin <repository-url>`
    - [ ] Set up project structure (e.g., `src/`, `data/`, `models/`, `notebooks/`, `docs/`, `tests/`).
        - **Condition:** Standard project directories are created and include `.gitkeep` files where necessary.
        - **Answer:** List the top-level directories created: `_________________________`
        - **Git:** `git add .`, `git commit -m "Initial project structure"`
    - [ ] Push initial structure to remote.
        - **Condition:** `main` branch exists on the remote repository.
        - **Git:** `git push -u origin main`
    - [ ] Set up GitHub Projects or Issue Tracker for task management.
        - **Condition:** A project board or issue labeling system is in place.
        - **Answer:** Link to Project Board/Issue Tracker: `_________________________`

- [ ] **Define and Document Git Workflow & Best Practices**
    - [ ] Create `GIT_GUIDELINES.md` document.
        - **Condition:** File exists in the repository root.
        - **Git:** Create branch `docs/git-guidelines`, add file, commit, create PR, merge.
    - [ ] Define Branching Strategy (e.g., `main`, `develop`, `feature/<>`, `bugfix/<>`, `experiment/<>`, `hotfix/<>`).
        - **Condition:** Strategy is clearly documented in `GIT_GUIDELINES.md`.
        - **Answer:** Summarize the primary branches and their purpose: `_________________________`
    - [ ] Define Pull Request (PR) Protocol.
        - **Condition:** Protocol documented, requiring descriptive titles/bodies, linking issues, mandatory code reviews (at least 1 approval), and passing CI checks (setup later).
        - **Answer:** Who are the designated reviewers initially? `_________________________`
    - [ ] Define Commit Message Convention (e.g., Conventional Commits).
        - **Condition:** Convention specified in `GIT_GUIDELINES.md`.
        - **Answer:** Link to chosen convention standard: `_________________________`
    - [ ] Define Merge Strategy (e.g., Squash and Merge, Rebase and Merge).
        - **Condition:** Chosen strategy documented in `GIT_GUIDELINES.md`.
        - **Answer:** What merge strategy will be used for feature branches into `develop`/`main`? `_________________________`
    - [ ] Document Strategy for Handling Merge Conflicts.
        - **Condition:** Step-by-step guide in `GIT_GUIDELINES.md` (e.g., pull latest `develop`, resolve locally, test, push).
        - **Answer:** What tool(s) are recommended for conflict resolution (e.g., VS Code, `git mergetool`)? `_________________________`
    - [ ] Document Strategy for Messy State Recovery & Code Cleanup.
        - **Condition:** `GIT_GUIDELINES.md` includes instructions for using `git reflog`, `git reset`, `git revert`, identifying/removing dead code/bloat, and potentially `git filter-branch` or alternatives (with caution).
        - **Answer:** What is the policy on force-pushing (e.g., allowed only on personal feature branches)? `_________________________`
    - [ ] Document Tagging Strategy for Releases and Milestones.
        - **Condition:** Strategy defined (e.g., Semantic Versioning for releases `vX.Y.Z`, milestone tags like `milestone-vis-engine-v1`).
        - **Answer:** Example of a release tag: `_________________________`

- [ ] **Set Up Development Environment**
    - [ ] Create `requirements.txt` or environment configuration (e.g., `conda env export > environment.yml`).
        - **Condition:** File exists and lists core dependencies (Python version, PyTorch, PyVista, Panel/IMGUI lib, NumPy, etc.).
        - **Answer:** What Python version is targeted? `____`; What PyTorch version? `____`; PyVista? `____`
        - **Git:** Commit environment file.
    - [ ] Ensure all collaborators can replicate the environment.
        - **Condition:** At least one other collaborator confirms successful environment setup from the configuration file.
        - **Answer:** Document any OS-specific setup notes: `_________________________`

- [ ] **Hardware Testing & Selection**
    - [ ] Develop benchmark script (`benchmarks/hardware_test.py`) for core operations (tensor math, potential KAN/NTM ops, PyVista rendering).
        - **Condition:** Script exists and measures execution time (ms) and GPU memory usage (MB) on CPU vs. available CUDA GPUs.
        - **Answer:** What specific operations are benchmarked? `_________________________`
        - **Git:** Create branch `feature/hardware-benchmarking`, commit script.
    - [ ] Run benchmarks on target hardware (specify collaborator machines if different).
        - **Condition:** Benchmark results are recorded.
        - **Answer:** List target GPUs tested: `_________________________`
    - [ ] Document hardware choice and rationale in `docs/hardware_choice.md`.
        - **Condition:** Document justifies the primary development/testing hardware based on performance (speed, VRAM) and efficiency.
        - **Answer:** Chosen primary GPU and why: `_________________________`
        - **Git:** Commit results and documentation, create PR, merge `feature/hardware-benchmarking`.

---

## Phase 1: Core ARC AGI Data Handling 💾

- [ ] **Acquire and Structure ARC Datasets**
    - [ ] Download ARC AGI 1 dataset.
        - **Condition:** Dataset files are stored locally (ideally outside the repo, path configured).
        - **Answer:** Path to ARC AGI 1 data: `_________________________`
    - [ ] Download ARC AGI 2 dataset (Training, Public Eval, potentially access structure for Private Eval).
        - **Condition:** Dataset files are stored locally.
        - **Answer:** Path to ARC AGI 2 data: `_________________________`
    - [ ] Define a standardized data loading format/interface in `src/data/arc_dataloader.py`.
        - **Condition:** A class or function exists that can load tasks from both ARC AGI 1 and ARC AGI 2 formats into a consistent internal representation (e.g., list of input/output grids).
        - **Answer:** Describe the internal task representation (e.g., `{'train': [{'input': grid, 'output': grid}], 'test': [{'input': grid, 'output': grid}]}`). `_________________________`
        - **Git:** Create branch `feature/data-handling`, commit dataloader.

- [ ] **Implement Dataset Merging and Selection**
    - [ ] Create functionality to merge ARC AGI 1 and ARC AGI 2 training sets.
        - **Condition:** A function/method exists that returns a combined dataset object.
        - **Answer:** How are potential overlaps or differences handled during merging? `_________________________`
        - **Git:** Commit merging logic.
    - [ ] Implement logic to select specific datasets or subsets (e.g., only ARC1, only ARC2, combined, specific task IDs).
        - **Condition:** Dataloader can be configured to yield data from selected sources.
        - **Answer:** How is dataset selection specified (e.g., config file, function argument)? `_________________________`
        - **Git:** Commit selection logic.
    - [ ] Add unit tests for data loading, merging, and selection.
        - **Condition:** Tests in `tests/test_data.py` verify correct loading and filtering.
        - **Answer:** What specific scenarios are tested (e.g., loading ARC1 task, ARC2 task, merged set size)? `_________________________`
        - **Git:** Commit tests, create PR, merge `feature/data-handling`.

---

## Phase 2: KAN Implementation (Kolmogorov-Arnold Networks) 🧠

- [ ] **Set Up Base KAN Library**
    - [ ] Integrate `pykan` library or implement core KAN structure.
        - **Condition:** KAN layers (`KANLayer`) and model (`KAN`) can be instantiated and run a forward pass.
        - **Answer:** Are we using the official `pykan` or a custom implementation? `_________________________`
        - **Git:** Create branch `feature/kan-core`, add library/implementation, commit.
    - [ ] Ensure GPU compatibility is enabled and tested.
        - **Condition:** Basic KAN model trains significantly faster on GPU than CPU (verify via hardware benchmark script).
        - **Answer:** Confirm successful GPU run: `Yes/No`.
        - **Git:** Commit any necessary GPU compatibility changes.

- [ ] **Implement/Integrate KAN 2.0 Features**
    - [ ] Implement or verify MultKAN functionality (KANs with multiplication nodes).
        - **Condition:** A KAN model can be defined with explicit multiplication layers (`n_m` parameter in `pykan`).
        - **Answer:** How are multiplication nodes specified in the model architecture definition? `_________________________`
        - **Git:** Commit MultKAN example/test.
    - [ ] Integrate `kanpiler`: Compile symbolic formulas into KANs.
        - **Condition:** `kanpiler(inputs, expression)` function successfully creates a KAN from a SymPy expression.
        - **Answer:** Test with a simple formula (e.g., `x*y` or `sin(x)+y^2`). Does it produce the expected KAN structure? `Yes/No`.
        - **Potential Use:** Use this for initializing parts of the network with known physics/math, if applicable to ARC AGI tasks.
        - **Git:** Commit `kanpiler` integration/test.
    - [ ] Integrate KAN Tree Converter: Identify modular structures.
        - **Condition:** `model.tree()` or equivalent function generates a tree graph representing functional modularity (Separability, General Separability, Symmetry).
        - **Answer:** Test on a known separable function (e.g., `f(x,y,z,w) = g(x,y) + h(z,w)`). Does the tree show the correct structure? `Yes/No`.
        - **Potential Use:** Analyze trained KANs to understand learned ARC AGI task structures.
        - **Git:** Commit tree converter integration/test.
    - [ ] Implement KAN pruning and expansion methods (`prune_input`, `expand_width`, `expand_depth`, `perturb`).
        - **Condition:** Functions exist and can modify the KAN architecture as described in KAN 2.0 paper.
        - **Answer:** Test pruning an unused input. Does the KAN structure update correctly? `Yes/No`.
        - **Git:** Commit pruning/expansion implementation/tests.
    - [ ] Implement enhanced KAN attribution scores (Section 4.1 of KAN 2.0 paper).
        - **Condition:** Function exists to calculate node/edge importance scores recursively from the output layer.
        - **Answer:** How do these scores differ from the basic L1 norm on a test case? `_________________________`
        - **Git:** Commit attribution score implementation.

- [ ] **Develop KAN Base Model for ARC AGI**
    - [ ] Define initial KAN architecture(s) suitable for processing ARC grid inputs.
        - **Condition:** Define network shape `[n_in, n_h1, ..., n_out]`. Input layer size `n_in` must handle flattened ARC grids or features extracted from them. Output layer `n_out` must be suitable for generating output grids.
        - **Answer:** Initial proposed KAN shape(s): `_________________________`; How are grid inputs/outputs handled? `_________________________`
    - [ ] Add unit tests for the KAN base model forward pass with dummy ARC data.
        - **Condition:** Tests in `tests/test_kan.py` verify correct input/output shapes.
        - **Answer:** Describe the dummy data used for testing: `_________________________`
        - **Git:** Commit KAN base model definition and tests, create PR, merge `feature/kan-core`.

---

## Phase 3: NTM / Memory Module Implementation 📝

- [ ] **Define NTM Module Interface**
    - [ ] Specify a standard Python abstract base class or interface for all NTM/Memory modules in `src/modules/memory_interface.py`.
        - **Condition:** Interface defines core methods like `read(memory_state, inputs)`, `write(memory_state, processed_data, inputs)`, `initialize_memory(batch_size)`. Outputs must be clearly defined (e.g., read data, new memory state).
        - **Answer:** List the exact method signatures defined in the interface: `_________________________`
        - **Git:** Create branch `feature/ntm-memory`, commit interface definition.

- [ ] **Implement Token Summarization Utilities** (Leveraging TTM Paper ideas, adaptable for KAN context if needed)
    - [ ] Implement MLP-based token summarization (`src/modules/token_summarization.py`).
        - **Condition:** Function `summarize_mlp(tokens, k)` exists, uses MLP to compute weights, performs weighted sum, returns tensor of shape `[batch, k, dim]`.
        - **Answer:** MLP architecture used (layers, activation): `_________________________`
        - **Git:** Commit MLP summarizer.
    - [ ] Implement Query-based token summarization.
        - **Condition:** Function `summarize_query(tokens, k, query_vectors)` exists, uses attention with learned queries, returns tensor of shape `[batch, k, dim]`.
        - **Answer:** How are query vectors managed/learned? `_________________________`
        - **Git:** Commit Query summarizer.
    - [ ] Implement Pooling-based token summarization (Average/Max).
        - **Condition:** Function `summarize_pooling(tokens, k, pool_type='avg')` exists, divides tokens into groups, pools, returns tensor of shape `[batch, k, dim]`.
        - **Answer:** Is a projection layer used after pooling? `Yes/No`.
        - **Git:** Commit Pooling summarizer.
    - [ ] Add tests for all summarization methods.
        - **Condition:** `tests/test_token_summarization.py` verifies output shapes and basic properties.
        - **Answer:** Test with varying input sizes `p` and output sizes `k`.
        - **Git:** Commit tests.

- [ ] **Implement NTM Variant 1: TTM-Style Memory** (Based on Token Turing Machines paper)
    - [ ] Create `TTMMemory` class implementing `MemoryInterface`.
        - **Condition:** Class exists in `src/modules/ttm_memory.py`.
    - [ ] Implement `read` operation using token summarization (e.g., `Sr([Mt || It])`). Specify `r` (read size).
        - **Condition:** `read` method returns `r` tokens. Requires positional embeddings for distinguishing memory/input.
        - **Answer:** Value of `r` chosen: `____`; Positional embedding implementation details: `_________________________`
        - **Git:** Commit `TTMMemory.read`.
    - [ ] Implement `write` operation using token summarization (e.g., `Sm([Mt || Ot || It])`). Specify `m` (memory size).
        - **Condition:** `write` method returns `m` tokens for the new memory state `Mt+1`. Requires positional embeddings.
        - **Answer:** Value of `m` chosen: `____`; Summarization method used (MLP, Query, Pool)? `_________________________`
        - **Git:** Commit `TTMMemory.write`.
    - [ ] Implement `initialize_memory`.
        - **Condition:** Method returns an initial memory tensor (e.g., learnable parameter or zeros) of shape `[batch_size, m, dim]`.
        - **Answer:** Initialization strategy: `_________________________`
        - **Git:** Commit `TTMMemory.initialize_memory`.
    - [ ] Add tests for `TTMMemory`.
        - **Condition:** `tests/test_ttm_memory.py` verifies read/write shapes and state updates.
        - **Git:** Commit tests.

- [ ] **Implement NTM Variant 2: Differentiable Neural Computer (DNC) Style Memory** (Optional, based on Graves et al.)
    - [ ] Create `DNCMemory` class implementing `MemoryInterface`.
    - [ ] Implement content-based addressing and location-based addressing mechanisms.
    - [ ] Implement DNC read operation (using read weights).
    - [ ] Implement DNC write operation (using write weights, erase/add vectors).
    - [ ] Implement memory allocation / usage weights.
    - [ ] Add tests for `DNCMemory`.
        - **Condition:** All DNC components implemented and tested.
        - **Answer:** Provide details on addressing mechanisms used: `_________________________`
        - **Git:** Commit `DNCMemory` implementation and tests.

- [ ] **Implement NTM Variant 3: Neural Abstract Reasoner (NAR) Style Memory** (Optional, based on Kolev et al.)
    - [ ] Create `NARMemory` class implementing `MemoryInterface`.
    - [ ] Implement the specific memory architecture described in the NAR paper (if details are available/inferrable). Requires careful study of the paper.
        - **Note:** The NAR paper focuses more on spectral regularization than the specific memory details, might need adaptation.
    - [ ] Add tests for `NARMemory`.
        - **Condition:** NAR memory components implemented and tested.
        - **Answer:** Describe the NAR memory mechanism implemented: `_________________________`
        - **Git:** Commit `NARMemory` implementation and tests.

- [ ] **Finalize Memory Module Integration**
    - [ ] Ensure any chosen NTM variant can be easily "plugged into" the main KAN model.
        - **Condition:** Main model class accepts a `memory_module` argument conforming to `MemoryInterface`.
        - **Answer:** How is the memory module integrated into the KAN's forward pass? `_________________________`
        - **Git:** Commit integration logic, create PR, merge `feature/ntm-memory`.

---

## Phase 4: DreamCoder Integration 💭

- [ ] **Define ARC Grid Transformation DSL**
    - [ ] Specify the Domain Specific Language (DSL) primitives for manipulating ARC grids (e.g., `copy_object`, `move_object`, `recolor`, `draw_line`, `tile`, `rotate`, `reflect`, basic control flow).
        - **Condition:** DSL primitives are defined, potentially as Python functions or a formal grammar, in `src/dreamcoder/arc_dsl.py`.
        - **Answer:** List some key DSL primitives defined: `_________________________`
        - **Git:** Create branch `feature/dreamcoder`, commit DSL definition.

- [ ] **Implement Core DreamCoder Algorithm Components**
    - [ ] Implement the "Recognition Model" (Neural Network).
        - **Condition:** A neural network (e.g., CNN or GNN for grids) is defined in `src/dreamcoder/recognition_model.py` that takes an ARC task (examples) and predicts promising DSL programs or components.
        - **Answer:** What is the architecture of the recognition model? `_________________________`
        - **Git:** Commit recognition model structure.
    - [ ] Implement the "Generative Model" (Stochastic Library/Grammar).
        - **Condition:** A probabilistic grammar or library exists in `src/dreamcoder/generative_model.py` that defines the prior `P(program | Library)` over DSL programs, incorporating learned abstractions.
        - **Answer:** How are probabilities assigned to primitives and learned abstractions? `_________________________`
        - **Git:** Commit generative model structure.
    - [ ] Implement the Program Synthesis Search Algorithm (Wake Phase).
        - **Condition:** A search function exists (e.g., enumeration guided by recognition model, potentially MCTS or beam search) in `src/dreamcoder/wake_phase.py` that finds DSL programs solving ARC tasks.
        - **Answer:** Describe the search algorithm used: `_________________________`
        - **Git:** Commit search algorithm.
    - [ ] Implement the Abstraction Phase (Sleep Phase 1).
        - **Condition:** Functionality exists in `src/dreamcoder/abstraction_phase.py` to refactor found programs and identify common sub-programs/patterns to add as new, higher-level primitives to the generative model's library. Uses Bayesian compression criterion (MDL).
        - **Answer:** How are common fragments identified and evaluated for abstraction? `_________________________`
        - **Git:** Commit abstraction logic.
    - [ ] Implement the Dreaming Phase (Sleep Phase 2).
        - **Condition:** Functionality exists in `src/dreamcoder/dreaming_phase.py` to train the recognition model on (1) "replays" (programs found during wake) and (2) "fantasies" (tasks generated by sampling programs from the current generative model).
        - **Answer:** How are fantasy tasks generated and used for training? `_________________________`
        - **Git:** Commit dreaming logic.

- [ ] **Integrate DreamCoder with Main Model**
    - [ ] Define how DreamCoder interacts with the KAN+NTM model. (Is DreamCoder generating the *entire* KAN+NTM, or is it generating high-level plans/subroutines executed *by* the KAN+NTM, or modifying the KAN+NTM structure?). This needs clarification based on the vision.
        - **Assumption:** DreamCoder generates DSL programs that represent the *solution logic* for an ARC task, which might then be executed or interpreted.
        - **Condition:** Clear interaction pathway defined in the main training/inference loop.
        - **Answer:** Describe the chosen interaction mechanism between DreamCoder and KAN+NTM: `_________________________`
        - **Git:** Commit integration code.
    - [ ] Add tests for DreamCoder components.
        - **Condition:** `tests/test_dreamcoder.py` verifies core phases (wake search, abstraction, dreaming).
        - **Git:** Commit tests, create PR, merge `feature/dreamcoder`.

---

## Phase 5: Visualization Engine (PyVista + IMGUI) 📊✨

- [ ] **State Capture & Graph Extraction**
    - [ ] Create `StateTracker` class (`src/visualization/state_tracker.py`).
        - **Condition:** Class exists.
        - **Git:** Create branch `feature/visualization-engine`, commit `StateTracker` skeleton.
    - [ ] Implement PyTorch Hook Registration mechanism in `StateTracker`.
        - **Condition:** `StateTracker.register_hooks(model, module_filter_fn)` can attach forward/backward hooks to specified KAN/NTM submodules.
        - **Answer:** How are modules selected for hooking (e.g., by name, type)? `_________________________`
        - **Git:** Commit hook registration logic.
    - [ ] Implement state recording logic within hooks.
        - **Condition:** Hooks capture input/output tensors, gradients (optional), and store them in `StateTracker` with metadata (module name, class, step, timestamp, input/output).
        - **Answer:** What specific metadata is captured for each tensor state? `_________________________`
        - **Git:** Commit state recording logic.
    - [ ] Implement Computational Graph Extraction.
        - **Condition:** `StateTracker` can traverse the captured states or use a library (e.g., `torchviz`, custom traversal) to build a representation of the computational graph connecting the hooked modules/tensors.
        - **Answer:** Method used for graph extraction: `_________________________`; Graph representation format: `_________________________`
        - **Git:** Commit graph extraction logic.
    - [ ] Standardize captured state and graph format.
        - **Condition:** Consistent data structure used for all captured info (e.g., nested dicts, custom objects). Documented in `docs/visualization_format.md`.
        - **Answer:** Provide snippet of the state/graph format: `_________________________`
        - **Git:** Commit standardized format usage and documentation.

- [ ] **Modular Visualization Mapping (`VisMapper`)**
    - [ ] Define abstract `VisMapper` base class (`src/visualization/vis_mapper.py`) with methods like `can_map(state_data) -> bool`, `map_to_pyvista(state_data) -> pv.BaseDataObject`, `get_ui_controls(state_data, imgui_context) -> UIElement`.
        - **Condition:** Abstract class and methods defined.
        - **Answer:** What specific PyVista object types will mappers return (e.g., `pv.StructuredGrid`, `pv.PolyData`, `pv.MultiBlock`)? `_________________________`
        - **Git:** Commit `VisMapper` interface.
    - [ ] Implement `TensorMapper` for 1D, 2D, 3D, >3D tensors.
        - **Condition:** Maps tensors to appropriate PyVista representations (e.g., 1D->line/bar chart, 2D->image/surface, 3D->volume rendering, >3D->slices/projections). Must support volume rendering with adjustable opacity/colormap via UI controls.
        - **Answer:** How is volume rendering configured (opacity transfer function, color map)? `_________________________`
        - **Git:** Commit `TensorMapper`.
    - [ ] Implement `GraphMapper` for computational graph.
        - **Condition:** Maps the extracted graph structure to a PyVista graph layout (`pv.PolyData` for nodes/edges). Nodes should be visually distinct based on module type/metadata.
        - **Answer:** How are graph nodes positioned/styled? `_________________________`
        - **Git:** Commit `GraphMapper`.
    - [ ] Implement Mappers for other potential data types (e.g., `MeshMapper`, `PointCloudMapper`, `ScalarMapper`).
        - **Condition:** Mappers exist for any other relevant data structures captured.
    - [ ] Implement `MapperRegistry` (`src/visualization/mapper_registry.py`).
        - **Condition:** `MapperRegistry.get_mapper(state_data)` returns the most appropriate `VisMapper` instance based on data type, shape, or metadata.
        - **Answer:** Describe the logic for selecting the best mapper: `_________________________`
        - **Git:** Commit `MapperRegistry`.

- [ ] **PyVista Rendering Engine & IMGUI Dashboard**
    - [ ] Set up core `VisualizationEngine` class (`src/visualization/engine.py`) using `pyvistaqt.BackgroundPlotter` or `panel` integration.
        - **Condition:** Engine creates a window, manages a PyVista plotter, and has an update loop. Uses a black background.
        - **Answer:** Which integration is used (`pyvistaqt`, `panel`, other)? `_________________________`
        - **Git:** Commit basic engine structure.
    - [ ] Implement dynamic scene updates.
        - **Condition:** Engine fetches latest states from `StateTracker`, uses `MapperRegistry` to get mappers, renders the results, and efficiently updates the PyVista scene (e.g., `plotter.update_scalars`, `plotter.add_mesh` with `overwrite=True`). Maximize GPU usage.
        - **Answer:** Update strategy to minimize lag/redundant rendering: `_________________________`
        - **Git:** Commit dynamic update logic.
    - [ ] Implement IMGUI Panel Integration (e.g., using `PyVista Panel` GUI or a dedicated IMGUI library like `dearpygui` linked to PyVista).
        - **Condition:** Engine displays interactive IMGUI panels alongside the PyVista scene.
        - **Answer:** IMGUI library chosen: `_________________________`
        - **Git:** Commit IMGUI integration setup.
    - [ ] Develop Core UI Panels:
        - **State Selector Panel:** Tree view or list to select modules/tensors captured by `StateTracker`. Selecting an item triggers visualization.
        - **Timeline Panel:** Slider/buttons to navigate through recorded steps (training or inference). Updates the visualization to the selected step.
        - **Visualization Control Panel:** Dynamically displays controls generated by the active `VisMapper`'s `get_ui_controls` method (e.g., opacity, colormap for volumes; node size/color for graphs). Parameters should be tweakable live.
        - **Computational Graph Panel:** Dedicated view for the `GraphMapper` output. Nodes should be clickable to select corresponding tensor states.
        - **Performance Monitor Panel:** Displays FPS, state capture rate, GPU memory usage.
        - **(Future) Interactive Editing Panel:** Allows modifying selected tensor values (requires propagating changes back, complex).
        - **(Future) Dataset Selector Panel:** Allows choosing ARC dataset (ARC1, ARC2, Merged) for visualization/inference.
        - **Condition:** Core UI panels are implemented and functional.
        - **Answer:** Describe the layout of the panels in the UI: `_________________________`
        - **Git:** Commit UI panel implementations.
    - [ ] Implement Voxel Hovering/Selection.
        - **Condition:** Using PyVista's picking capabilities (`plotter.enable_picking`), hovering/clicking on visual elements (voxels, graph nodes) displays relevant info from `StateTracker` metadata in a tooltip or panel.
        - **Answer:** Picking mechanism used (cell/point picking, hardware picking)? `_________________________`
        - **Git:** Commit picking implementation.

- [ ] **Integration and Testing**
    - [ ] Integrate `StateTracker` feed into `VisualizationEngine`.
        - **Condition:** Engine correctly receives and processes states captured during a dummy model run.
        - **Answer:** Data transfer mechanism (queue, direct call, callback)? `_________________________`
    - [ ] Test visualization with a simple KAN/NTM model running on ARC data.
        - **Condition:** Live updates appear correctly during a short training/inference run. All core UI panels function.
        - **Answer:** Observed impact on training/inference speed: `_________________________`
    - [ ] Test scalability with larger models / more steps.
        - **Condition:** Engine maintains reasonable FPS (e.g., >15-30 FPS) under expected load. Identify bottlenecks.
        - **Answer:** Target FPS: `____`; Main performance bottlenecks: `_________________________`
    - [ ] Create comprehensive demonstration script (`run_visualization_demo.py`).
        - **Condition:** Script showcases state capture, parallel visualization of different tensor types, graph visualization, timeline navigation, and interactive controls.
        - **Answer:** Key features highlighted in the demo: `_________________________`
        - **Git:** Commit demo script, create PR, merge `feature/visualization-engine`.

---

## Phase 6: Training, Evaluation, and Ablation 📈📉

- [ ] **Implement Training Loop**
    - [ ] Create main training script (`train.py`).
    - [ ] Integrate KAN model, chosen NTM variant, (optional) DreamCoder components, dataset loader, optimizer, scheduler, loss function.
    - [ ] Implement epoch/iteration loop, training step, periodic evaluation, logging (to console, file, and/or WandB/TensorBoard).
    - [ ] Implement Checkpoint Saving (saving model state, optimizer state, epoch/step number, library state if using DreamCoder). Save best model based on validation metric.
    - [ ] Implement Early Stopping based on validation metric patience.
    - [ ] **(Advanced)** Investigate and implement Adaptive Computation strategies (e.g., inspired by AdA - DeepMind) if feasible for ARC AGI.
        - **Condition:** Training loop runs end-to-end, saves checkpoints, logs metrics.
        - **Answer:** Logging framework used (WandB, TensorBoard, basic logging)? `_________________________`; Early stopping metric and patience: `_________________________`
        - **Git:** Create branch `feature/training-loop`, commit training script components.

- [ ] **Implement Evaluation Protocol**
    - [ ] Create evaluation script (`evaluate.py`).
    - [ ] Load trained model checkpoint.
    - [ ] Run inference on ARC AGI 2 evaluation sets (Public, Semi-Private - need Kaggle setup).
    - [ ] Calculate metrics: Pass@k (specifically Pass@2 for ARC), overall accuracy, per-task results.
    - [ ] Implement comparison against baseline models/human performance if available.
        - **Condition:** Evaluation script produces accuracy scores on specified datasets.
        - **Answer:** How is Pass@2 calculated (model gets 1 or 2 attempts per task)? `_________________________`
        - **Git:** Commit evaluation script.

- [ ] **Conduct Ablation Studies**
    - [ ] Plan ablation experiments to test the impact of key components:
        - KAN vs. MLP baseline.
        - Different KAN configurations (depth, width, KAN 2.0 features).
        - Different NTM variants (TTM, DNC, NAR, none).
        - With vs. Without DreamCoder.
        - Different token summarization methods.
        - Impact of curriculum learning (ARC1 -> ARC1+2).
        - Impact of specific visualization hooks (performance overhead).
    - [ ] Run ablation experiments systematically, tracking results.
        - **Condition:** Ablation plan documented (`docs/ablation_plan.md`), experiments run, results recorded.
        - **Answer:** Summarize key findings from initial ablations: `_________________________`
        - **Git:** Commit ablation results and analysis.

- [ ] **Refine and Iterate**
    - [ ] Analyze training logs, evaluation results, and ablation studies.
    - [ ] Use insights (including those from the visualization engine) to refine model architecture, hyperparameters, training strategy.
    - [ ] Repeat training/evaluation cycles.
        - **Condition:** Iterative refinement process is followed.
        - **Git:** Use separate experiment branches for significant changes.

---

## Phase 7: Finalization and Submission 🏆📄

- [ ] **Achieve Target Performance**
    - [ ] Continue iteration until the model achieves the target performance threshold (e.g., ≥80% Pass@2) on a held-out validation set representative of ARC AGI 2 Private Eval.
        - **Condition:** Target performance met and verified.
        - **Answer:** Final best score achieved locally: `____% Pass@2`.

- [ ] **Final Code Cleanup and Documentation**
    - [ ] Remove dead code, experimental branches, unused files.
    - [ ] Ensure all code is well-commented, especially complex parts.
    - [ ] Update all documentation (`README.md`, `GIT_GUIDELINES.md`, `docs/`, etc.) to reflect the final state.
    - [ ] Ensure reproducibility: requirements file is up-to-date, instructions for setup, training, and evaluation are clear.
        - **Condition:** Codebase is clean, well-documented, and reproducible.

- [ ] **Write Research Paper**
    - [ ] Draft paper describing the approach, architecture, experiments, results, ablation studies, visualization engine, and key findings.
    - [ ] Include diagrams, visualizations, and comparison to prior work.
    - [ ] Refine and finalize paper for submission.
        - **Condition:** Paper draft completed.

- [ ] **Prepare for Submission (e.g., ARC Prize Kaggle)**
    - [ ] Package code according to competition rules (if applicable).
    - [ ] Perform final test runs on the target evaluation platform (e.g., Kaggle).
    - [ ] Submit solution.
        - **Condition:** Submission completed.

- [ ] **Open Source Release**
    - [ ] Ensure license is included (e.g., MIT, Apache 2.0).
    - [ ] Push final, cleaned code and documentation to the public repository.
    - [ ] Create final release tag (e.g., `v1.0.0`).
        - **Condition:** Code is publicly released.
        - **Git:** `git tag -a v1.0.0 -m "Stable release for ARC AGI 2 submission"`, `git push origin v1.0.0`

---

*This checklist is a guide. Adapt, add, or remove items as necessary based on project progress and discoveries.*
