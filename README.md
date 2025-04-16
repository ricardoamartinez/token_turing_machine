# ARC AGI 2 Project: Comprehensive Implementation Checklist 🚀🎯

This document outlines the full implementation plan, workflow, and technical checklist for the ARC AGI 2 project. Our goal is to develop a novel, modular system capable of achieving high performance on the ARC AGI 2 benchmark by integrating cutting-edge concepts like Kolmogorov-Arnold Networks (KAN 2.0), Neural Turing Machines (NTM), DreamCoder program synthesis, and an advanced interactive visualization engine. This checklist is designed for collaborative development, emphasizing robust source control, modularity, rigorous testing, and clear documentation.

**Guiding Principles:**
- **Modularity:** All major components (KAN, NTM variants, Memory Ops, DreamCoder, Visualization) should be designed as independent, hot-swappable modules with clearly defined interfaces. [See Modularity Principles](#modularity-principles)
- **Interpretability & Debugging:** The PyVista/IMGUI visualization engine is central, providing deep, real-time insight into every tensor and computational state via hooks, enabling interactive debugging and analysis with minimal coding post-setup. [See Visualization Engine Principles](#visualization-engine-principles)
- **Collaboration:** Strict adherence to documented Git best practices is crucial for managing complexity and facilitating teamwork. [See Git Workflow Principles](#git-workflow-principles)
- **Rigorous Evaluation:** Success is defined by achieving high performance (e.g., ≥80%) on ARC AGI 2 locally *before* submission, backed by thorough ablation studies. [See Evaluation Strategy](#evaluation-strategy)
- **Evolvability:** This checklist is a living document; update it as the project evolves and new insights emerge.

---

## Phase 0: Project Setup & Source Control Foundation 🔧 Git

- [ ] **Initialize Repository & Collaboration Tools**
    - [ ] Create GitHub repository. [Details](#init-repo-create-details)
    - [ ] Set up project structure (e.g., `src/`, `data/`, `models/`, `notebooks/`, `docs/`, `tests/`). [Details](#init-structure-details)
    - [ ] Push initial structure to remote. [Details](#init-push-details)
    - [ ] Set up GitHub Projects or Issue Tracker for task management. [Details](#init-tracker-details)

- [ ] **Define and Document Git Workflow & Best Practices**
    - [ ] Create `GIT_GUIDELINES.md` document. [Details](#git-guidelines-doc-details)
    - [ ] Define Branching Strategy (e.g., `main`, `develop`, `feature/<>`, `bugfix/<>`, `experiment/<>`, `hotfix/<>`) in `GIT_GUIDELINES.md`. [Details](#git-branching-details)
    - [ ] Define Pull Request (PR) Protocol in `GIT_GUIDELINES.md` (Reviews, CI checks). [Details](#git-pr-details)
    - [ ] Define Commit Message Convention (e.g., Conventional Commits) in `GIT_GUIDELINES.md`. [Details](#git-commit-convention-details)
    - [ ] Define Merge Strategy (e.g., Squash, Rebase) in `GIT_GUIDELINES.md`. [Details](#git-merge-strategy-details)
    - [ ] Document Strategy for Handling Merge Conflicts in `GIT_GUIDELINES.md`. [Details](#git-conflict-resolution-details)
    - [ ] Document Strategy for Messy State Recovery & Code Cleanup in `GIT_GUIDELINES.md`. [Details](#git-recovery-cleanup-details)
    - [ ] Document Tagging Strategy for Releases and Milestones in `GIT_GUIDELINES.md`. [Details](#git-tagging-details)

- [ ] **Set Up Development Environment**
    - [ ] Create `requirements.txt` or `environment.yml` with core dependencies. [Details](#env-config-details)
    - [ ] Ensure all collaborators can replicate the environment. [Details](#env-replication-details)

- [ ] **Hardware Testing & Selection**
    - [ ] Develop benchmark script (`benchmarks/hardware_test.py`) for core operations. [Details](#hw-benchmark-script-details)
    - [ ] Run benchmarks on target hardware and record results. [Details](#hw-benchmark-run-details)
    - [ ] Document hardware choice and rationale in `docs/hardware_choice.md`. [Details](#hw-choice-doc-details)

---

## Phase 1: Core ARC AGI Data Handling 💾

- [ ] **Acquire and Structure ARC Datasets**
    - [ ] Download ARC AGI 1 dataset. [Details](#data-acquisition-arc1-details)
    - [ ] Download ARC AGI 2 dataset (Training, Public Eval). [Details](#data-acquisition-arc2-details)
    - [ ] Define a standardized data loading format/interface in `src/data/arc_dataloader.py`. [Details](#data-loader-interface-details)

- [ ] **Implement Dataset Merging and Selection**
    - [ ] Create functionality to merge ARC AGI 1 and ARC AGI 2 training sets. [Details](#data-merging-details)
    - [ ] Implement logic to select specific datasets or subsets (e.g., ARC1, ARC2, combined, task IDs). [Details](#data-selection-details)
    - [ ] Add unit tests for data loading, merging, and selection (`tests/test_data.py`). [Details](#data-tests-details)

---

## Phase 2: KAN Implementation (Kolmogorov-Arnold Networks) 🧠

- [ ] **Set Up Base KAN Library**
    - [ ] Integrate `pykan` library or implement core KAN structure. [Details](#kan-library-setup-details)
    - [ ] Ensure GPU compatibility is enabled and tested. [Details](#kan-gpu-test-details)

- [ ] **Implement/Integrate KAN 2.0 Features**
    - [ ] Implement or verify MultKAN functionality (KANs with multiplication nodes). [Details](#kan-multkan-details)
    - [ ] Integrate `kanpiler`: Compile symbolic formulas into KANs. [Details](#kan-kanpiler-details)
    - [ ] Integrate KAN Tree Converter: Identify modular structures. [Details](#kan-tree-converter-details)
    - [ ] Implement KAN pruning and expansion methods (`prune_input`, `expand_width`, `expand_depth`, `perturb`). [Details](#kan-pruning-expansion-details)
    - [ ] Implement enhanced KAN attribution scores (Section 4.1 of KAN 2.0 paper). [Details](#kan-attribution-details)

- [ ] **Develop KAN Base Model for ARC AGI**
    - [ ] Define initial KAN architecture(s) suitable for processing ARC grid inputs. [Details](#kan-base-model-arch-details)
    - [ ] Add unit tests for the KAN base model forward pass with dummy ARC data (`tests/test_kan.py`). [Details](#kan-base-model-test-details)

---

## Phase 3: NTM / Memory Module Implementation 📝

- [ ] **Define NTM Module Interface**
    - [ ] Specify a standard Python abstract base class or interface for all NTM/Memory modules in `src/modules/memory_interface.py`. [Details](#ntm-interface-details)

- [ ] **Implement Token Summarization Utilities** (Leveraging TTM Paper ideas)
    - [ ] Implement MLP-based token summarization (`src/modules/token_summarization.py`). [Details](#ts-mlp-details)
    - [ ] Implement Query-based token summarization. [Details](#ts-query-details)
    - [ ] Implement Pooling-based token summarization (Average/Max). [Details](#ts-pooling-details)
    - [ ] Add tests for all summarization methods (`tests/test_token_summarization.py`). [Details](#ts-tests-details)

- [ ] **Implement NTM Variant 1: TTM-Style Memory** (Based on Token Turing Machines paper)
    - [ ] Create `TTMMemory` class implementing `MemoryInterface`. [Details](#ntm-ttm-class-details)
    - [ ] Implement `read` operation using token summarization (e.g., `Sr([Mt || It])`) with positional embeddings. [Details](#ntm-ttm-read-details)
    - [ ] Implement `write` operation using token summarization (e.g., `Sm([Mt || Ot || It])`) with positional embeddings. [Details](#ntm-ttm-write-details)
    - [ ] Implement `initialize_memory`. [Details](#ntm-ttm-init-details)
    - [ ] Add tests for `TTMMemory` (`tests/test_ttm_memory.py`). [Details](#ntm-ttm-test-details)

- [ ] **Implement NTM Variant 2: Differentiable Neural Computer (DNC) Style Memory** (Optional, based on Graves et al.)
    - [ ] Create `DNCMemory` class implementing `MemoryInterface`. [Details](#ntm-dnc-class-details)
    - [ ] Implement content-based addressing and location-based addressing mechanisms. [Details](#ntm-dnc-addressing-details)
    - [ ] Implement DNC read operation (using read weights). [Details](#ntm-dnc-read-details)
    - [ ] Implement DNC write operation (using write weights, erase/add vectors). [Details](#ntm-dnc-write-details)
    - [ ] Implement memory allocation / usage weights. [Details](#ntm-dnc-allocation-details)
    - [ ] Add tests for `DNCMemory` (`tests/test_dnc_memory.py`). [Details](#ntm-dnc-test-details)

- [ ] **Implement NTM Variant 3: Neural Abstract Reasoner (NAR) Style Memory** (Optional, based on Kolev et al.)
    - [ ] Create `NARMemory` class implementing `MemoryInterface`. [Details](#ntm-nar-class-details)
    - [ ] Implement the specific memory architecture described in the NAR paper (if details available/inferrable). [Details](#ntm-nar-arch-details)
    - [ ] Add tests for `NARMemory` (`tests/test_nar_memory.py`). [Details](#ntm-nar-test-details)

- [ ] **Finalize Memory Module Integration**
    - [ ] Ensure any chosen NTM variant can be easily "plugged into" the main KAN model via the defined interface. [Details](#ntm-integration-details)

---

## Phase 4: DreamCoder Integration 💭

- [ ] **Define ARC Grid Transformation DSL**
    - [ ] Specify the Domain Specific Language (DSL) primitives for manipulating ARC grids (e.g., copy, move, recolor, draw, tile, rotate, reflect, control flow) in `src/dreamcoder/arc_dsl.py`. [Details](#dc-dsl-details)

- [ ] **Implement Core DreamCoder Algorithm Components**
    - [ ] Implement the "Recognition Model" (Neural Network) in `src/dreamcoder/recognition_model.py` (e.g., CNN/GNN predicts DSL components from ARC tasks). [Details](#dc-recognition-model-details)
    - [ ] Implement the "Generative Model" (Stochastic Library/Grammar) in `src/dreamcoder/generative_model.py` (defines `P(program | Library)` including learned abstractions). [Details](#dc-generative-model-details)
    - [ ] Implement the Program Synthesis Search Algorithm (Wake Phase) in `src/dreamcoder/wake_phase.py` (finds DSL programs solving ARC tasks). [Details](#dc-wake-phase-details)
    - [ ] Implement the Abstraction Phase (Sleep Phase 1) in `src/dreamcoder/abstraction_phase.py` (refactors programs, finds common fragments, adds to library using MDL). [Details](#dc-abstraction-phase-details)
    - [ ] Implement the Dreaming Phase (Sleep Phase 2) in `src/dreamcoder/dreaming_phase.py` (trains recognition model on replays and fantasy tasks). [Details](#dc-dreaming-phase-details)

- [ ] **Integrate DreamCoder with Main Model**
    - [ ] Define and implement the interaction mechanism between DreamCoder and KAN+NTM (e.g., DC generates DSL plan executed by KAN+NTM). [Details](#dc-integration-mechanism-details)
    - [ ] Add tests for DreamCoder components (`tests/test_dreamcoder.py`). [Details](#dc-tests-details)

---

## Phase 5: Visualization Engine (PyVista + IMGUI) 📊✨

- [ ] **State Capture & Graph Extraction**
    - [ ] Create `StateTracker` class (`src/visualization/state_tracker.py`). [Details](#vis-state-tracker-class-details)
    - [ ] Implement PyTorch Hook Registration mechanism in `StateTracker` (forward/backward hooks on specified modules). [Details](#vis-hook-registration-details)
    - [ ] Implement state recording logic within hooks (capture tensors, metadata like module name, step, etc.). [Details](#vis-state-recording-details)
    - [ ] Implement Computational Graph Extraction (traverse states or use library). [Details](#vis-graph-extraction-details)
    - [ ] Standardize and document captured state/graph format (`docs/visualization_format.md`). [Details](#vis-format-details)

- [ ] **Modular Visualization Mapping (`VisMapper`)**
    - [ ] Define abstract `VisMapper` base class (`src/visualization/vis_mapper.py`) with `can_map`, `map_to_pyvista`, `get_ui_controls` methods. [Details](#vis-mapper-interface-details)
    - [ ] Implement `TensorMapper` for 1D, 2D, 3D (volume rendering), >3D tensors. [Details](#vis-tensor-mapper-details)
    - [ ] Implement `GraphMapper` for computational graph. [Details](#vis-graph-mapper-details)
    - [ ] Implement Mappers for other potential data types (e.g., Mesh, PointCloud, Scalar). [Details](#vis-other-mappers-details)
    - [ ] Implement `MapperRegistry` for automatic mapper selection (`src/visualization/mapper_registry.py`). [Details](#vis-mapper-registry-details)

- [ ] **PyVista Rendering Engine & IMGUI Dashboard**
    - [ ] Set up core `VisualizationEngine` class (`src/visualization/engine.py`) using `pyvistaqt` or `panel`. [Details](#vis-engine-setup-details)
    - [ ] Implement dynamic scene updates from `StateTracker` data (efficiently update PyVista actors). [Details](#vis-dynamic-updates-details)
    - [ ] Implement IMGUI Panel Integration (using `panel` or external lib like `dearpygui`). [Details](#vis-imgui-integration-details)
    - [ ] Develop Core UI Panels:
        - [ ] State Selector Panel (Tree/List). [Details](#vis-ui-panel-state-selector)
        - [ ] Timeline Panel (Slider/Buttons). [Details](#vis-ui-panel-timeline)
        - [ ] Visualization Control Panel (Dynamic controls from active `VisMapper`). [Details](#vis-ui-panel-vis-controls)
        - [ ] Computational Graph Panel (Clickable nodes). [Details](#vis-ui-panel-graph)
        - [ ] Performance Monitor Panel (FPS, Memory). [Details](#vis-ui-panel-performance)
        - [ ] (Future) Interactive Editing Panel. [Details](#vis-ui-panel-edit)
        - [ ] (Future) Dataset Selector Panel. [Details](#vis-ui-panel-dataset)
    - [ ] Implement Voxel/Node Hovering & Selection using PyVista picking. [Details](#vis-picking-details)

- [ ] **Integration and Testing**
    - [ ] Integrate `StateTracker` data feed into `VisualizationEngine`. [Details](#vis-feed-integration-details)
    - [ ] Test visualization with a simple KAN/NTM model running on ARC data (live updates). [Details](#vis-live-test-details)
    - [ ] Test scalability and performance (maintain target FPS). [Details](#vis-scalability-test-details)
    - [ ] Create comprehensive demonstration script (`run_visualization_demo.py`). [Details](#vis-demo-script-details)

---

## Phase 6: Training, Evaluation, and Ablation 📈📉

- [ ] **Implement Training Loop**
    - [ ] Create main training script (`train.py`). [Details](#train-script-details)
    - [ ] Integrate KAN model, chosen NTM variant, DreamCoder components, dataset loader, optimizer, scheduler, loss function. [Details](#train-integration-details)
    - [ ] Implement epoch/iteration loop, training step, periodic evaluation, logging (Console, File, WandB/TensorBoard). [Details](#train-loop-logic-details)
    - [ ] Implement Checkpoint Saving (model, optimizer, epoch, library state) and Loading/Resuming. [Details](#train-checkpointing-details)
    - [ ] Implement Early Stopping based on validation metric. [Details](#train-early-stopping-details)
    - [ ] **(Advanced)** Investigate and implement Adaptive Computation strategies (e.g., inspired by AdA - DeepMind). [Details](#train-ada-details)

- [ ] **Implement Evaluation Protocol**
    - [ ] Create evaluation script (`evaluate.py`). [Details](#eval-script-details)
    - [ ] Implement loading checkpoints and running inference on ARC AGI 2 eval sets (Public, Semi-Private). [Details](#eval-inference-details)
    - [ ] Calculate metrics: Pass@k (specifically Pass@2), overall accuracy, per-task results. [Details](#eval-metrics-details)
    - [ ] Implement comparison against baseline models/human performance (optional). [Details](#eval-comparison-details)

- [ ] **Conduct Ablation Studies**
    - [ ] Plan ablation experiments (KAN vs. MLP, NTM variants, DC on/off, summarization methods, curriculum, vis overhead). Document in `docs/ablation_plan.md`. [Details](#ablation-plan-details)
    - [ ] Run planned ablation experiments systematically, tracking results. [Details](#ablation-run-details)

- [ ] **Refine and Iterate**
    - [ ] Analyze training logs, evaluation results, visualization insights, and ablation studies. [Details](#refine-analysis-details)
    - [ ] Use insights to refine model architecture, hyperparameters, training strategy (using experiment branches). [Details](#refine-iteration-details)

---

## Phase 7: Finalization and Submission 🏆📄

- [ ] **Achieve Target Performance**
    - [ ] Continue iteration until model achieves ≥80% Pass@2 on a representative validation set. [Details](#final-performance-details)

- [ ] **Final Code Cleanup and Documentation**
    - [ ] Remove dead code, experimental branches, unused files. [Details](#final-cleanup-code-details)
    - [ ] Ensure all code is well-commented/docstringed. [Details](#final-cleanup-comments-details)
    - [ ] Update all documentation (`README.md`, `GIT_GUIDELINES.md`, `docs/`) to reflect final state. [Details](#final-docs-update-details)
    - [ ] Ensure reproducibility (requirements, setup/train/eval instructions). [Details](#final-reproducibility-details)

- [ ] **Write Research Paper**
    - [ ] Draft paper (Intro, Related Work, Method, Experiments, Results, Ablations, Vis Engine, Conclusion). [Details](#final-paper-draft-details)
    - [ ] Include diagrams, visualizations, comparisons. [Details](#final-paper-content-details)
    - [ ] Refine and finalize paper. [Details](#final-paper-finalize-details)

- [ ] **Prepare for Submission (e.g., ARC Prize Kaggle)**
    - [ ] Package code according to competition rules (e.g., Docker). [Details](#final-submission-package-details)
    - [ ] Perform final test runs on the target evaluation platform. [Details](#final-submission-test-details)
    - [ ] Submit solution. [Details](#final-submission-action-details)

- [ ] **Open Source Release**
    - [ ] Add `LICENSE` file (e.g., MIT, Apache 2.0). [Details](#final-release-license-details)
    - [ ] Push final, cleaned code and documentation to public repository `main` branch. [Details](#final-release-push-details)
    - [ ] Create final release tag (e.g., `v1.0.0`). [Details](#final-release-tag-details)

---

# Detailed Specifications and Conditions

*(This section contains the detailed explanations, conditions, and technical specifics referenced by the checklist above. Fill in details as development progresses.)*

## Guiding Principles Details

### Modularity Principles
- **Definition:** Each major functional block (KAN processing, NTM variant, Token Summarization, DreamCoder components, Visualization Mapper) must be implemented as a distinct Python class or module.
- **Interface:** Modules must interact *only* through pre-defined Abstract Base Classes (ABCs) or clearly documented function signatures specified in designated interface files (e.g., `src/modules/memory_interface.py`).
- **Hot-Swapping:** It should be possible to switch between different implementations of an interface (e.g., swapping `TTMMemory` for `DNCMemory`) by changing only configuration files or initialization parameters in the main script, without modifying the core model logic that *uses* the module.
- **Testing:** Each module must have dedicated unit tests (`tests/test_module_name.py`) that test its functionality in isolation, mocking its dependencies if necessary. **Condition:** `pytest tests/test_module_name.py` passes for the module.

### Visualization Engine Principles
- **Real-time:** The visualization must update dynamically during training and inference with minimal performance impact. **Condition:** Target <15% overhead on training step time when visualization hooks are active compared to inactive. Measure using `timeit` or profiling.
- **Comprehensive:** Capture and visualize *all* relevant intermediate tensor states (activations, optionally gradients) from user-selected modules via hooks. **Condition:** `StateTracker` successfully captures data from designated KAN layers, NTM read/write heads, etc. during a test run.
- **Interactive Debugging:** Allow users to inspect tensor values (via hovering/selection), understand the computational graph flow, control visualization parameters (opacity, colormaps, slice indices) live via IMGUI panels, and navigate through time steps without writing new code. **Condition:** All core UI panels are functional and interactive during a live visualization session.
- **Extensible Mapping:** Easily add new `VisMapper` implementations for custom data types or visualization styles by subclassing `VisMapper` and registering with `MapperRegistry`. **Condition:** Adding a new dummy mapper requires minimal code changes outside the mapper class itself.
- **GPU Accelerated:** Leverage PyVista/VTK's capabilities for efficient rendering. **Condition:** Volume rendering of a moderately sized tensor (e.g., 64x64x64) maintains interactive framerates (>15 FPS) on target hardware.

### Git Workflow Principles
- **Consistency:** All collaborators must adhere strictly to the documented branching, commit message, PR, and merge strategies outlined in `GIT_GUIDELINES.md`. **Condition:** PRs failing to meet documented standards (e.g., poor description, wrong commit format) are rejected during review until fixed.
- **Clarity:** PRs must have clear descriptions linking to the relevant issue(s) or task(s) using GitHub's `#issue-number` syntax. Commit messages must follow the chosen convention. **Condition:** Reviewers check for linked issues and adherence to commit format.
- **Safety:** Direct pushes to `main` and `develop` branches must be disabled in GitHub repository settings. Merges require specified number of approvals and passing status checks (CI). Force-pushing is disallowed on shared branches (`main`, `develop`). **Condition:** GitHub branch protection rules are configured and active.
- **Recovery:** Collaborators demonstrate understanding of `git reflog`, `git reset`, `git revert` as documented in `GIT_GUIDELINES.md` when needed. **Condition:** Messy states are resolved following the documented procedures.

### Evaluation Strategy
- **Primary Metric:** Pass@2 on the official ARC AGI 2 evaluation sets. **Condition:** Evaluation script correctly calculates and reports Pass@2 scores.
- **Local Target:** Aim for ≥80% Pass@2 on a local validation set. **Condition:** This target is documented and tracked.
- **Ablations:** Systematically test the contribution of each major component. **Condition:** Ablation plan exists (`docs/ablation_plan.md`) and results are documented.
- **Efficiency:** Track computational cost. **Condition:** Training logs and evaluation reports include timing/cost metrics.

## Phase 0 Details

### init-repo-create-details
- **Condition:** Repository exists on GitHub. `git remote -v` shows an `origin` pointing to the correct GitHub URL.
- **Answer:** URL: `[Insert GitHub Repo URL here]`

### init-structure-details
- **Condition:** Running `ls -R` (or equivalent) shows the directories: `src/`, `data/` (empty or with `.gitignore`), `models/` (empty), `notebooks/`, `docs/`, `tests/`, `benchmarks/`. Root `.gitignore` file exists, potentially ignoring `data/*`, `models/*`, `*.pt`, `__pycache__/`, virtual env folders, etc.
- **Answer:** Directories listed: `src, data, models, notebooks, docs, tests, benchmarks`

### init-push-details
- **Condition:** The `main` branch on GitHub shows the initial commit with the project structure files (e.g., `.gitkeep`, `.gitignore`). `git status` locally shows `nothing to commit, working tree clean`.

### init-tracker-details
- **Condition:** A GitHub Project board is created and linked, or the issue template/labeling convention for tasks is defined in `GIT_GUIDELINES.md`.
- **Answer:** Link/Convention: `[Insert Link or Describe Convention]`

### git-guidelines-doc-details
- **Condition:** The file `GIT_GUIDELINES.md` exists in the repository root. Running `git log GIT_GUIDELINES.md` shows its commit history.

### git-branching-details
- **Condition:** `GIT_GUIDELINES.md` contains a "Branching Strategy" section detailing the purpose and lifecycle of `main`, `develop`, `feature/*`, `bugfix/*`, `experiment/*`, `hotfix/*`.
- **Answer:** Summary: `main` (releases), `develop` (integration), `feature` (dev), `experiment` (research), `bugfix` (fixes on develop), `hotfix` (fixes on main).

### git-pr-details
- **Condition:** `GIT_GUIDELINES.md` contains a "Pull Request Process" section detailing: descriptive titles/bodies, link issues, minimum # reviewer approvals (e.g., 1), required passing CI checks (e.g., linting, unit tests - placeholder).
- **Answer:** # Approvals Required: `1`; Designated Reviewers: `[List GitHub Usernames]`

### git-commit-convention-details
- **Condition:** `GIT_GUIDELINES.md` contains a "Commit Messages" section specifying the format (e.g., linking to `conventionalcommits.org`). Provides examples.
- **Answer:** Link: `https://www.conventionalcommits.org/`

### git-merge-strategy-details
- **Condition:** `GIT_GUIDELINES.md` contains a "Merging Strategy" section. Specifies strategy for `feature/*` -> `develop` and `develop` -> `main`.
- **Answer:** Strategy: `[e.g., Squash and Merge for features, Regular Merge commit for develop->main]`

### git-conflict-resolution-details
- **Condition:** `GIT_GUIDELINES.md` contains a "Handling Merge Conflicts" section with steps: `git checkout <feature-branch>`, `git fetch origin`, `git merge origin/develop` (or `rebase`), resolve conflicts using specified tool, test changes, `git add .`, `git commit`, `git push`.
- **Answer:** Recommended Tools: `[e.g., VS Code merge editor]`

### git-recovery-cleanup-details
- **Condition:** `GIT_GUIDELINES.md` contains a "Recovery and Cleanup" section explaining `git reflog`, `git reset --hard`, `git revert`, policy on force-pushing, and strategy for periodic code cleanup sprints (e.g., identifying and removing dead code).
- **Answer:** Force-push policy: `Allowed ONLY on personal feature branches AFTER coordinating with collaborators if necessary.`

### git-tagging-details
- **Condition:** `GIT_GUIDELINES.md` contains a "Tagging" section defining format for releases (e.g., `vX.Y.Z` following SemVer) and milestones (e.g., `milestone-phaseX-complete`). Explains `git tag -a <tagname> -m "message"` and `git push origin <tagname>`.
- **Answer:** Example Release Tag: `v0.1.0`; Milestone Tag: `milestone-phase0-complete`

### env-config-details
- **Condition:** `requirements.txt` (pip) or `environment.yml` (conda) exists, is committed. `cat requirements.txt` (or `environment.yml`) lists specific versions for `python`, `torch`, `torchvision`, `torchaudio`, `pyvista`, `numpy`, `pyvistaqt` (if used), `panel` (if used), `pykan` (if used), `dearpygui` (if used), `pytest`, `flake8`, etc.
- **Answer:** Python: `3.10.x`; PyTorch: `2.1.x+cu118`; PyVista: `0.4x.y`

### env-replication-details
- **Condition:** A collaborator successfully creates the environment using the committed file (`pip install -r requirements.txt` or `conda env create -f environment.yml`) and runs a simple script that imports `torch`, `pyvista`, `numpy` without errors.
- **Answer:** OS notes: `[e.g., On Windows, may need specific Visual C++ Redistributable for PyVista dependencies.]`

### hw-benchmark-script-details
- **Condition:** `benchmarks/hardware_test.py` exists. Can be run via `python benchmarks/hardware_test.py`. Script imports necessary libraries, defines benchmark functions for specified ops (e.g., `benchmark_matmul(size)`, `benchmark_kan_forward(model, input_shape)`, `benchmark_pv_volume_render(data)`), runs them on CPU and GPU (if `torch.cuda.is_available()`), prints timing (using `time.perf_counter`) and peak GPU memory (`torch.cuda.max_memory_allocated()`).
- **Answer:** Ops benchmarked: `Matrix Multiply (1kx1k, 4kx4k), KAN Layer (typical size) forward, PyVista Volume Render (64^3)`

### hw-benchmark-run-details
- **Condition:** A file like `benchmarks/RESULTS.md` exists and contains a table with columns: `Operation | GPU Model | CPU Time (ms) | GPU Time (ms) | GPU Peak Memory (MB)`. Results from running the script on all relevant team hardware are recorded.
- **Answer:** GPUs Tested: `[e.g., RTX 4080 SUPER, RTX 3090, A100]`

### hw-choice-doc-details
- **Condition:** `docs/hardware_choice.md` exists. States the primary GPU chosen for development and minimum required specs (e.g., VRAM amount). Justifies the choice based on benchmark results (e.g., "RTX 4080S chosen for best speed/VRAM balance among available team hardware"). Acknowledges potential differences on other hardware.
- **Answer:** Chosen GPU: `NVIDIA GeForce RTX 4080 SUPER`; Rationale: `Best performance/VRAM available to core team, meets estimated requirements for KAN+NTM size.`

## Phase 1 Details

### data-acquisition-arc1-details
- **Condition:** ARC AGI 1 dataset (from `https://github.com/fchollet/ARC`) is downloaded. The `data/training` and `data/evaluation` directories exist locally at the configured path.
- **Answer:** ARC1 Path Configured In: `src/config.py` as `ARC1_DATA_PATH`

### data-acquisition-arc2-details
- **Condition:** ARC AGI 2 datasets (Training, Public Eval from `https://arcprize.org/data`) are downloaded. The corresponding directories exist locally at the configured path.
- **Answer:** ARC2 Path Configured In: `src/config.py` as `ARC2_DATA_PATH`

### data-loader-interface-details
- **Condition:** `src/data/arc_dataloader.py` contains `class ARCDataset(torch.utils.data.Dataset):` which takes `source` and `task_ids` arguments. `__getitem__(self, idx)` returns one task dictionary in the format `{'id': str, 'train': [{'input': np.array, 'output': np.array}, ...], 'test': [{'input': np.array, 'output': np.array}, ...]}`. Grids are represented as NumPy arrays.
- **Answer:** Internal Representation Confirmed: Yes.

### data-merging-details
- **Condition:** `ARCDataset` constructor or a separate function handles a list of sources (e.g., `sources=['arc1_train', 'arc2_train']`). It correctly loads tasks from all specified sources and makes them accessible via `__getitem__` and `__len__`.
- **Answer:** Overlap Handling: `Tasks are assumed unique by filename/path across official datasets. No explicit deduplication implemented initially.`

### data-selection-details
- **Condition:** `ARCDataset` constructor accepts `source` (str or list) and optional `task_ids` (list of str). If `task_ids` is provided, only those tasks are loaded/accessible from the specified `source`(s).
- **Answer:** Selection Mechanism: `Constructor arguments 'source' and 'task_ids'.`

### data-tests-details
- **Condition:** `tests/test_data.py` contains `pytest` tests: `test_load_arc1_task`, `test_load_arc2_task`, `test_merged_dataset_length`, `test_select_by_source`, `test_select_by_ids`. Tests use sample task IDs known to exist. `pytest tests/test_data.py` passes.
- **Answer:** Scenarios Tested: `Load ARC1 train task, Load ARC2 public eval task, Verify len(merged_set) == len(set1) + len(set2), Verify loading only 'arc1_train' works, Verify loading specific task IDs works.`

## Phase 2 Details

### kan-library-setup-details
- **Condition:** `import pykan` succeeds OR KAN classes are present in `src/kan/`. `model = KAN([100, 50, 100])` (adjust dims) instantiates. `model(torch.randn(4, 100))` returns output tensor without error.
- **Answer:** Using: `pykan library` (or specify if custom)

### kan-gpu-test-details
- **Condition:** Code snippet like `model.to('cuda'); dummy_input.to('cuda'); start=time.perf_counter(); model(dummy_input); print(time.perf_counter()-start)` shows significantly lower time than CPU equivalent. `model.device` correctly reports `cuda`.
- **Answer:** GPU run confirmed: `Yes`

### kan-multkan-details
- **Condition:** `model = KAN([2, [2, 2], 1], n_m=[0, 1, 0])` instantiates. `model.plot()` shows a multiplication node (if visualization supported) or structure implies multiplication layer exists. `model(torch.randn(4, 2))` runs.
- **Answer:** Multiplication Node Specification: `Using pykan n_m list.`

### kan-kanpiler-details
- **Condition:** `import sympy; x,y = sympy.symbols('x y'); expr = sympy.sin(x)+y**2; model = pykan.kanpiler([x,y], expr)` runs. `model.plot()` shows structure corresponding to sin(x) and y**2 feeding into an addition node.
- **Answer:** Test successful: `Yes`

### kan-tree-converter-details
- **Condition:** For a KAN trained on `f(x,y,z,w)=g(x,y)+h(z,w)`, `model.tree()` (or equivalent) produces output indicating separation of `(x,y)` and `(z,w)`. E.g., Text output shows grouping, or plot shows distinct branches.
- **Answer:** Test successful: `Yes`

### kan-pruning-expansion-details
- **Condition:** Create a KAN `model = KAN([3, 5, 1])`. `model.prune_input([2])` results in `model.layer_widths[0] == 2`. `model.expand_width(1, 10)` changes `model.layer_widths[1] == 10`. `model.perturb(0.1)` changes model parameters slightly.
- **Answer:** Pruning test successful: `Yes`

### kan-attribution-details
- **Condition:** Function `calculate_kan_attribution(model, data)` is implemented. For a test case `f(x,y)=x`, the attribution score for input `x` should be significantly higher than for input `y`, whereas L1 norm might be similar if y activates initial layers strongly but is zeroed out later.
- **Answer:** Comparison shows attribution correctly identifies `x` as important, `y` as unimportant: `Yes`

### kan-base-model-arch-details
- **Condition:** Function `create_arc_kan_model(config)` in `src/models/kan_model.py` returns a `pykan.KAN` instance. Input `n_in` = 900 (for 30x30 grid flattened). Output `n_out` = 900. Hidden layers defined in `config`. Function handles grid flattening/reshaping.
- **Answer:** Initial Shape(s): `[900, 128, 128, 900]`; Input/Output Handling: `Flatten input grid (B, 30, 30) -> (B, 900); Reshape output (B, 900) -> (B, 30, 30)`

### kan-base-model-test-details
- **Condition:** `tests/test_kan.py` includes `test_arc_kan_forward`. Creates model via `create_arc_kan_model`, passes `torch.randn(4, 900)`, asserts output shape is `(4, 900)`. `pytest tests/test_kan.py` passes.
- **Answer:** Dummy Data: `torch.randn(4, 900)`

## Phase 3 Details

### ntm-interface-details
- **Condition:** `src/modules/memory_interface.py` exists. Defines `AbstractMemoryInterface(ABC, nn.Module)`. Includes abstract methods `@abstractmethod def read(self, memory_state, inputs)` returning `read_output`, `@abstractmethod def write(self, memory_state, processed_data, inputs)` returning `new_memory_state`, `@abstractmethod def initialize_memory(self, batch_size)` returning `initial_memory_state`. Type hints are used.

### ts-mlp-details
- **Condition:** `src/modules/token_summarization.py` contains `summarize_mlp(tokens, k)`. Returns shape `[batch, k, dim]`. `pytest tests/test_token_summarization.py::test_summarize_mlp` passes.
- **Answer:** MLP Architecture: `Linear(d, 128), ReLU, Linear(128, 1)` per token, weights computed, softmax, weighted sum.

### ts-query-details
- **Condition:** `summarize_query(tokens, k, query_vectors)` exists. Returns shape `[batch, k, dim]`. `pytest tests/test_token_summarization.py::test_summarize_query` passes.
- **Answer:** Query Vectors: `nn.Parameter(k, d), initialized kaiming_uniform_`

### ts-pooling-details
- **Condition:** `summarize_pooling(tokens, k, pool_type='avg')` exists. Returns shape `[batch, k, dim]`. `pytest tests/test_token_summarization.py::test_summarize_pooling` passes.
- **Answer:** Projection Layer: `Yes, Linear(d, d) after pooling.`

### ts-tests-details
- **Condition:** `pytest tests/test_token_summarization.py` passes all tests for MLP, Query, and Pooling summarizers, verifying output shapes for various input `p`, `k`, `d`.

### ntm-ttm-class-details
- **Condition:** `src/modules/ttm_memory.py` contains `class TTMMemory(AbstractMemoryInterface):`. `__init__` takes arguments like `memory_size`, `read_size`, `input_dim`, `summarization_type`.

### ntm-ttm-read-details
- **Condition:** `TTMMemory.read` implemented. Adds positional embeddings `self.pos_embed_read` (`nn.Embedding(m+n, d)`) to `torch.cat([memory_state, inputs], dim=1)`. Calls chosen summarizer (e.g., `summarize_mlp`) with `k=self.read_size`. Returns tensor of shape `[b, r, d]`.
- **Answer:** `r = 16`; Positional Embeddings: `nn.Embedding(m+n, d)`

### ntm-ttm-write-details
- **Condition:** `TTMMemory.write` implemented. Adds positional embeddings `self.pos_embed_write` (`nn.Embedding(m+r+n, d)`) to `torch.cat([memory_state, processed_data, inputs], dim=1)`. Calls chosen summarizer with `k=self.memory_size`. Returns tensor of shape `[b, m, d]`.
- **Answer:** `m = 96`; Summarization: `summarize_mlp`

### ntm-ttm-init-details
- **Condition:** `TTMMemory.initialize_memory` implemented. Uses `self.register_buffer('initial_memory', torch.zeros(1, self.memory_size, self.input_dim))`. Returns `self.initial_memory.expand(batch_size, -1, -1)`.
- **Answer:** Initialization: `Zero-initialized buffer`

### ntm-ttm-test-details
- **Condition:** `pytest tests/test_ttm_memory.py` passes, verifying `initialize_memory`, `read`, and `write` methods produce correct shapes and that the memory state tensor passed to `write` differs from the returned `new_memory_state`.

### ntm-dnc-class-details
- **Condition:** `src/modules/dnc_memory.py` contains `class DNCMemory(AbstractMemoryInterface):`. (Optional)

### ntm-dnc-addressing-details
- **Condition:** Implements calculation of read/write keys, strengths, content-based weights (cosine sim), allocation weighting, precedence weighting, and final read/write weights per head. (Optional)
- **Answer:** Addressing Details: `Cosine Similarity (Content), Temporal Links (Location)`

### ntm-dnc-read-details
- **Condition:** `DNCMemory.read` uses read weights to retrieve weighted sum of memory rows. (Optional)

### ntm-dnc-write-details
- **Condition:** `DNCMemory.write` uses write weights, erase vector (sigmoid), and add vector (tanh) to update memory matrix `M_t+1 = M_t * (1 - w_w * e) + w_w * a`. Updates usage vector. (Optional)

### ntm-dnc-allocation-details
- **Condition:** Implements calculation of usage vector based on read/write weights and free gates. Calculates allocation weights based on usage. Updates temporal link matrix. (Optional)

### ntm-dnc-test-details
- **Condition:** `pytest tests/test_dnc_memory.py` passes, verifying core addressing and memory update logic. (Optional)

### ntm-nar-class-details
- **Condition:** `src/modules/nar_memory.py` contains `class NARMemory(AbstractMemoryInterface):`. (Optional)

### ntm-nar-arch-details
- **Condition:** Implements the DNC memory module controlled by a Transformer (or KAN) processing unit, as interpreted from the NAR paper. (Optional)
- **Answer:** NAR Memory Description: `DNC memory module where read/write keys/vectors are generated by a separate KAN/Transformer controller.`

### ntm-nar-test-details
- **Condition:** `pytest tests/test_nar_memory.py` passes, verifying interactions and memory updates. (Optional)

### ntm-integration-details
- **Condition:** `src/models/main_model.py` `KAN_NTM_Model.__init__` takes `memory_variant='ttm'` config. Instantiates the correct `AbstractMemoryInterface` implementation. `forward` pass calls `self.memory.initialize_memory`, `read_data = self.memory.read(...)`, `processed = self.kan_processor(read_data)`, `self.memory_state = self.memory.write(...)`.
- **Answer:** Integration Points: `Initialize at start -> Read before KAN -> Process read data -> Write after KAN processing`.

## Phase 4 Details

### dc-dsl-details
- **Condition:** `src/dreamcoder/arc_dsl.py` defines functions like `def move_object(grid, obj_spec, dx, dy): ...`, `def recolor(grid, obj_spec, new_color): ...` etc. A list `ARC_DSL_PRIMITIVES` contains these functions or representations.
- **Answer:** Key Primitives: `GetObject, CopyObject, MoveObject, RecolorObject, RotateObject, ReflectObject, DrawLine, FillRectangle`

### dc-recognition-model-details
- **Condition:** `src/dreamcoder/recognition_model.py` defines `ARCRecognitionCNN(nn.Module)`. `forward(self, tasks)` takes a batch of tasks (e.g., list of dicts or padded tensor of examples) and outputs `log_probabilities` over the current DSL library (primitives + abstractions).
- **Answer:** Architecture: `CNN layers processing input/output grids -> Flatten -> MLP -> Output logits per DSL primitive`

### dc-generative-model-details
- **Condition:** `src/dreamcoder/generative_model.py` defines `Library` class storing `(primitive, log_probability)` pairs. `sample_program()` recursively samples primitives based on probabilities. `program_log_prob(program)` calculates total log prob.
- **Answer:** Probability Assignment: `Uniform initially, updated via MDL during abstraction (higher prob for reused abstractions)`

### dc-wake-phase-details
- **Condition:** `src/dreamcoder/wake_phase.py` `search_for_program` implements enumeration or search. Uses `recognition_model` to prioritize expansions. Uses `generative_model` for prior `P(program)`. Checks candidate programs by executing them on all task examples. Returns highest posterior `P(program | task)` solution(s).
- **Answer:** Search Algorithm: `Type-directed enumeration search with beam, guided by recognition network scores combined with generative prior.`

### dc-abstraction-phase-details
- **Condition:** `src/dreamcoder/abstraction_phase.py` `find_abstractions` takes programs found during wake. Uses techniques (e.g., common subexpression identification, possibly version space algebra from EC2) to find reusable fragments. Scores fragments by compression achieved (`ΔMDL = old_cost - new_cost`). Adds best fragment(s) to the library.
- **Answer:** Fragment Identification: `Common subexpression analysis on program ASTs.`

### dc-dreaming-phase-details
- **Condition:** `src/dreamcoder/dreaming_phase.py` `train_recognition_model` loops: Samples program `p` from `library`, generates task `t` by running `p`, runs `search_for_program(t)` to potentially find `p` or equivalent `p'`, adds `(t, p')` to training batch. Also adds `(task, program)` from wake phase replays. Performs optimizer step on recognition model using this batch.
- **Answer:** Fantasy Generation: `Sample program -> Execute on canonical inputs -> Synthesize task -> Train recognition model to predict program from task.`

### dc-integration-mechanism-details
- **Condition:** The main training loop (`train.py`) includes calls to `wake_phase.search_for_program` for each training task, `abstraction_phase.find_abstractions` periodically (e.g., every N epochs), and `dreaming_phase.train_recognition_model` periodically. The KAN+NTM model might potentially use the output program from DreamCoder as part of its input or configuration.
- **Answer:** Interaction: `DreamCoder runs independently initially to build a library for solving ARC tasks purely symbolically. KAN+NTM might later leverage DC's recognition model or library.` (Needs final decision)

### dc-tests-details
- **Condition:** `pytest tests/test_dreamcoder.py` passes tests covering: DSL execution, recognition model forward pass shape, generative model sampling/scoring, wake search on a trivial task, abstraction identifying a simple `(+ 1 x)` pattern, dreaming phase training step.

## Phase 5 Details

### vis-state-tracker-class-details
- **Condition:** `src/visualization/state_tracker.py` contains `class StateTracker:` with methods `add_state`, `get_states`, `clear_states`, `register_hooks`, `remove_hooks`. Internal storage `self.history` is a `defaultdict(list)`.

### vis-hook-registration-details
- **Condition:** `register_hooks` takes `model` and `module_filter_fn`. Iterates `model.named_modules()`. If `module_filter_fn(name, module)` returns `True`, attaches forward hook `module.register_forward_hook(partial(self._hook_fn, name, 'forward'))` and optionally backward hook. Stores handles in `self.hook_handles`.
- **Answer:** Module Selection: `lambda name, mod: isinstance(mod, (pykan.KANLayer, AbstractMemoryInterface, nn.MultiheadAttention))`

### vis-state-recording-details
- **Condition:** `_hook_fn(self, name, hook_type, module, input, output)` correctly captures `input` (tuple) and `output` (tensor or tuple), detaches, moves to CPU. Records state dict `{'id': uuid, 'step': self.current_step, 'module_name': name, ... 'data': tensor}` to `self.history[self.current_step]`. Handles tuples.
- **Answer:** Metadata Captured: `id, step, module_name, module_class, hook_type, io_type, shape, dtype, timestamp`

### vis-graph-extraction-details
- **Condition:** `build_computational_graph()` analyzes `self.history` to infer connections (e.g., output tensor ID from module A at step T matches input tensor ID for module B at step T or T+1). Returns nodes `[{'id': module_name, 'label': ...}]` and edges `[{'source': name1, 'target': name2}]`.
- **Answer:** Method: `Custom traversal connecting states based on module execution order and tensor identity (requires capturing tensor IDs or using object identity carefully).` Format: `NetworkX graph object or list of node/edge dicts.`

### vis-format-details
- **Condition:** `docs/visualization_format.md` exists and details the dictionary structure for states and the node/edge list format for the graph.
- **Answer:** Format Snippet: `State: {'id': ..., 'step': ..., 'module_name': ..., 'data': tensor(...), 'shape': ..., ...}`

### vis-mapper-interface-details
- **Condition:** `src/visualization/vis_mapper.py` defines `VisMapper(ABC)` with specified abstract methods. `get_ui_controls` should accept a context object (e.g., the UI builder instance) and return a list of UI element definitions.
- **Answer:** Return Types: `pv.StructuredGrid, pv.PolyData, pv.ImageData, pv.MultiBlock, pv.ChartXY`

### vis-tensor-mapper-details
- **Condition:** `TensorMapper` implemented. `map_to_pyvista` returns `pv.ChartXY` (1D), `pv.ImageData` (2D), `pv.StructuredGrid` (3D+). For 3D, sets up data for volume rendering. `get_ui_controls` returns widgets for opacity map editor (e.g., `plotter.add_volume_opacity_editor`), colormap selector.
- **Answer:** Volume Config: `Interactive opacity editor widget provided by PyVista/VTK, standard colormap dropdown.`

### vis-graph-mapper-details
- **Condition:** `GraphMapper` implemented. `map_to_pyvista` creates `pv.PolyData` with points (nodes) and lines (edges). Node scalars map to module types. `get_ui_controls` returns controls for node size, label visibility, layout algorithm selector.
- **Answer:** Node Positioning: `Initial force-directed layout (e.g., vtkForceDirectedLayoutStrategy), option for other layouts.`

### vis-other-mappers-details
- **Condition:** E.g., `ARCTaskMapper` exists that takes a raw ARC task dictionary and creates a `pv.MultiBlock` dataset with input/output grids visualized as `pv.ImageData`.

### vis-mapper-registry-details
- **Condition:** `MapperRegistry` implemented. `get_mapper(state_data)` uses `isinstance` on `state_data['data']` (if tensor) and checks `state_data['shape']` or looks for specific keys like `'graph_nodes'` to select appropriate mapper class from registered list.
- **Answer:** Selection Logic: `Priority list: Check for graph type, then tensor type/dims, then scalar, etc.`

### vis-engine-setup-details
- **Condition:** `VisualizationEngine` initializes `pyvistaqt.BackgroundPlotter` or `panel.pane.VTK`. `run()` shows the window. Background color is verified black. Basic camera interaction works.
- **Answer:** Integration: `pyvistaqt` (or `panel`)

### vis-dynamic-updates-details
- **Condition:** Engine's `_update` method retrieves states for `current_step`, clears actors related to dynamic states (not static UI), gets mappers via registry, calls `map_to_pyvista`, adds/updates actors using names (`plotter.add_mesh(..., name=state_id, overwrite=True)`).
- **Answer:** Update Strategy: `Overwrite actors using unique names per visualized state.`

### vis-imgui-integration-details
- **Condition:** Selected IMGUI library (`panel` widgets or `dearpygui`) is used to create dockable or laid-out windows/widgets alongside the PyVista rendering view.
- **Answer:** IMGUI Lib: `panel` (or `dearpygui`)

### vis-ui-panel-state-selector
- **Condition:** A panel displays a tree or list of hooked module names. Clicking a name selects it as the current item for visualization.

### vis-ui-panel-timeline
- **Condition:** A slider widget's range matches the number of recorded steps. Dragging the slider updates `engine.current_step` and triggers a scene update. Play/Pause/Step buttons exist.

### vis-ui-panel-vis-controls
- **Condition:** A panel area is designated. When a state is selected, this panel is populated by UI elements returned from `active_mapper.get_ui_controls()`. Interacting with these controls (e.g., changing colormap) updates the PyVista actor live.

### vis-ui-panel-graph
- **Condition:** A panel displays the computational graph from `GraphMapper`. Nodes are selectable, and selection highlights the corresponding module in the State Selector panel.

### vis-ui-panel-performance
- **Condition:** A panel displays text showing current FPS (calculated from `plotter.iren.fps_timer`) and memory usage (`torch.cuda.memory_summary()`).

### vis-ui-panel-edit
- **Condition:** (Future) Panel allows editing values of selected tensor elements. Button triggers callback to update data in `StateTracker` (requires careful implementation).

### vis-ui-panel-dataset
- **Condition:** (Future) Dropdown allows selecting active dataset (ARC1, ARC2, Merged). Triggers appropriate data loading/filtering.

### vis-picking-details
- **Condition:** `plotter.enable_cell_picking(...)` is active. The callback function identifies the picked actor/cell, finds the corresponding state data in `StateTracker` (e.g., via actor name or spatial query), and displays metadata (module name, step, shape, value at point) in a status bar or dedicated tooltip area.
- **Answer:** Picking Mech: `Cell Picking with custom callback.`

### vis-feed-integration-details
- **Condition:** `StateTracker` has a flag `is_active`. `VisualizationEngine._update` checks flag and calls `tracker.get_states(self.current_step)`. `train.py` toggles `tracker.is_active` based on `--visualize` flag and calls `tracker.increment_step()` per batch.
- **Answer:** Transfer Mech: `Engine polls tracker in its update loop.`

### vis-live-test-details
- **Condition:** Running `python train.py --visualize --epochs 1 --max_steps 50` shows the visualization window updating dynamically for 50 steps. Interaction with timeline and controls works as expected.
- **Answer:** Performance Impact: `[Measure % slowdown compared to run without --visualize]`

### vis-scalability-test-details
- **Condition:** Run visualization for a longer duration (e.g., 1 full epoch) or with a larger model. Monitor FPS in the Performance Panel. If FPS < target, use profiling (`cProfile`, `torch.profiler`) to identify bottlenecks.
- **Answer:** Target FPS: `20`; Bottlenecks: `[Identify specific slow functions/operations]`

### vis-demo-script-details
- **Condition:** `python run_visualization_demo.py` runs without errors. Loads sample data/states. Demonstrates selecting different modules, changing steps via timeline, interacting with dynamic controls (e.g., volume opacity), viewing the graph.
- **Answer:** Demo Highlights: `Live update simulation / Replay of captured states; Multi-view tensor visualization; Graph navigation; Interactive controls.`

## Phase 6 Details

### train-script-details
- **Condition:** `train.py` parses arguments using `argparse`. Includes args for model config path, dataset config path, NTM variant, DC enable/disable, learning rate, batch size, epochs, checkpoint dir, resume path, logging framework, visualize flag.

### train-integration-details
- **Condition:** `train.py` correctly imports and initializes KAN model, chosen NTM module, optional DreamCoder components, `ARCDataset`, AdamW optimizer, LR scheduler (e.g., warmup+cosine), and combined loss function based on parsed arguments/config files.

### train-loop-logic-details
- **Condition:** Main loop iterates epochs. Inner loop iterates batches from `DataLoader`. Calls `train_step` (includes `optimizer.zero_grad()`, `loss.backward()`, `clip_grad_norm_`, `optimizer.step()`, `scheduler.step()`). Calls `eval_step` periodically. Logs train/val loss and metrics using chosen logger (e.g., `wandb.log({...})`).
- **Answer:** Logging: `WandB`

### train-checkpointing-details
- **Condition:** Checkpoint dictionary includes `'epoch'`, `'model_state_dict'`, `'optimizer_state_dict'`, `'scheduler_state_dict'`, `'best_val_metric'`, optionally `'dreamcoder_library'`. Saving occurs every N epochs and when `val_metric` improves. Resuming correctly loads all these states.
- **Answer:** Save Frequency: `Every 5 epochs + Best validation Pass@2`

### train-early-stopping-details
- **Condition:** Training loop checks if `current_val_metric` hasn't improved vs `best_val_metric` for `config.early_stopping_patience` epochs. If so, `break` loop.
- **Answer:** Metric: `val_pass_at_2` (higher is better); Patience: `50`

### train-ada-details
- **Condition:** (Advanced) If implemented, code shows mechanism for adjusting computation (e.g., changing KAN active layers, NTM read/write frequency) based on task properties or internal model states.
- **Answer:** Strategy: `Not Implemented`

### eval-script-details
- **Condition:** `evaluate.py` exists. Parses args for checkpoint path, dataset source (`arc2_public_eval`, `arc2_semi_private_eval`), output results path, number of attempts (`k` for Pass@k).

### eval-inference-details
- **Condition:** Script loads model state dict. Sets `model.eval()`. Uses `torch.no_grad()`. Iterates specified dataset via `DataLoader`. For each task, calls `model(task['test'][0]['input'])` (potentially `k` times if stochastic or multiple outputs needed for Pass@k). Stores predictions.
- **Answer:** Eval Datasets: `arc2_public_eval` primarily.

### eval-metrics-details
- **Condition:** Script compares prediction(s) for each task to `task['test'][0]['output']`. Implements Pass@k check (correct if any of first `k` predictions match ground truth exactly). Calculates overall `% Pass@2`. Saves per-task pass/fail results to output file.
- **Answer:** Pass@2 Logic: `Model's primary output is attempt 1. If model can generate alternative, that's attempt 2. Check if prediction1 == target OR prediction2 == target.`

### eval-comparison-details
- **Condition:** (Optional) Evaluation script includes known scores for baselines (e.g., human average, o3 scores from ARC Prize website) for comparison in the output report.

### ablation-plan-details
- **Condition:** `docs/ablation_plan.md` exists. Contains table: `Experiment Name | Component Varied | Configuration | Dataset Subset | Metric | Status | Results Summary`.

### ablation-run-details
- **Condition:** Scripts allow running specific ablation configurations (e.g., `python train.py --config configs/ablation_no_ntm.yaml`). Results are logged to separate directories/WandB runs and summary updated in `docs/ablation_plan.md`.
- **Answer:** Key Findings Summary: `[Updated regularly as ablations complete]`

### refine-analysis-details
- **Condition:** Team meetings or issue discussions reference specific logs, evaluation reports, visualization screenshots, or ablation results to justify proposed changes.

### refine-iteration-details
- **Condition:** Significant changes (e.g., major architecture modification) are developed on `experiment/*` branches with clear goals. PRs reference the analysis motivating the change.

## Phase 7 Details

### final-performance-details
- **Condition:** Running `python evaluate.py --checkpoint <best_model.pt> --dataset arc2_public_eval` (or designated validation set) reports Pass@2 >= 80%.
- **Answer:** Final Score: `____% Pass@2` on `arc2_public_eval`

### final-cleanup-code-details
- **Condition:** Run static analysis tools (`flake8`, `mypy` optional). Review code manually for unused variables/functions/imports. Remove commented-out code blocks. Delete obsolete experiment branches.

### final-cleanup-comments-details
- **Condition:** All public functions/classes have docstrings explaining purpose, args, returns. Complex algorithms or non-obvious code sections have explanatory comments.

### final-docs-update-details
- **Condition:** `README.md` accurately reflects the final project state and features. Instructions in `docs/` for setup, training, evaluation, visualization are verified to work with the final codebase.

### final-reproducibility-details
- **Condition:** A new collaborator can clone the repo, create the environment from the requirements file, download data (if needed, based on docs), and successfully run `train.py` (short test run) and `evaluate.py` using documented commands.

### final-paper-draft-details
- **Condition:** A paper draft exists (`paper/main.tex` or shared doc). Includes all standard sections relevant to the work.

### final-paper-content-details
- **Condition:** Draft includes generated figures (model diagrams, result plots, visualizations), tables (results, ablations), and discusses related work, methodology, findings.

### final-paper-finalize-details
- **Condition:** Paper proofread, formatted for target venue (e.g., NeurIPS, ICML, arXiv), references checked. Ready for submission.

### final-submission-package-details
- **Condition:** If submitting to Kaggle, `submission.zip` or Dockerfile created according to ARC Prize 2025 rules. Includes necessary code, model checkpoint, and inference script.

### final-submission-test-details
- **Condition:** Packaged solution tested locally simulating the platform environment (e.g., build and run Docker container, run inference script within limits).

### final-submission-action-details
- **Condition:** Submission uploaded to the platform. Confirmation page or email received.

### final-release-license-details
- **Condition:** `LICENSE` file exists in the root directory containing the text of a chosen open-source license (e.g., MIT, Apache 2.0).

### final-release-push-details
- **Condition:** `git status` is clean on `main`. `git push origin main` completes successfully. Public repository reflects the final state.

### final-release-tag-details
- **Condition:** `git tag -a v1.0.0 -m "Stable release for ARC AGI 2 submission"` command executed. `git push origin v1.0.0` command executed successfully. Tag appears on GitHub releases page.
