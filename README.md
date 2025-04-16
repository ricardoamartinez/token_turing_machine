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
- **Definition:** Each major functional block (KAN processing, NTM variant, Token Summarization, DreamCoder components, Visualization Mapper) must be implemented as a distinct Python class or module within the `src/` directory (e.g., `src/models/kan/`, `src/modules/memory/`, `src/dreamcoder/`, `src/visualization/mappers/`).
- **Interface:** Modules must interact *only* through pre-defined Abstract Base Classes (ABCs) located in shared interface files (e.g., `src/modules/memory_interface.py`, `src/visualization/vis_mapper_interface.py`). Interfaces must use type hinting (`typing` module) for all arguments and return values. Docstrings must clearly explain the purpose and expected behavior of each interface method.
- **Hot-Swapping:** Configuration files (e.g., YAML using `pyyaml` or Python dicts in `src/config.py`) must allow specifying which implementation class to use for each interface (e.g., `memory_module: 'TTMMemory'` vs. `memory_module: 'DNCMemory'`). The main model/training script must dynamically import and instantiate the correct class based on this configuration. Switching implementations requires *only* changing the config value.
- **Testing:** Each module must have a corresponding test file in `tests/` (e.g., `tests/modules/memory/test_ttm_memory.py`). Tests must use `pytest` and `unittest.mock` (if needed) to test module logic in isolation. **Condition:** Running `pytest tests/modules/module_name/` passes with >= 90% line coverage for the module (measured using `pytest-cov`).

### Visualization Engine Principles
- **Real-time:** Measure training step time with hooks disabled vs. enabled using `time.perf_counter()` over ~100 steps. **Condition:** Average step time with hooks enabled is less than 1.15x the average time with hooks disabled. Optimize hook data capture (e.g., selective tensor cloning, non-blocking transfer if using multiprocessing).
- **Comprehensive:** Define a list of target module types/names for state capture (e.g., KAN layers, Attention heads, NTM read/write outputs). **Condition:** During a test run with visualization, the `StateTracker` history contains entries corresponding to *all* specified target modules for each executed step. Verify captured tensor shapes match expectations.
- **Interactive Debugging:** **Condition:** During a live visualization, the user can: (1) Select a module in the tree, see its tensor visualized. (2) Hover over a voxel/node, see correct metadata tooltip. (3) Drag the timeline slider, see the visualization update to the corresponding step. (4) Change a control in the Vis Controls panel (e.g., colormap), see the visualization update instantly.
- **Extensible Mapping:** **Condition:** Create a simple `DummyMapper(VisMapper)` that maps scalar data to a `pv.Sphere`. Register it. Run visualization with dummy scalar state. Verify the sphere appears and the registry selected the `DummyMapper`. This process should not require changes to the engine, registry, or other mappers.
- **GPU Accelerated:** **Condition:** Use `plotter.add_volume()` for 3D tensor rendering. Monitor GPU usage using `nvidia-smi` during visualization. Verify that rendering load is primarily on the GPU, not CPU. Maintain >15 FPS during interaction with a 64^3 volume render.

### Git Workflow Principles
- **Consistency:** All collaborators must adhere strictly to the documented branching, commit message, PR, and merge strategies outlined in `GIT_GUIDELINES.md`. **Condition:** PR reviews actively check for adherence to `GIT_GUIDELINES.md` (branch name, commit messages, description, linked issue). A PR checklist template is added to the repository (`.github/pull_request_template.md`) requiring confirmation of adherence.
- **Clarity:** **Condition:** All non-trivial PRs must link to at least one GitHub Issue using `Fixes #issue-number` or `Related #issue-number` in the description. Commit messages must follow the `<type>(<scope>): <subject>` format of Conventional Commits. Reviewers check for linked issues and adherence to commit format.
- **Safety:** **Condition:** In GitHub repo settings > Branches > Branch protection rules: `main` and `develop` require status checks to pass (placeholder for CI), require at least 1 approval, disallow force pushes, and optionally require linear history. Attempting a direct push fails. GitHub branch protection rules are configured and active.
- **Recovery:** **Condition:** A developer facing a complex merge conflict or accidental reset can successfully recover their intended state by following the documented steps for `git reflog`, `git reset`, interactive rebase, or cherry-picking as outlined in `GIT_GUIDELINES.md`. Messy states are resolved following the documented procedures.

### Evaluation Strategy
- **Primary Metric:** Pass@2 on official ARC AGI 2 eval sets. **Condition:** `evaluate.py` script calculates this metric by checking if `target == prediction_attempt_1 or target == prediction_attempt_2`.
- **Local Target:** ≥80% Pass@2 on ARC AGI 2 Public Eval set. **Condition:** This target is explicitly stated in the project goals/README, and evaluation runs track progress towards it.
- **Ablations:** `docs/ablation_plan.md` exists and lists specific experiments. **Condition:** For each completed ablation, the results (metric scores, charts) are documented and linked from the plan.
- **Efficiency:** Track cost per task evaluation ($) and/or inference time (ms). **Condition:** Evaluation script logs inference time; cost estimated based on hardware usage/API calls if applicable (relevant for ARC Prize).

## Phase 0 Details

### init-repo-create-details
- **Technical Details:** Create repository on GitHub. Clone locally using `git clone <repository-url>`.
- **Condition:** `git remote -v` output shows: `origin <repository-url> (fetch)` and `origin <repository-url> (push)`.
- **Answer:** URL: `[Paste Repository URL]`
- **Question:** Is the repository private or public initially?

### init-structure-details
- **Technical Details:** Create directories using `mkdir`. Add empty `.gitkeep` files using `touch <dir>/.gitkeep` to ensure empty directories are tracked by Git. Create a basic `.gitignore` file (e.g., from `gitignore.io` for Python).
- **Condition:** Running `tree -L 1` (or `ls`) shows `src`, `data`, `models`, `notebooks`, `docs`, `tests`, `benchmarks`. `cat .gitignore` shows relevant patterns like `*.pyc`, `__pycache__/`, `*.pt`, `data/`, `models/`.
- **Answer:** Directories listed: `src, data, models, notebooks, docs, tests, benchmarks`.
- **Git:** `git add .gitignore src/ data/ models/ notebooks/ docs/ tests/ benchmarks/`; `git commit -m "feat: Initial project structure and gitignore"`

### init-push-details
- **Condition:** Run `git push -u origin main`. Verify on GitHub that the `main` branch exists and contains the committed files. Local `git status` shows `Your branch is up to date with 'origin/main'. nothing to commit, working tree clean`.

### init-tracker-details
- **Condition:** GitHub Project board created (kanban style recommended). Link provided. OR `GIT_GUIDELINES.md` has section "Issue Tracking" defining labels like `bug`, `feature`, `enhancement`, `documentation`, `phase-X`, `priority-high/medium/low`.
- **Answer:** Link/Convention: `[Paste Link to Project Board or Describe Labeling]`
- **Question:** Who is responsible for maintaining the project board/triaging issues?

### git-guidelines-doc-details
- **Technical Details:** Create `GIT_GUIDELINES.md` in the root directory. Add sections for Introduction, Branching Strategy, Commit Messages, Pull Request Process, Merging Strategy, Handling Conflicts, Recovery, Tagging.
- **Condition:** File exists. `git add GIT_GUIDELINES.md`; `git commit -m "docs: Create initial GIT_GUIDELINES.md"`. PR created from `docs/git-guidelines` branch, reviewed, and merged.

### git-branching-details
- **Technical Details:** Write content for the "Branching Strategy" section in `GIT_GUIDELINES.md`. Explain `git checkout -b <branch_name>`, typical branch naming.
- **Condition:** Section exists and clearly defines the purpose of each main branch type.
- **Answer:** Summary: `main` (stable releases), `develop` (integration), `feature/<issue-id>-<desc>` (dev), `experiment/<idea>` (research), `bugfix/<issue-id>-<desc>` (fixes on develop), `hotfix/<issue-id>-<desc>` (fixes on main).

### git-pr-details
- **Technical Details:** Write content for "Pull Request Process" in `GIT_GUIDELINES.md`. Mention using GitHub UI. Specify required fields (description, linked issues), review process. Add `.github/pull_request_template.md`.
- **Condition:** Section and template exist. Template prompts for description, linked issues, testing done, checklist adherence.
- **Answer:** # Approvals Required: `1`; Initial Reviewers: `[List GitHub Usernames]`
- **Question:** Will automated CI checks (linting, testing) be added later as a requirement? (Yes)

### git-commit-convention-details
- **Technical Details:** Write "Commit Messages" section in `GIT_GUIDELINES.md`. Link to `conventionalcommits.org`. Provide examples: `feat: Add KAN base model`, `fix: Correct off-by-one error in dataloader`, `docs: Update README`, `test: Add unit tests for NTM read op`, `refactor: Simplify visualization state update`.
- **Condition:** Section exists, links to standard, provides examples.
- **Answer:** Link: `https://www.conventionalcommits.org/`

### git-merge-strategy-details
- **Technical Details:** Write "Merging Strategy" section in `GIT_GUIDELINES.md`. Explain options (Merge commit, Squash, Rebase). State the chosen strategy for merging PRs into `develop` and `develop` into `main`.
- **Condition:** Section clearly defines the strategy for different merge scenarios.
- **Answer:** Strategy: `Squash and Merge for feature/* -> develop; Merge commit for develop -> main; Merge commit for hotfix/* -> main & develop`.

### git-conflict-resolution-details
- **Technical Details:** Write "Handling Merge Conflicts" section. Detail steps using `git checkout <feature>`, `git fetch origin`, `git rebase origin/develop` (preferred over merge for cleaner history on features), resolve conflicts in editor, `git add <resolved_files>`, `git rebase --continue`, test, `git push --force-with-lease`. Alternative using `merge` also acceptable if documented.
- **Condition:** Step-by-step instructions provided for the chosen method (rebase preferred). Recommended tools mentioned.
- **Answer:** Recommended Tools: `VS Code (built-in 3-way merge editor)`

### git-recovery-cleanup-details
- **Technical Details:** Write "Recovery and Cleanup" section. Explain `git reflog` to find lost commits, `git reset --hard <commit>` (use with extreme caution), `git revert <commit>` (safer alternative). Define policy on `git push --force` (discouraged on shared branches, use `--force-with-lease` on own branches). Suggest periodic review of old branches (`git branch -a --merged develop | grep -vE '^\*|develop|main$' | xargs -n 1 git branch -d` cleans up merged branches (use with care).
- **Condition:** Section explains common recovery commands and force-push policy.
- **Answer:** Force-push policy: `Allowed ONLY on personal feature branches with --force-with-lease after ensuring no one else pulled the old state.`

### git-tagging-details
- **Technical Details:** Write "Tagging" section. Explain semantic versioning (`vMAJOR.MINOR.PATCH`). Show commands `git tag -a v0.1.0 -m "Phase 1 complete"` and `git push origin v0.1.0`. Explain milestone tags.
- **Condition:** Section defines formats and shows commands.
- **Answer:** Example Release Tag: `v0.1.0`; Milestone Tag: `milestone-phase0-complete`

### env-config-details
- **Technical Details:** Run `pip freeze > requirements.txt` or `conda env export > environment.yml`. Manually edit the file to pin major/minor versions of critical libraries (torch, pyvista, numpy, pykan, etc.) remove system-specific packages if using conda export. Add libraries needed for all phases: `torch`, `torchvision`, `torchaudio`, `pyvista[all]`, `pyvistaqt` (or `panel`), `numpy`, `pyyaml`, `pykan` (if using lib), `sympy` (for kanpiler), `networkx` (for vis graph), `pytest`, `pytest-cov`, `flake8`, `wandb` (or `tensorboard`), `imgui` (if using `dearpygui` or `imgui-bundle`), `jupyter`.
- **Condition:** File exists, is committed. Contains specific versions like `torch==2.1.0`, `pyvista==0.42.3`, `numpy==1.26.0`, `pykan==...`. Ensure `pyvista[all]` is used for optional dependencies like `pyvistaqt`/`panel`.
- **Answer:** Python: `~3.10`; PyTorch: `~2.1+cu118`; PyVista: `~0.42`

### env-replication-details
- **Condition:** Collaborator checks out repo, runs `python -m venv venv; source venv/bin/activate; pip install -r requirements.txt` (or conda equivalent), runs `python -c "import torch; import pyvista; import numpy; print(torch.__version__)"`. Command succeeds and shows expected versions.
- **Answer:** OS notes: `[Confirm if any platform-specific build tools or libraries are needed for dependencies like VTK/OpenGL. E.g., Linux needs 'sudo apt-get install libgl1-mesa-glx libxrender1']`

### hw-benchmark-script-details
- **Technical Details:** Create `benchmarks/hardware_test.py`. Use `torch.randn` for data. Use `@profile` decorator (if using `line_profiler` or similar) or `time.perf_counter()` blocks. Include functions testing `torch.matmul`, `kan_layer.forward`, basic NTM read/write ops (using dummy memory module), `pyvista.Plotter().add_volume`. Use `torch.cuda.synchronize()` before stopping timers for GPU ops. Print results clearly, label units (ms, MB).
- **Condition:** Script runs via `python benchmarks/hardware_test.py`. Output includes timings in ms and peak GPU memory in MB for each benchmarked operation on both CPU and GPU (if available).
- **Answer:** Ops benchmarked: `MatMul(1k,4k), KAN Layer(128,128), Dummy NTM Read(m=96,n=100,d=128), PV VolumeRender(64^3)`

### hw-benchmark-run-details
- **Condition:** `benchmarks/RESULTS.md` exists and contains a Markdown table populated with results from running the script on relevant GPUs. Table is formatted clearly.
- **Answer:** GPUs Tested: `[List GPUs, e.g., RTX 4080 SUPER, RTX 3090, A100]`

### hw-choice-doc-details
- **Condition:** `docs/hardware_choice.md` exists. Contains "Chosen Hardware" and "Rationale" sections. Clearly states the primary dev GPU model and minimum required specs (e.g., VRAM amount). Rationale references specific performance numbers from `benchmarks/RESULTS.md`.
- **Answer:** Chosen GPU: `NVIDIA GeForce RTX 4080 SUPER (16GB)`; Rationale: `Best performance/VRAM available to core team, meets estimated requirements for KAN+NTM size based on benchmarks (e.g., KAN layer ~X ms, NTM ops ~Y ms).`

## Phase 1 Details

### data-acquisition-arc1-details
- **Technical Details:** Clone `https://github.com/fchollet/ARC`. Configure `ARC1_DATA_PATH` in `src/config.py` to point to the cloned `/data` directory. Add `data/` (or the specific path) to `.gitignore`.
- **Condition:** `os.path.exists(config.ARC1_DATA_PATH + '/training')` is `True`. Running `git status` does not show the ARC1 data directory as untracked/modified.
- **Answer:** ARC1 Path Configured In: `src/config.py` as `ARC1_DATA_PATH`
- **Question:** Should we store data centrally or expect each dev to download? (Assume each dev downloads, path configured).

### data-acquisition-arc2-details
- **Technical Details:** Download zip files from `https://arcprize.org/data`. Unzip. Configure `ARC2_DATA_PATH` in `src/config.py` to point to the directory containing `arc-agi_training`, `arc-agi_evaluation_public`. Add path to `.gitignore`.
- **Condition:** `os.path.exists(config.ARC2_DATA_PATH + '/arc-agi_training')` is `True`. Running `git status` does not show the ARC2 data directory.
- **Answer:** ARC2 Path Configured In: `src/config.py` as `ARC2_DATA_PATH`

### data-loader-interface-details
- **Technical Details:** Implement `ARCDataset` in `src/data/arc_dataloader.py` inheriting `torch.utils.data.Dataset`. `__init__` parses specified source paths, loads task JSON files (using `json` module), stores them in `self.tasks` (list of loaded task dicts). `__len__` returns `len(self.tasks)`. `__getitem__` returns the task dict, converting grids to `np.array(..., dtype=int)`. Extract task ID from filename.
- **Condition:** `dataset = ARCDataset(source='arc1_training'); task = dataset[0]` returns a dictionary with keys `'id'` (str, filename), `'train'` (list of dicts), `'test'` (list of dicts). Grids are `np.ndarray` with `dtype=int`.
- **Answer:** Internal Representation Confirmed: `{id: str, train: [{'input': np.ndarray, 'output': np.ndarray}], test: [{'input': np.ndarray, 'output': np.ndarray}]}`
- **Question:** How to handle potentially different grid sizes within a batch later? (Need custom collate_fn for DataLoader).

### data-merging-details
- **Technical Details:** Modify `ARCDataset.__init__` to accept `source` as a list (e.g., `['arc1_train', 'arc2_train']`). Loop through sources, determine full path based on source string (e.g., `config.ARC1_DATA_PATH + '/training'` if 'arc1_train'), load all JSONs from that path, append to `self.tasks`.
- **Condition:** `d1 = ARCDataset(source='arc1_train'); d2 = ARCDataset(source='arc2_train'); d_merged = ARCDataset(source=['arc1_train', 'arc2_train'])`. `len(d_merged)` equals `len(d1) + len(d2)`. Inspecting `d_merged.tasks` shows tasks from both origins.
- **Answer:** Overlap Handling: `Assumes task filenames are unique identifiers across datasets. No explicit content-based deduplication implemented initially.`

### data-selection-details
- **Technical Details:** In `ARCDataset.__init__`, after loading all tasks for the specified `source`(s), if `task_ids` argument is not `None`, filter `self.tasks = [t for t in self.tasks if t['id'] in task_ids]`. Ensure `t['id']` extraction from filename is robust.
- **Condition:** `dataset = ARCDataset(source='arc1_training', task_ids=['007bbfb7.json', '00d62c1b.json'])`. `len(dataset)` is 2. `dataset[0]['id']` and `dataset[1]['id']` match the provided IDs (allow for order difference).
- **Answer:** Selection Mechanism: `Constructor arguments 'source' (str or list) and 'task_ids' (list or None).`

### data-tests-details
- **Technical Details:** Create `tests/fixtures/data` with minimal example ARC1/ARC2 JSON task files. Use `pytest.mark.parametrize` to test loading different sources and IDs.
- **Condition:** `pytest tests/test_data.py` passes. Tests cover loading single ARC1 task, single ARC2 task, verifying merged dataset length, selecting only ARC1 source, selecting only specific ARC2 task IDs using fixture data. Test that grid data types are `int`.
- **Answer:** Scenarios Tested: `Load ARC1 task by ID, Load ARC2 task by ID, Merged length check, Source filtering, ID filtering, Grid dtype check.`

## Phase 2 Details

### kan-library-setup-details
- **Technical Details:** Add `pykan` to `requirements.txt`. Run `pip install pykan`. Use `from pykan import KAN`.
- **Condition:** `import pykan` succeeds. `model = KAN([100, 50, 100])` instantiates. `output = model(torch.randn(4, 100))` returns tensor of shape `(4, 100)`.
- **Answer:** Using: `pykan library`
- **Question:** Which version of pykan? Should we fork it if modifications are needed? (Start with official pip version).

### kan-gpu-test-details
- **Technical Details:** Ensure CUDA toolkit and compatible PyTorch version are installed. Test script: `model = KAN(...).to('cuda'); inp = torch.randn(..., device='cuda'); start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True); start.record(); out = model(inp); loss=out.mean(); loss.backward(); end.record(); torch.cuda.synchronize(); print(f"GPU time: {start.elapsed_time(end)} ms")`. Compare with CPU timing.
- **Condition:** `torch.cuda.is_available()` is `True`. GPU forward/backward time is significantly faster (>5x) than CPU. No CUDA errors.
- **Answer:** GPU run confirmed: `Yes`

### kan-multkan-details
- **Technical Details:** Use `KAN(..., n_m=[...])` syntax. Example `n_m=[0, 1, 0]` for `[in, hidden, out]` means hidden layer has 1 multiplication node, output layer has 0. The `pykan` implementation handles the `Ml` layer internally.
- **Condition:** `model = KAN([2, [2, 2], 1], n_m=[0, 1, 0])` instantiates. `model(torch.randn(4, 2))` executes. `model.plot()` (if supported for MultKAN) visualizes the structure correctly OR inspecting `model.layers[1].shapes` reflects the combined addition/multiplication width if accessible.
- **Answer:** Multiplication Node Specification: `Using pykan n_m list argument during KAN initialization.`
- **Question:** Does `pykan`'s visualization fully support MultKAN diagrams? (Check `pykan` docs/issues).

### kan-kanpiler-details
- **Technical Details:** `pip install sympy`. Use `pykan.kanpiler`. Ensure input variables are `sympy.Symbol` objects.
- **Condition:** `import sympy; from pykan import kanpiler; x, y = sympy.symbols('x y'); expr = sympy.sin(x) + y**2; model = kanpiler([x, y], expr)` returns a `pykan.KAN` instance. `model.plot()` generates a graph showing structure: x -> sin -> +, y -> sq -> +.
- **Answer:** Test successful: `Yes`

### kan-tree-converter-details
- **Technical Details:** Requires `graphviz` installed (`pip install graphviz`, may need system package `sudo apt-get install graphviz`). Use `model.tree()` method. It calculates Hessians internally.
- **Condition:** For a KAN model (either compiled or trained) representing `f(x,y,z,w)= (x+y)**2 + z*w`, `model.tree(style='box')` returns a graphviz object that, when rendered (`graph.render('tree')`), shows boxes grouping `(x,y)` and `(z,w)` separately before a final combination step. Test requires a suitable KAN model.
- **Answer:** Test successful: `Yes` (Requires trained/compiled model)
- **Question:** How robust is tree conversion to KAN training noise/approximation? (KAN 2.0 paper notes caveats).

### kan-pruning-expansion-details
- **Technical Details:** Use documented methods: `model.prune_input(input_ids_to_prune)`, `model.expand_width(layer_index, new_width)`, `model.expand_depth(layer_index, new_nodes)`, `model.perturb(magnitude)`. Check layer dimensions and parameter changes.
- **Condition:** Create `model = KAN([3, 5, 1])`. `model.prune_input([2])`. Assert `model.layer_widths == [2, 5, 1]`. `model.expand_width(1, 7)`. Assert `model.layer_widths == [2, 7, 1]`. Capture parameter state before/after `model.perturb(0.01)` and assert inequality.
- **Answer:** Pruning test successful: `Yes`

### kan-attribution-details
- **Technical Details:** Implement function `calculate_kan_attribution(model, dataloader)` per KAN 2.0 paper Sec 4.1 (Eq. 9). Requires a forward pass over `dataloader` to compute activation stats (`Nl,i` = stddev of node activation, `El,i,j` = stddev of edge activation pre-summation). Compute `Al,i` and `Bl,i,j` iteratively backwards.
- **Condition:** Function returns array `A0` of size `n_in`. Test with `f(x,y)=x` KAN: `A0[0]` >> `A0[1]`. Test with `f(x,y)=x+y` KAN: `A0[0]` approx `A0[1]`. Values differ from simple L1 norm of first layer activations.
- **Answer:** Comparison shows attribution correctly identifies active vs inactive paths: `Yes`
- **Question:** How sensitive is this to the data distribution in the dataloader used for stats? (Likely needs representative data).

### kan-base-model-arch-details
- **Technical Details:** Create `src/models/kan_model.py`. Function `create_arc_kan_model(config)` reads `config['model']['kan']['layer_widths']`, `grid_size`, `k`, `input_channels` (e.g., 10 if one-hot color), `output_channels`. Input handling: `x (B, H, W)` -> maybe `nn.Conv2d` pre-processing -> `nn.Flatten(start_dim=1)` -> KAN. Output handling: KAN output `(B, H_out*W_out*C_out)` -> `view(B, C_out, H_out, W_out)` -> maybe `nn.ConvTranspose2d` post-processing or direct argmax for color prediction.
- **Condition:** Function returns instantiated `pykan.KAN` (or wrapper `nn.Module`). Handles input tensor `(B, H, W)` or `(B, C, H, W)` and returns output tensor suitable for ARC grid prediction (e.g., `(B, H_out, W_out)` with integer class per pixel or `(B, num_colors, H_out, W_out)` logits).
- **Answer:** Initial Shape(s): `[Needs refinement based on Input/Output handling]`; Input/Output Handling: `Input: (B, H, W) -> OneHot(10 colors) -> (B, 10, H, W) -> Conv2D layers -> Flatten -> KAN -> Output: (B, H_out*W_out*10) -> Reshape(B, 10, H_out, W_out) -> Argmax(dim=1) -> (B, H_out, W_out)`. This is just one possibility.
- **Question:** Should we use convolutions before/after KAN, or feed flattened grids directly? How to handle variable grid sizes? (Requires padding or more complex architecture).

### kan-base-model-test-details
- **Condition:** `tests/test_kan.py::test_arc_kan_forward` exists. Instantiates model with typical config (e.g., 30x30 grids, 10 colors). Passes dummy input `torch.randint(0, 10, (4, 30, 30))` (or pre-processed shape). Asserts output tensor has the final expected grid shape (e.g., `(4, 30, 30)`). `pytest tests/test_kan.py` passes.
- **Answer:** Dummy Data: `torch.randint(0, 10, (4, 30, 30))`

## Phase 3 Details

### ntm-interface-details
- **Technical Details:** Create file `src/modules/memory/memory_interface.py`. Use `from abc import ABC, abstractmethod`, `import torch`, `import torch.nn as nn`. Define `class AbstractMemoryInterface(ABC, nn.Module):`. Define abstract methods with precise type hints.
- **Condition:** File exists. Class and methods defined. `python -m py_compile src/modules/memory/memory_interface.py` runs without error.
- **Question:** Should the interface support multiple read/write heads? (Yes, DNC/NAR might. Inputs/outputs should perhaps be lists of tensors or have a head dimension). Let's refine signatures: `read(self, memory_state, read_inputs) -> List[torch.Tensor]`, `write(self, memory_state, write_inputs) -> torch.Tensor`. `initialize_memory(self, batch_size) -> torch.Tensor`. Inputs need clearer definition based on variant.

### ts-mlp-details
- **Technical Details:** Implement `summarize_mlp` in `src/modules/token_summarization.py`. Use `nn.Module` to hold MLP layers for better parameter management. Input `tokens` (B, P, D). Output `(B, k, D)`.
- **Condition:** Function/Class exists. `pytest tests/test_token_summarization.py::test_summarize_mlp` passes.
- **Answer:** MLP Architecture: `nn.Sequential(nn.Linear(D, 128), nn.ReLU(), nn.Linear(128, k))` applied to each input token's features independently is NOT correct. TTM computes weights FROM tokens. Correct: MLP takes tokens (B, P, D), outputs weights (B, P, k). `MLP = nn.Sequential(nn.Linear(D, 128), nn.ReLU(), nn.Linear(128, k))`. `weights = MLP(tokens) # (B, P, k)`. `weights = F.softmax(weights / temperature, dim=1)`. `output = torch.einsum('bpk,bpd->bkd', weights, tokens)`.
- **Question:** Temperature for softmax? (Add as argument, default 1.0).

### ts-query-details
- **Technical Details:** Implement `SummarizeQuery(nn.Module)` holding `self.query_vectors = nn.Parameter(torch.randn(k, D))`. `forward(self, tokens)` performs bmm attention as described before.
- **Condition:** Class exists. `pytest tests/test_token_summarization.py::test_summarize_query` passes.
- **Answer:** Query Vectors: `nn.Parameter(k, D)`, initialized `nn.init.xavier_uniform_`.
- **Question:** Should queries be learned per instance or shared? (Shared `nn.Parameter` is standard).

### ts-pooling-details
- **Technical Details:** Implement `SummarizePooling(nn.Module)` with `pool_type`, `k`, `projection`. Use `einops.rearrange` and `einops.reduce`. `tokens_grouped = rearrange(tokens, 'b (k group_size) d -> b k group_size d', k=self.k)`. `pooled = reduce(tokens_grouped, 'b k group_size d -> b k d', reduction=self.pool_type)`. Apply optional linear projection. Handle non-divisible P using padding or adaptive pooling.
- **Condition:** Class exists. `pytest tests/test_token_summarization.py::test_summarize_pooling` passes.
- **Answer:** Projection Layer: `Yes, nn.Linear(D, D)`. Pooling: `einops`. Non-divisible handling: Pad P to nearest multiple of k before rearrange.

### ts-tests-details
- **Condition:** `pytest tests/test_token_summarization.py` passes all tests.

### ntm-ttm-class-details
- **Technical Details:** Create `src/modules/memory/ttm_memory.py`. `class TTMMemory(AbstractMemoryInterface):`. `__init__` stores args. Instantiates `self.pos_embed_read = nn.Embedding(...)`, `self.pos_embed_write = nn.Embedding(...)`, and the chosen summarization modules/functions (e.g., `self.read_summarizer = SummarizeQuery(k=read_size, D=input_dim)`).
- **Condition:** File/Class exist. Instantiation works.

### ntm-ttm-read-details
- **Technical Details:** Implement `read`. Input `memory_state (B, M, D)`, `inputs (B, N, D)`. Create type embeddings (B, M, D) for memory, (B, N, D) for input (e.g., using two `nn.Parameter(1, 1, D)` expanded). Add type embeddings. Concatenate `combined = torch.cat([mem_typed, inp_typed], dim=1)`. Add positional embeddings `pos_embed = self.pos_embed_read(...)`. Call `read_output = self.read_summarizer(combined + pos_embed)`.
- **Condition:** Method returns `(B, self.read_size, D)`. Uses distinct type embeddings and shared positional embeddings.
- **Answer:** `r = 16`; Positional/Type Embeddings: Separate learnable type embeddings for memory/input added first, then shared positional embedding for M+N positions added.

### ntm-ttm-write-details
- **Technical Details:** Implement `write`. Inputs `memory_state (B, M, D)`, `processed_data (B, R, D)`, `inputs (B, N, D)`. Create type embeddings for memory, processed, input. Add type embeddings. Concatenate `combined = torch.cat([mem_typed, proc_typed, inp_typed], dim=1)`. Add positional embeddings `pos_embed = self.pos_embed_write(...)` for M+R+N positions. Call `new_memory_state = self.write_summarizer(combined + pos_embed)`.
- **Condition:** Method returns `(B, self.memory_size, D)`. Uses distinct type embeddings and shared positional embeddings.
- **Answer:** `m = 96`; Summarization: `SummarizeQuery` (Configurable). Type embeddings used.

### ntm-ttm-init-details
- **Technical Details:** Implement `initialize_memory`. Use `self.register_buffer`.
- **Condition:** Returns `(batch_size, self.memory_size, D)`.
- **Answer:** Initialization: `Zero-initialized non-trainable buffer.`

### ntm-ttm-test-details
- **Condition:** `pytest tests/test_ttm_memory.py` passes, verifying `initialize_memory`, `read`, and `write` methods produce correct shapes and that the memory state tensor passed to `write` differs from the returned `new_memory_state`.

### ntm-dnc-class-details - ntm-dnc-test-details
- **Condition:** (Optional) All implementation and tests completed if DNC variant is pursued.

### ntm-nar-class-details - ntm-nar-test-details
- **Condition:** (Optional) All implementation and tests completed if NAR variant is pursued.

### ntm-integration-details
- **Condition:** Main model (`src/models/main_model.py`) can be configured via YAML/dict to use `TTMMemory`, `DNCMemory`, or `NARMemory`. The `forward` pass correctly sequences `initialize_memory`, `read_data = self.memory.read(...)`, `processed = self.kan_processor(read_data)`, `self.memory_state = self.memory.write(...)`.
- **Answer:** Integration Points: Confirmed sequence `init -> read -> process -> write`.

## Phase 4 Details

### dc-dsl-details
- **Technical Details:** Define primitives in `src/dreamcoder/arc_dsl.py`. Use type hints for arguments (e.g., `grid: np.ndarray`, `color: int`). Include error handling (e.g., what if `GetObject` finds no object?).
- **Condition:** File exists. Core primitives implemented and individually testable via `pytest`. `ARC_DSL_PRIMITIVES` list defined.
- **Answer:** Key Primitives: `GetObjectByColorSize, CopyObject, MoveObject, RecolorObject, RotateObject, ReflectObject, DrawLine, FillRectangle, ComposeGrids(Overlay/Tile), GetColor, GetSize, CountObjects`
- **Question:** Need a canonical representation for programs (ASTs? Lambda calculus?). Use `Parsita` or `funcparserlib` for parsing if needed? (Start with Python ASTs).

### dc-recognition-model-details
- **Technical Details:** Implement `ARCRecognitionCNN` in `src/dreamcoder/recognition_model.py`. Preprocessing: pad grids to 30x30, stack `(input, output)` pairs along channel dim (e.g., `(B, num_examples * 2, 30, 30)`), maybe one-hot colors `(B, num_examples * 2 * 10, 30, 30)`. Pass through CNN -> Pool -> MLP -> Logits `(B, num_dsl_primitives)`.
- **Condition:** Class exists. Forward pass handles list of tasks (variable examples) and returns `(B, num_primitives)` logits. `pytest tests/test_dreamcoder.py::test_recognition_model_forward` passes.
- **Answer:** Architecture: `Variable examples -> Pad/Stack -> CNN Encoder -> Attentional Pooling over examples -> MLP -> Logits`
- **Question:** How to represent the target program for loss calculation? (Indices of primitives used? One-hot vector?). Use simple primitive indices.

### dc-generative-model-details
- **Technical Details:** Implement `Library` in `src/dreamcoder/generative_model.py`. Store primitives/abstractions with associated log probabilities. `sample_program` needs to handle function types and arguments recursively. `program_log_prob` traverses the program AST and sums log probs. Need a type system.
- **Condition:** Class exists. Can add abstractions. Sampling produces typed programs. Scoring works. Sum of probabilities for choices at any node type is 1.
- **Answer:** Probability Assignment: `MDL-based: log_prob ~ -complexity - usage_count`. Need normalization.
- **Question:** Implement full polymorphic type system or simplified version? (Start simple, maybe just base types like grid, int, color, list[coord]).

### dc-wake-phase-details
- **Technical Details:** Implement `search_for_program` using best-first search. Priority queue stores `(score, program_ast)`. Score is `log P(program | library) + log P(primitives | task)`. Expand node by applying available primitives respecting types. Check validity against task examples.
- **Condition:** Function finds simple known program for test task within timeout.
- **Answer:** Search Algorithm: `Best-first enumeration with neural guidance.`
- **Question:** Search depth limit? Beam width?

### dc-abstraction-phase-details
- **Technical Details:** Implement `find_abstractions`. Iterate found `programs` (ASTs). Identify common subtrees. Evaluate candidate `abstraction` by re-writing programs using it and calculating `ΔMDL = sum(old_logP) - sum(new_logP) - C`. Add best abstraction(s) to library and update probabilities (renormalize).
- **Condition:** Identifies and abstracts simple common pattern `(lambda x: (+ 1 x))` from `[(+ 1 5), (+ 1 7)]`. Library probabilities updated.
- **Answer:** Fragment Identification: `AST common subexpression.` Complexity `C` = 1.0 (tunable).

### dc-dreaming-phase-details
- **Technical Details:** Implement `train_recognition_model`. Loop `num_dreams`: Sample `p`, generate `t`. Get training pairs `(t, p)` from dreams and replays. Compute loss: For each step in target program `p`, calculate cross-entropy between recognition model logits (given `t`) and the target primitive used at that step. Average loss over program steps/batch. Apply optimizer step.
- **Condition:** Function runs one training step. Recognition model parameters are updated. Loss value is reasonable.
- **Answer:** Fantasy Generation: `Sample program -> Execute on random 10x10 grid -> Task`. Loss: Cross-entropy over primitives at each program step.

### dc-integration-mechanism-details
- **Technical Details:** Implement chosen option (DC Guides). KAN+NTM model's `forward` method accepts optional `dsl_program_embedding`. If provided, this embedding (e.g., from GNN on DSL AST) conditions attention layers or memory interface operations. `train.py` calls DC search first, then passes embedding to model.
- **Condition:** KAN+NTM model signature updated. Training script includes DC search call. Tested that providing embedding changes model output.
- **Answer:** Interaction: `Option 2 (DC Guides)`: DC program AST embedded using Graph Neural Network, embedding vector passed as additional input to KAN+NTM model.

### dc-tests-details
- **Technical Details:** Implement `pytest tests/test_dreamcoder.py` to cover all sub-tests for DSL, recognition, generation, wake, abstraction, dreaming.
- **Condition:** All sub-tests pass.
- **Answer:** Test Coverage: All sub-tests pass.

## Phase 5 Details

### vis-state-tracker-class-details
- **Technical Details:** Implement `StateTracker` class with `self.history` as `defaultdict(lambda: defaultdict(list))` to store `history[step][module_name] = [state_dict, ...]`. Has methods `add_state`, `get_states`, `clear_states`, `register_hooks`, `remove_hooks`.
- **Condition:** `self.history` populated correctly during run. Contains all specified metadata.
- **Answer:** Metadata Captured: `id, step, module_name, module_class, hook_type, io_type, shape, dtype, timestamp, tensor_data`

### vis-hook-registration-details
- **Technical Details:** Use `functools.partial` to pass module name and hook type to `_hook_fn`. Store handles in `self.hook_handles: List[torch.utils.hooks.RemovableHandle]`. `remove_hooks` calls `h.remove()` for h in handles and clears list.
- **Condition:** Hooks attach/detach correctly. `module_filter_fn` works.
- **Answer:** Module Selection: `lambda name, mod: isinstance(mod, (pykan.KANLayer, AbstractMemoryInterface, nn.MultiheadAttention))`

### vis-state-recording-details
- **Technical Details:** Implement `_hook_fn` signature `(self, module_name, hook_type, io_type, module, args, output)`. Handle `args` (input tuple) and `output`. Use `copy.deepcopy` for mutable metadata if needed. Store tensors after `.clone().detach().cpu()`. Add `timestamp = time.time()`.
- **Condition:** `self.history` populated correctly during run. Contains all specified metadata.
- **Answer:** Metadata Captured: `id, step, module_name, module_class, hook_type, io_type, shape, dtype, timestamp, tensor_data`

### vis-graph-extraction-details
- **Technical Details:** Implement `build_computational_graph`. Option 2 (Sequential Assumption) is more feasible. Maintain `last_hooked_module_name` in tracker. When `_hook_fn` called for module B, if `last_hooked_module_name` is set (module A), add edge A->B to `networkx.DiGraph`. Update `last_hooked_module_name = B`. Reset `last_hooked_module_name` at start of step. Add nodes with metadata.
- **Condition:** Returns `networkx.DiGraph`. Graph for simple sequential model is correct. Handles parallel branches if hook order is consistent.
- **Answer:** Method: `Sequential assumption based on hook execution order within a step.` Format: `NetworkX DiGraph`.

### vis-format-details
- **Technical Details:** Implement `docs/visualization_format.md` to show example state dict and describes NetworkX node/edge attributes (e.g., `node['label']`, `node['type']`, `edge['step']`).
- **Condition:** File exists and shows format.
- **Answer:** Format Snippet: `State: {'id': ..., 'step': ..., 'module_name': ..., 'data': tensor(...), 'shape': ..., ...}`

### vis-mapper-interface-details
- **Technical Details:** Define `VisMapper(ABC)` in `src/visualization/vis_mapper_interface.py`. `get_ui_controls` signature `(self, state_data, ui_context) -> List`. UI elements depend on chosen library (Panel widgets or DearPyGui IDs).
- **Condition:** Interface defined.
- **Answer:** Return Types: `pv.BaseDataObject`, `List[panel.layout.Panel | dpg.ItemId]`

### vis-tensor-mapper-details
- **Technical Details:** Implement `TensorMapper`. `map_to_pyvista`: Use `pv.ImageData` for 2D (texture map or scalars), `pv.StructuredGrid` for 3D. Use `np.arange` for coordinates. For volume render, use `plotter.add_volume(grid, ..., name=state_id)`. `get_ui_controls`: Return `panel.widgets.Select(options=pv.colormaps, name='Colormap')`, `panel.widgets.EditableRangeSlider(name='Opacity Map', ...)`, connect their `.param.watch` to methods that call `plotter.update_scalar_bar()` or `plotter.update_volume_opacity()`.
- **Condition:** Maps tensors. Volume rendering via `add_volume` works. UI controls dynamically adjust opacity/colormap of the rendered volume actor.
- **Answer:** Volume Config: `Uses add_volume linked to Panel widgets for colormap selection and opacity map (via EditableRangeSlider).`

### vis-graph-mapper-details
- **Technical Details:** Implement `GraphMapper`. Use `networkx` to compute layout (`nx.spring_layout`). Create `pv.PolyData` with node points and edge lines. Add node scalars for coloring by module type. `get_ui_controls`: Return `panel.widgets.Select(options=['spring', 'kamada_kawai'])`, `panel.widgets.FloatSlider(name='Node Size')`, `panel.widgets.Checkbox(name='Show Labels')`. Callbacks recompute layout/update point size/toggle labels.
- **Condition:** Maps `networkx.DiGraph`. UI controls adjust layout, size, labels.
- **Answer:** Node Positioning: `NetworkX spring_layout default, options available.`

### vis-other-mappers-details
- **Technical Details:** Implement necessary mappers, e.g., `ARCTaskMapper` displays input/output pairs using `pv.ImageData` in a `pv.MultiBlock`.
- **Condition:** `ARCTaskMapper` exists and works.
- **Answer:** Implemented: `ARCTaskMapper`

### vis-mapper-registry-details
- **Technical Details:** Implement `MapperRegistry`. `get_mapper` checks `state_data['metadata'].get('vis_type')` first (e.g., 'graph', 'arc_task'), then checks `isinstance(state_data['data'], torch.Tensor)` and `state_data['shape']` dimensions.
- **Condition:** Registry returns correct mapper based on state type/shape.
- **Answer:** Selection Logic: `Check metadata hint ('vis_type') -> Check data type (Tensor, Graph) -> Check tensor dimensions.`

### vis-engine-setup-details
- **Technical Details:** Implement `VisualizationEngine`. Use `pyvistaqt.BackgroundPlotter(window_size=(1600, 900), ...)`.
- **Condition:** Engine runs, shows black window, responds to camera.
- **Answer:** Integration: `pyvistaqt`

### vis-dynamic-updates-details
- **Technical Details:** Use `QtCore.QTimer` to call `engine._update` periodically. `_update` gets states, identifies selected state, gets mapper, gets pv_object, uses `plotter.add_mesh(..., name=state_id, overwrite=True)` or `plotter.add_volume(..., name=state_id)`. Store actor references in `self.actor_dict`. Remove old actors using names before adding new ones.
- **Condition:** Visualization updates smoothly when timeline step changes or new states arrive. Overwriting actors prevents memory leaks.
- **Answer:** Update Strategy: `Overwrite actors using unique names based on state ID.`

### vis-imgui-integration-details
- **Technical Details:** Assuming Panel: Use `panel.extension('vtk')`. Main layout `layout = pn.GridSpec(...)`. Place `vtk_pane = panel.pane.VTK(plotter,...)` in a cell. Place Panel widgets (sliders, selectors) in other cells.
- **Condition:** Both PyVista rendering and Panel widgets are visible and interactive within the same application window.
- **Answer:** IMGUI Lib: `panel`

### vis-ui-panel-state-selector - vis-ui-panel-dataset
- **Technical Details:** Implement all specified Panel widgets to work correctly.
- **Condition:** All widgets are functional and interactive.
- **Answer:** Panel Widgets: State Selector, Timeline, Visualization Controls, Computational Graph, Performance Monitor, Interactive Editing, Dataset Selector

### vis-picking-details
- **Technical Details:** Implement `_handle_pick` callback. Needs logic to map picked `cell_id` or `point_id` back to original tensor index/graph node ID, potentially using information stored when creating the PyVista mesh/graph. Display result in `self.info_pane = pn.pane.Markdown(...)`.
- **Condition:** Clicking voxels/nodes displays correct corresponding information (module name, step, value/coords) in the info pane.
- **Answer:** Picking Mech: `Cell Picking`. Info Display: `pn.pane.Markdown`

### vis-feed-integration-details
- **Technical Details:** Implement `StateTracker` with `get_max_step()` and `increment_step()`.
- **Condition:** `StateTracker` correctly tracks step count.
- **Answer:** Transfer Mech: `Engine polls tracker in its update loop.`

### vis-live-test-details
- **Technical Details:** Run `python train.py --visualize --epochs 1 --max_steps 50` to verify live updates.
- **Condition:** Visualization window shows live updates.
- **Answer:** Performance Impact: `[Measure % slowdown compared to run without --visualize]`

### vis-scalability-test-details
- **Technical Details:** Run `python train.py --visualize` for ~1000 steps. Monitor FPS and memory usage.
- **Condition:** Visualization maintains >15 FPS and uses GPU efficiently.
- **Answer:** Target FPS: `20`; Bottlenecks: `[Identify specific slow functions/operations]`

### vis-demo-script-details
- **Technical Details:** Implement `python run_visualization_demo.py` to load saved state history and demonstrate visualization.
- **Condition:** Demo script runs without errors and shows live updates.
- **Answer:** Demo Highlights: `Replay from file, Volume Rendering Controls, Graph Exploration, Timeline Scrubbing.`

## Phase 6 Details

### train-script-details
- **Technical Details:** Implement `train.py` to parse arguments and call DC search.
- **Condition:** Running `python train.py --help` shows expected arguments.
- **Answer:** Command-line Arguments: `model_config_path`, `data_config_path`, `memory_variant`, `dc_enable`, `learning_rate`, `batch_size`, `epochs`, `checkpoint_dir`, `resume_path`, `visualize`, `wandb_log`.

### train-integration-details
- **Technical Details:** Implement `train.py` to correctly initialize all components.
- **Condition:** `train.py` successfully initializes all components without error.
- **Answer:** Components Initialized: KAN, NTM, DreamCoder, Dataset, Optimizer, Scheduler, Logger, StateTracker

### train-loop-logic-details
- **Technical Details:** Implement `run_training_loop` to execute training loop.
- **Condition:** Training loss decreases over initial batches.
- **Answer:** Logging: `WandB`

### train-checkpointing-details
- **Technical Details:** Implement checkpoint saving and loading.
- **Condition:** Checkpoints are saved and loaded correctly.
- **Answer:** Save Frequency: `Every 5 epochs + Best val_pass_at_2`

### train-early-stopping-details
- **Technical Details:** Implement early stopping logic.
- **Condition:** Training stops early when validation metric doesn't improve.
- **Answer:** Metric: `val_pass_at_2` (higher is better); Patience: `50`

### train-ada-details
- **Technical Details:** Implement adaptive computation logic.
- **Condition:** Adaptive computation logic is implemented and tested.
- **Answer:** Strategy: `Not Implemented`

### eval-script-details
- **Technical Details:** Implement `evaluate.py` to calculate Pass@2.
- **Condition:** `evaluate.py` correctly calculates Pass@2.
- **Answer:** Pass@2 Logic: `Model outputs ranked list of predictions or runs twice stochastically. Check if top 1 or top 2 prediction matches ground truth grid exactly.`

### eval-inference-details
- **Technical Details:** Implement `evaluate.py` to run inference on ARC AGI 2 eval sets.
- **Condition:** `evaluate.py` runs without CUDA errors and calculates Pass@2.
- **Answer:** Eval Datasets: `arc2_public_eval`, `arc2_semi_private_eval`

### eval-metrics-details
- **Technical Details:** Implement `evaluate.py` to calculate Pass@2 and overall accuracy.
- **Condition:** `evaluate.py` correctly calculates Pass@2 and overall accuracy.
- **Answer:** Pass@2 Logic: `Model's primary output is attempt 1. If model can generate alternative, that's attempt 2. Check if prediction1 == target OR prediction2 == target.`

### eval-comparison-details
- **Technical Details:** Implement optional comparison against baseline scores.
- **Condition:** Optional comparison against baseline scores is implemented.
- **Answer:** Comparison Logic: `Model's primary output is attempt 1. If model can generate alternative, that's attempt 2. Check if prediction1 == target OR prediction2 == target.`

### ablation-plan-details
- **Technical Details:** Implement `docs/ablation_plan.md` to list planned experiments.
- **Condition:** `docs/ablation_plan.md` exists and lists planned experiments.
- **Answer:** Key Findings Summary: `[Updated with findings as runs complete]`

### ablation-run-details
- **Technical Details:** Implement `train.py` with ablation config files.
- **Condition:** Running `train.py --config configs/ablation_no_ntm.yaml` produces results logged to distinct WandB runs.
- **Answer:** Key Findings Summary: `[Updated with findings as runs complete]`

### refine-analysis-details
- **Technical Details:** Implement `refine_analysis.py` to analyze training logs, evaluation results, visualization insights, and ablation studies.
- **Condition:** `refine_analysis.py` correctly analyzes the project state and suggests improvements.
- **Answer:** Analysis Logic: `Team meetings or issue discussions reference specific evidence to justify proposed changes.`

### refine-iteration-details
- **Technical Details:** Implement `refine_iteration.py` to develop significant changes on experiment branches.
- **Condition:** Significant changes are developed on experiment branches.
- **Answer:** Significant Changes: `[List changes]`

## Phase 7 Details

### final-performance-details
- **Technical Details:** Implement `evaluate.py` to run inference on ARC AGI 2 eval sets.
- **Condition:** `evaluate.py` correctly calculates Pass@2.
- **Answer:** Final Score: `____% Pass@2` on `arc2_public_eval`

### final-cleanup-code-details
- **Technical Details:** Implement `final_cleanup.py` to remove dead code, experimental branches, and unused files.
- **Condition:** Dead code, experimental branches, and unused files are removed.
- **Answer:** Dead Code Removed: `[List removed files/functions]`

### final-cleanup-comments-details
- **Technical Details:** Implement `final_cleanup.py` to ensure all code is well-commented/docstringed.
- **Condition:** All public functions/classes have docstrings explaining purpose, args, returns. Complex algorithms or non-obvious code sections have explanatory comments.
- **Answer:** Commented Code: `[List commented sections]`

### final-docs-update-details
- **Technical Details:** Implement `final_docs_update.py` to update all documentation.
- **Condition:** `README.md`, `GIT_GUIDELINES.md`, and `docs/` are accurate and complete.
- **Answer:** Documentation Updated: `[List updated files]`

### final-reproducibility-details
- **Technical Details:** Implement `final_reproducibility.py` to ensure reproducibility.
- **Condition:** A new collaborator can clone the repo, create the environment, download data, and successfully run `train.py` and `evaluate.py`.
- **Answer:** Confirm Reproducibility: `Yes` by `[Collaborator Name]` on `[Date]`

### final-paper-draft-details
- **Technical Details:** Implement `final_paper_draft.py` to draft the research paper.
- **Condition:** Paper draft exists and is complete.
- **Answer:** Paper Draft: `[Paste Paper Draft Link]`

### final-paper-content-details
- **Technical Details:** Implement `final_paper_content.py` to include figures, tables, and discussions.
- **Condition:** Paper includes generated figures, tables, and discussions.
- **Answer:** Paper Content: `[Paste Paper Content Link]`

### final-paper-finalize-details
- **Technical Details:** Implement `final_paper_finalize.py` to proofread and finalize the paper.
- **Condition:** Final paper is proofread and formatted for submission.
- **Answer:** Final Paper: `[Paste Final Paper Link]`

### final-submission-package-details
- **Technical Details:** Implement `final_submission_package.py` to package the code for submission.
- **Condition:** Submission package is created according to ARC Prize rules.
- **Answer:** Submission Package: `[Paste Submission Package Link]`

### final-submission-test-details
- **Technical Details:** Implement `final_submission_test.py` to test the submission package.
- **Condition:** Submission package is tested locally and passes all checks.
- **Answer:** Test Results: `[Paste Test Results Link]`

### final-submission-action-details
- **Technical Details:** Implement `final_submission_action.py` to submit the solution.
- **Condition:** Submission is uploaded to the platform and confirmed as received.
- **Answer:** Submission Confirmation: `[Paste Submission Confirmation Link]`

### final-release-license-details
- **Technical Details:** Implement `final_release_license.py` to add the LICENSE file.
- **Condition:** LICENSE file exists and contains full text of MIT or Apache 2.0 license.
- **Answer:** License: `Apache 2.0`

### final-release-push-details
- **Technical Details:** Implement `final_release_push.py` to push the final code and documentation to the public repository.
- **Condition:** `git status` on `main` is clean. `git push origin main` completes successfully. Public repository reflects the final state.
- **Answer:** Push Completed: `Yes`

### final-release-tag-details
- **Technical Details:** Implement `final_release_tag.py` to create and push the final release tag.
- **Condition:** `git tag -a v1.0.0 -m "Stable release for ARC AGI 2 submission"` command executed. `git push origin v1.0.0` command executed successfully. Tag appears on GitHub releases page.
- **Answer:** Tag Created and Pushed: `v1.0.0`
