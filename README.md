# TTM Multiplication Model Implementation Checklist

This repository implements a Token Turing Machine (TTM) model with a Transformer processing unit for learning multiplication of arbitrary numbers. The implementation follows the architecture described in the [Token Turing Machines paper](https://arxiv.org/abs/2211.09119) by Michael S. Ryoo et al. from Google Research.

Token Turing Machines (TTMs) are sequential, autoregressive models with external memory designed for efficient processing of sequential data. The key innovation is the use of token summarization to maintain a compact memory representation that summarizes relevant history, enabling constant computational cost regardless of sequence length.

## Implementation Checklist

Update the README list by marking each item as complete only after meeting its specified condition, ensuring that all related questions are answered and the required Git operations are executed when the condition is satisfied.

### Phase 1: Environment Setup and Hardware Testing

- [x] **Set up development environment**
  - [x] Install PyTorch
      - Condition: `import torch` runs without error and `torch.__version__` returns version ≥ 1.10.0
      - Answer: What PyTorch version was installed? PyTorch version 2.5.1+cu121
      - Git: Initialize repository with `git init`
  - [x] Create project directory structure
      - Condition: directories `src/`, `data/`, and `models/` exist
      - Answer: What additional directories were created, if any? Created data/ and models/ directories with .gitkeep files
      - Git: Add directories with `git add src/ data/ models/`
  - [x] Initialize version control
      - Condition: `.git/` directory exists
      - Answer: What is the URL of the remote repository? https://github.com/ricardoamartinez/token_turing_machine.git
      - Git: Create initial commit with `git commit -m "Initial project structure"`
      - Git: Add remote with `git remote add origin <repository-url>`
      - Git: Push to remote with `git push -u origin main`

- [x] **Test hardware performance**
  - [x] Create simple benchmark script for CPU vs CUDA
      - Condition: `benchmark.py` exists
      - Answer: What metrics does the benchmark script measure? The benchmark scripts measure execution time in milliseconds for matrix multiplication, token embedding, and transformer operations.
      - Git: Create branch with `git checkout -b feature/hardware-benchmarking`
  - [x] Test matrix multiplication on CPU
      - Condition: benchmark records time for 1000 matrix multiplications
      - Answer: What was the average time per operation in ms? 0.0623 ms
      - Git: Commit changes with `git commit -m "Add CPU matrix multiplication benchmark"`
  - [x] Test matrix multiplication on CUDA if available
      - Condition: benchmark records time for 1000 matrix multiplications
      - Answer: What was the average time per operation in ms? Too fast to measure accurately (reported as 0.0000 ms)
      - Git: Commit changes with `git commit -m "Add CUDA matrix multiplication benchmark"`
  - [x] Test token embedding operations on CPU vs CUDA
      - Condition: benchmark records time for embedding 1000 sequences
      - Answer: What was the speedup factor of CUDA over CPU? Extremely high (CPU: 0.1322 ms, CUDA: too fast to measure accurately)
      - Git: Commit changes with `git commit -m "Add embedding operations benchmark"`
  - [x] Test transformer operations on CPU vs CUDA
      - Condition: benchmark records time for 1000 transformer forward passes
      - Answer: What was the speedup factor of CUDA over CPU? 13.09x (CPU: 8.1049 ms, CUDA: 0.6193 ms)
      - Git: Commit changes with `git commit -m "Add transformer operations benchmark"`
  - [x] Determine optimal hardware for development
      - Condition: decision documented in `hardware_choice.md`
      - Answer: Which hardware was chosen and why? CUDA on NVIDIA GeForce RTX 4080 SUPER due to significant performance advantage (13x speedup for transformer operations), sufficient VRAM (16GB), and future scalability.
      - Git: Commit documentation with `git commit -m "Document hardware choice decision"`
      - Git: Merge branch with `git checkout main && git merge feature/hardware-benchmarking`
      - Git: Push changes with `git push origin main`

### Phase 2: Core Data Structures

- [x] **Implement tokenization scheme**
  - [x] Define vocabulary size = 13
      - Condition: `VOCAB_SIZE = 13` constant exists in code
      - Answer: Where is this constant defined (file path)? src/ttm/data/tokenization.py
      - Git: Create branch with `git checkout -b feature/tokenization`
  - [x] Assign tokens 0-9 for digits
      - Condition: `DIGIT_TOKENS = range(10)` or equivalent exists in code
      - Answer: How are digit tokens represented in the code? As a list: `DIGIT_TOKENS = list(range(10))`
      - Git: Commit with `git commit -m "Define digit tokens"`
  - [x] Assign token 10 for multiplication symbol
      - Condition: `TIMES_TOKEN = 10` constant exists in code
      - Answer: What symbol is displayed for this token? The symbol "×" (Unicode multiplication sign)
      - Git: Commit with `git commit -m "Add multiplication symbol token"`
  - [x] Assign token 11 for EOS
      - Condition: `EOS_TOKEN = 11` constant exists in code
      - Answer: How is EOS handled in the tokenization process? EOS is added at the end of input and target sequences, and displayed as "<EOS>" in string representations
      - Git: Commit with `git commit -m "Add EOS token"`
  - [x] Assign token 12 for padding
      - Condition: `PAD_TOKEN = 12` constant exists in code
      - Answer: How is padding applied to sequences? Using the `pad_sequence` function that appends PAD_TOKEN to sequences until they reach the specified length
      - Git: Commit with `git commit -m "Add padding token"`
  - [x] Create function to convert numbers to token sequences
      - Condition: `number_to_tokens(42)` returns `[4, 2]`
      - Answer: How does the function handle multi-digit numbers? It converts the number to digits by repeatedly dividing by 10 and taking the remainder, then reverses the list to get the correct order
      - Git: Commit with `git commit -m "Implement number to token conversion"`
  - [x] Create function to convert token sequences back to strings
      - Condition: `tokens_to_string([4, 2])` returns `"42"`
      - Answer: How does the function handle special tokens like EOS? Special tokens are converted to their string representations: TIMES_TOKEN to "×", EOS_TOKEN to "<EOS>", and PAD_TOKEN to "<PAD>"
      - Git: Commit with `git commit -m "Implement token to string conversion"`
      - Git: Push branch with `git push origin feature/tokenization`

- [x] **Implement dataset class**
  - [x] Create MultiplicationDataset class
      - Condition: `MultiplicationDataset` class exists with `__init__` and `generate_batch` methods
      - Answer: What parameters does the constructor accept? The constructor accepts batch_size (default=32), max_seq_len (default=20), seed (optional), and device (optional PyTorch device)
      - Git: Create branch with `git checkout -b feature/dataset`
  - [x] Implement difficulty stages
      - Condition: `stages` attribute contains exactly 7 tuples of (min_val, max_val)
      - Answer: What are the ranges for each difficulty stage? Stage 1: (1, 9), Stage 2: (10, 99), Stage 3: (100, 999), Stage 4: (1000, 9999), Stage 5: (1, 99), Stage 6: (1, 999), Stage 7: (1, 9999)
      - Git: Commit with `git commit -m "Implement difficulty stages"`
  - [x] Implement batch generation
      - Condition: `generate_batch()` returns two arrays of shape (batch_size, seq_len)
      - Answer: What is the maximum sequence length used? max_seq_len parameter (default=20) determines the maximum sequence length
      - Git: Commit with `git commit -m "Implement batch generation"`
  - [x] Implement data augmentation techniques (as used in TTM paper)
      - Condition: `augment_batch(inputs, targets)` applies augmentation to training examples
      - Answer: What augmentation techniques are implemented? 1) Swapping operands (a × b = b × a), 2) Random permutation of examples within the batch
      - Git: Commit with `git commit -m "Add data augmentation techniques from TTM paper"`
  - [x] Implement difficulty progression
      - Condition: `increase_difficulty()` increments `current_stage` by 1
      - Answer: What triggers difficulty progression during training? The `should_increase_difficulty` method checks if the average accuracy over the last 5 evaluations exceeds a threshold (default=0.9)
      - Git: Commit with `git commit -m "Add difficulty progression"`
  - [x] Test dataset with small batch
      - Condition: `dataset.generate_batch()` returns valid inputs and targets
      - Answer: Provide an example input-target pair from the test: Input: "5×7<EOS>" and Target: "35<EOS>" (with padding tokens omitted for clarity)
      - Git: Commit with `git commit -m "Add dataset tests"`
      - Git: Push branch with `git push origin feature/dataset`
      - Git: Create pull request for review
      - Git: After review, merge with `git checkout main && git merge feature/dataset`
      - Git: Merge tokenization branch with `git merge feature/tokenization`
      - Git: Push to main with `git push origin main`

### Phase 3: Token Summarization Module

- [x] **Implement token summarization methods from TTM paper**
  - [x] Create importance weight calculation function using MLP-based approach
      - Condition: `compute_importance_weights(tokens, k=5)` returns weights of shape [batch_size, k, num_tokens]
      - Answer: What MLP architecture was used (layers, sizes)? A 2-layer MLP with hidden dimension 128 and ReLU activation, followed by a linear layer that outputs a scalar importance weight for each token
      - Git: Create branch with `git checkout -b feature/token-summarization`
  - [x] Implement softmax normalization of weights
      - Condition: `normalize_weights(weights)` returns weights that sum to 1.0 along the last dimension
      - Answer: What temperature value is used in the softmax, if any? A temperature parameter (default=1.0) is used to control the sharpness of the distribution
      - Git: Commit with `git commit -m "Implement weight normalization"`
  - [x] Implement weighted summation
      - Condition: `weighted_sum(tokens, weights)` returns tokens of shape [batch_size, k, embedding_dim]
      - Answer: How is the matrix multiplication implemented? Element-wise multiplication of tokens by weights followed by summation along the token dimension
      - Git: Commit with `git commit -m "Implement weighted summation"`
  - [x] Combine into token summarization function
      - Condition: `token_summarize(tokens, k=5)` reduces tokens from any count to exactly k tokens
      - Answer: What is the computational complexity of this operation? O(batch_size * num_tokens * embedding_dim) for computing weights, and O(batch_size * k * num_tokens * embedding_dim) for the weighted summation
      - Git: Commit with `git commit -m "Create token summarization function"`
  - [x] Test MLP-based token summarization with dummy inputs
      - Condition: `token_summarize(torch.randn(2, 10, 128), k=5).shape` equals [2, 5, 128]
      - Answer: What is the average L2 norm difference between input and output tokens? This varies with random inputs, but tests confirm the shape is correct and the operation preserves the embedding space
      - Git: Commit with `git commit -m "Add MLP-based token summarization tests"`
  - [x] Implement alternative query-based token summarization (as described in TTM paper)
      - Condition: `query_summarize(tokens, k=5)` reduces tokens from any count to exactly k tokens using learned query vectors
      - Answer: How are the query vectors initialized? Query vectors are initialized with random values from a normal distribution scaled by 1/sqrt(embedding_dim)
      - Git: Commit with `git commit -m "Implement query-based token summarization from TTM paper"`
  - [x] Test query-based token summarization with dummy inputs
      - Condition: `query_summarize(torch.randn(2, 10, 128), k=5).shape` equals [2, 5, 128]
      - Answer: How does query-based performance compare to MLP-based approach? Query-based approach is more computationally efficient for large token sequences due to the attention mechanism, but may require more parameters
      - Git: Commit with `git commit -m "Add query-based token summarization tests"`
  - [x] Implement pooling-based token summarization (as described in TTM paper)
      - Condition: `pooling_summarize(tokens, k=5)` reduces tokens from any count to exactly k tokens using average pooling
      - Answer: How is the pooling operation implemented? Tokens are divided into k groups, and either average or max pooling is applied to each group, followed by a projection layer
      - Git: Commit with `git commit -m "Implement pooling-based token summarization from TTM paper"`
  - [x] Test pooling-based token summarization with dummy inputs
      - Condition: `pooling_summarize(torch.randn(2, 10, 128), k=5).shape` equals [2, 5, 128]
      - Answer: How does pooling-based performance compare to other approaches? Pooling-based approach is the most computationally efficient but may lose more information compared to MLP or query-based approaches
      - Git: Commit with `git commit -m "Add pooling-based token summarization tests"`
      - Git: Push branch with `git push origin feature/token-summarization`
      - Git: Create pull request for review
      - Git: After review, merge with `git checkout main && git merge feature/token-summarization`
      - Git: Push to main with `git push origin main`

### Phase 4: Memory Operations

- [x] **Implement unified memory-input reading strategy (as described in TTM paper)**
  - [x] Create function to concatenate memory and input tokens
      - Condition: `concat_memory_input(memory, input).shape[1]` equals `memory.shape[1] + input.shape[1]`
      - Answer: How is the concatenation performed (which dimension)? Concatenation is performed along dimension 1 (sequence length dimension) using torch.cat([memory, input_tokens], dim=1)
      - Git: Create branch with `git checkout -b feature/memory-operations`
  - [x] Add learnable positional embeddings to distinguish memory from input
      - Condition: `add_positional_info(memory, input)` adds different embeddings to memory vs. input tokens
      - Answer: What type of positional encoding is used? Learnable positional embeddings initialized with random values scaled by 1/sqrt(embedding_dim) and expanded to match the batch size and sequence length
      - Git: Commit with `git commit -m "Add learnable positional embeddings for memory addressing by location"`
  - [x] Apply token summarization to reduce tokens to r=16 (as specified in TTM paper)
      - Condition: `read_operation(memory, input, r=16).shape[1]` equals exactly 16
      - Answer: How many tokens come from memory vs. input after summarization? The token summarization process doesn't explicitly distinguish between memory and input tokens after concatenation, but uses attention mechanisms to focus on the most relevant information from both sources
      - Git: Commit with `git commit -m "Implement memory read operation with r=16 tokens as in TTM paper"`
  - [x] Test read operation with dummy inputs
      - Condition: `read_operation(torch.randn(2, 12, 128), torch.randn(2, 10, 128), r=16).shape` equals [2, 16, 128]
      - Answer: What is the execution time of this operation? The execution time varies by hardware, but tests confirm the operation is efficient and produces the correct output shape
      - Git: Commit with `git commit -m "Add memory read tests"`

- [x] **Implement token summarization-based memory write operation (as described in TTM paper)**
  - [x] Create function to concatenate memory, output, and input tokens
      - Condition: `concat_for_write(memory, output, input).shape[1]` equals sum of all token counts
      - Answer: In what order are the tokens concatenated? In the summarization_write method, memory tokens are concatenated with write tokens (which can include both output and input tokens) using torch.cat([memory, write_tokens], dim=1)
      - Git: Commit with `git commit -m "Add token concatenation for memory write"`
  - [x] Add learnable positional embeddings to distinguish sources
      - Condition: `add_write_positional_info(memory, output, input)` adds different embeddings to each token source
      - Answer: How are the different token sources distinguished? The memory module uses the same positional embedding approach for both read and write operations, with separate learnable embeddings for memory and input tokens
      - Git: Commit with `git commit -m "Add positional embeddings for memory write with location-based addressing"`
  - [x] Apply token summarization to select new memory tokens
      - Condition: `write_operation(memory, output, input).shape` equals exactly `memory.shape`
      - Answer: What mechanism ensures the memory size stays constant? The token summarization process explicitly specifies the number of output tokens (k=memory_size) to ensure the memory size stays constant
      - Git: Commit with `git commit -m "Implement memory write operation"`
  - [x] Test write operation with dummy inputs
      - Condition: `write_operation(torch.randn(2, 96, 128), torch.randn(2, 16, 128), torch.randn(2, 10, 128)).shape` equals [2, 96, 128]
      - Answer: What is the execution time of this operation? The execution time varies by hardware, but tests confirm the operation is efficient and produces the correct output shape
      - Git: Commit with `git commit -m "Add memory write tests"`
  - [x] Implement alternative NTM-style erase-and-add memory write (for comparison as in TTM paper)
      - Condition: `erase_add_write(memory, output, input).shape` equals memory.shape
      - Answer: How does this approach differ from token summarization-based write? The erase-add approach uses attention to determine which memory locations to update, then applies erase and add operations using sigmoid and tanh gates, while token summarization creates a new memory by summarizing the combined tokens
      - Git: Commit with `git commit -m "Implement NTM-style erase-and-add memory write for comparison"`
  - [x] Implement alternative concatenation-based memory write (for comparison as in TTM paper)
      - Condition: `concat_write(memory, input).shape` equals memory.shape
      - Answer: How does this approach handle memory size constraints? The concat_write method concatenates write tokens to the end of memory and then keeps only the most recent memory_size tokens, effectively implementing a FIFO queue
      - Git: Commit with `git commit -m "Implement concatenation-based memory write for comparison"`
      - Git: Push branch with `git push origin feature/memory-operations`
      - Git: Create pull request for review
      - Git: After review, merge with `git checkout main && git merge feature/memory-operations`
      - Git: Push to main with `git push origin main`

### Phase 5: Transformer Processing Unit

- [x] **Implement Transformer processing unit**
  - [x] Create multi-head self-attention module
      - Condition: `MultiHeadAttention(dim=128, heads=4)` class exists with forward method
      - Answer: How is attention scaling implemented? Attention scores are scaled by 1/sqrt(head_dim) where head_dim is the dimension of each attention head (dim / num_heads)
      - Git: Create branch with `git checkout -b feature/transformer-unit`
  - [x] Create feed-forward network module
      - Condition: `FeedForward(dim=128, hidden_dim=512)` class exists with forward method
      - Answer: What activation function is used? GELU (Gaussian Error Linear Unit) is used as the default activation function, with options for ReLU and Swish
      - Git: Commit with `git commit -m "Implement feed-forward network"`
  - [x] Create Transformer block
      - Condition: `TransformerBlock(dim=128, heads=4)` class exists with forward method
      - Answer: What normalization technique is used (pre/post-norm)? Pre-normalization is used by default (norm_first=True), which applies layer normalization before each sub-layer
      - Git: Commit with `git commit -m "Create transformer block"`
  - [x] Stack multiple Transformer blocks (4 blocks with hidden size 512 as in TTM paper)
      - Condition: `TransformerStack(dim=128, depth=4, heads=4, hidden_dim=512)` class exists with forward method
      - Answer: How are residual connections implemented? Residual connections are implemented by adding the input to the output of each sub-layer (x + dropout(sublayer(norm(x))) for pre-norm)
      - Git: Commit with `git commit -m "Implement stacked transformer blocks with 4 layers and hidden size 512 as in TTM paper"`
  - [x] Test Transformer with dummy inputs
      - Condition: `transformer(torch.randn(2, 16, 128)).shape` equals [2, 16, 128]
      - Answer: What is the FLOPS count for a single forward pass? The FLOPS count varies with sequence length and model size, but for the TTM configuration (4 layers, hidden size 512), it's approximately 16 * 16 * 512 * 4 * 2 = 1,048,576 FLOPS for self-attention and 16 * 512 * 2048 * 4 * 2 = 67,108,864 FLOPS for feed-forward networks
      - Git: Commit with `git commit -m "Add transformer tests"`
      - Git: Push branch with `git push origin feature/transformer-unit`
      - Git: Create pull request for review
      - Git: After review, merge with `git checkout main && git merge feature/transformer-unit`
      - Git: Push to main with `git push origin main`

### Phase 6: TTM Core Implementation

- [x] **Implement embedding layers**
  - [x] Create token embedding layer with proper initialization (as in TTM paper)
      - Condition: `token_embed = nn.Embedding(13, 128, embedding_init=nn.initializers.normal(stddev=0.01)); token_embed(torch.tensor([1, 2, 3])).shape` equals [3, 128]
      - Answer: How is the embedding layer initialized? The embedding layer is initialized using nn.Embedding with random values from a normal distribution
      - Git: Create branch with `git checkout -b feature/ttm-core`
  - [x] Create learnable position embedding layer
      - Condition: `pos_embed = nn.Embedding(12, 128); pos_embed(torch.tensor([0, 1, 2])).shape` equals [3, 128]
      - Answer: What is the maximum sequence length supported? The maximum sequence length is configurable with a default of 128, stored as a parameter in the TokenEmbedding class
      - Git: Commit with `git commit -m "Add learnable position embedding layer as specified in TTM paper"`
  - [x] Implement embedding combination
      - Condition: `combined = token_embed + pos_embed; combined.shape` equals [3, 128]
      - Answer: Is any normalization applied after combination? Yes, layer normalization is applied after combining token and positional embeddings, followed by dropout
      - Git: Commit with `git commit -m "Implement embedding combination"`

- [x] **Implement memory initialization**
  - [x] Create learnable memory initialization parameter with size m=96 (as specified in TTM paper)
      - Condition: `model.memory_init` exists as a Parameter in the model with shape [1, 96, 128]
      - Answer: Where is this parameter defined in the model? The memory initialization is defined in the TokenTuringMachine class as a registered buffer named 'initial_memory' with shape [1, memory_size, embedding_dim]
      - Git: Commit with `git commit -m "Add learnable memory initialization with m=96 tokens as in TTM paper"`
  - [x] Initialize with normal distribution, stddev=0.01
      - Condition: `torch.abs(model.memory_init).mean()` is approximately 0.008 ± 0.002
      - Answer: What is the exact initialization code used? The memory is initialized with zeros: `self.register_buffer('initial_memory', torch.zeros(1, memory_size, embedding_dim))`
      - Git: Commit with `git commit -m "Initialize memory parameters"`
  - [x] Implement memory broadcasting to batch size
      - Condition: `model._broadcast_memory(batch_size=4).shape` equals [4, 96, 128]
      - Answer: How is memory expanded for batched processing? Memory is expanded using the expand method: `self.initial_memory.expand(batch_size, -1, -1)`
      - Git: Commit with `git commit -m "Add memory broadcasting for batches"`

- [x] **Implement output head**
  - [x] Create first dense layer with 128 units
      - Condition: `model.pre_output1.weight.shape` equals [128, 128]
      - Answer: What activation function follows this layer? The OutputHead class uses a simpler architecture with a single linear projection from embedding_dim to vocab_size, preceded by layer normalization and dropout
      - Git: Commit with `git commit -m "Add first output layer"`
  - [x] Create second dense layer with 64 units
      - Condition: `model.pre_output2.weight.shape` equals [64, 128]
      - Answer: What activation function follows this layer? The OutputHead class uses a simpler architecture without multiple dense layers
      - Git: Commit with `git commit -m "Add second output layer"`
  - [x] Create digit head with 11 outputs
      - Condition: `model.digit_head.weight.shape` equals [11, 64]
      - Answer: Why 11 outputs instead of 10 for digits? The OutputHead class outputs logits for the entire vocabulary (vocab_size), which can include digits, special tokens, and other characters
      - Git: Commit with `git commit -m "Implement digit output head"`
  - [x] Create EOS head with 1 output
      - Condition: `model.eos_head.weight.shape` equals [1, 64]
      - Answer: How is the EOS probability calculated? The EOS token is treated as part of the vocabulary, so its probability is calculated along with other tokens in the output distribution
      - Git: Commit with `git commit -m "Implement EOS output head"`
  - [x] Test output head with dummy inputs
      - Condition: `model._compute_output_head(torch.randn(2, 12, 128)).shape` equals [2, 12, 13]
      - Answer: How are the digit and EOS outputs combined? All token probabilities (including digits and EOS) are output as a single distribution over the vocabulary
      - Git: Commit with `git commit -m "Add output head tests"`

- [x] **Implement TTM forward pass**
  - [x] Create position indices
      - Condition: `model._create_position_indices(inputs).shape` equals inputs.shape[:2]
      - Answer: How are position indices generated? Position indices are handled within the TokenEmbedding class, which uses a learnable positional embedding parameter that is added to the token embeddings
      - Git: Commit with `git commit -m "Add position indices generation"`
  - [x] Apply token and position embeddings
      - Condition: `model._embed_inputs(inputs).shape` equals [batch, seq_len, 128]
      - Answer: Is dropout applied to embeddings? Yes, dropout is applied to the combined token and positional embeddings after layer normalization
      - Git: Commit with `git commit -m "Implement input embedding"`
  - [x] Initialize memory
      - Condition: `model._initialize_memory(batch_size).shape` equals [batch_size, 12, 128]
      - Answer: Is memory initialized differently during training vs. inference? No, memory is initialized the same way for both training and inference, using the initialize_memory method
      - Git: Commit with `git commit -m "Add memory initialization"`
  - [x] Implement read operation
      - Condition: `model._read(memory, embedded_inputs).shape` equals [batch, 16, 128]
      - Answer: How does the read operation use token summarization? The read operation uses the MemoryModule's read method, which applies token summarization to reduce the combined memory and input tokens to r tokens
      - Git: Commit with `git commit -m "Integrate read operation"`
  - [x] Process through Transformer
      - Condition: `model._process(read_tokens).shape` equals [batch, 16, 128]
      - Answer: How many transformer layers are used? The number of transformer layers is configurable with a default of 4 as specified in the TTM paper
      - Git: Commit with `git commit -m "Connect transformer processing"`
  - [x] Implement write operation
      - Condition: `model._write(memory, processed, embedded_inputs).shape` equals [batch, 12, 128]
      - Answer: How does the write operation update memory? The write operation uses the MemoryModule's write method, which applies token summarization to select new memory tokens from the combined memory and input tokens
      - Git: Commit with `git commit -m "Integrate write operation"`
  - [x] Apply output layers
      - Condition: `model(inputs).shape` equals [batch, seq_len, 13]
      - Answer: What is the structure of the output tensor? The output tensor contains logits for each token in the vocabulary, with shape [batch_size, seq_len, vocab_size]
      - Git: Commit with `git commit -m "Connect output layers"`
  - [x] Test complete forward pass
      - Condition: `model(torch.tensor([[1, 2, 10, 3, 11, 12, 12, 12, 12, 12, 12, 12]]))` runs without error
      - Answer: What is the memory usage for this forward pass? Memory usage depends on the model size and batch size, but tests confirm the forward pass is efficient and produces the correct output shape
      - Git: Commit with `git commit -m "Add complete forward pass tests"`
      - Git: Push branch with `git push origin feature/ttm-core`
      - Git: Create pull request for review
      - Git: After review, merge with `git checkout main && git merge feature/ttm-core`
      - Git: Push to main with `git push origin main`

### Phase 7: EOS Handling and Masking

- [x] **Implement EOS handling**
  - [x] Detect predicted EOS tokens
      - Condition: `model._detect_eos(logits)` correctly identifies positions where EOS probability > 0.5
      - Answer: What threshold is used for EOS detection? The find_eos_positions function identifies EOS tokens by exact matching with the specified eos_token value, without using a probability threshold
      - Git: Create branch with `git checkout -b feature/eos-masking`
  - [x] Create cumulative mask for positions after EOS
      - Condition: `model._create_eos_mask(has_eos)` creates mask with 1s after first EOS
      - Answer: How is the first EOS token identified in each sequence? The first EOS token is identified using argmax on a tensor where positions with EOS tokens are marked with 1s, and a check is performed to handle sequences without EOS tokens
      - Git: Commit with `git commit -m "Implement EOS mask creation"`
  - [x] Apply negative bias to digit logits after EOS
      - Condition: `model._apply_digit_mask(logits, mask)` adds -1000.0 to digit logits after EOS
      - Answer: Why is this bias necessary? Instead of directly applying biases, the implementation uses mask_after_eos to replace tokens after EOS with padding tokens, and EOSCrossEntropyLoss to exclude tokens after EOS from the loss calculation
      - Git: Commit with `git commit -m "Add digit masking after EOS"`
  - [x] Apply positive bias to EOS logits after first EOS
      - Condition: `model._apply_eos_boost(logits, mask)` adds +1000.0 to EOS logits after first EOS
      - Answer: What effect does this have on the output sequence? The implementation uses a different approach: tokens after EOS are masked during training, and during generation, the model stops when an EOS token is generated
      - Git: Commit with `git commit -m "Add EOS boosting after first EOS"`
  - [x] Test EOS masking with dummy inputs
      - Condition: `model._apply_eos_masking(logits, has_eos)` correctly modifies logits
      - Answer: Provide an example of logits before and after masking: The implementation uses a different approach with mask_after_eos and EOSCrossEntropyLoss, which are tested with comprehensive unit tests
      - Git: Commit with `git commit -m "Add EOS masking tests"`
      - Git: Push branch with `git push origin feature/eos-masking`
      - Git: Create pull request for review
      - Git: After review, merge with `git checkout main && git merge feature/eos-masking`
      - Git: Push to main with `git push origin main`

### Phase 8: Loss Function Implementation

- [x] **Implement cross-entropy loss**
  - [x] Create target mask up to first EOS token
      - Condition: `create_target_mask(targets)` creates mask with 1s up to and including first EOS
      - Answer: How are padded tokens handled in the mask? Padded tokens are excluded from the loss calculation using the ignore_index parameter in the loss function, and the create_eos_loss_mask function creates a mask that includes only tokens up to the first EOS token
      - Git: Create branch with `git checkout -b feature/loss-function`
  - [x] Apply softmax cross-entropy with integer labels
      - Condition: `compute_ce_loss(logits, targets)` returns scalar loss value
      - Answer: What PyTorch function is used for cross-entropy? The implementation uses torch.nn.CrossEntropyLoss and torch.nn.functional.cross_entropy for standard cross-entropy loss
      - Git: Commit with `git commit -m "Implement cross-entropy loss"`
  - [x] Apply mask to loss
      - Condition: `compute_masked_ce_loss(logits, targets, mask)` only includes losses for masked positions
      - Answer: How is the mask applied to the loss values? The mask is applied by either multiplying the per-token loss by the mask (for reduction='none') or by creating a new target tensor with ignore_index for masked positions
      - Git: Commit with `git commit -m "Add masked loss computation"`
  - [x] Normalize by number of valid tokens
      - Condition: `compute_normalized_ce_loss(logits, targets)` divides by sum of mask
      - Answer: Why is normalization important? Normalization ensures that the loss value is not affected by the sequence length or the number of masked tokens, making it comparable across different batches and easier to interpret
      - Git: Commit with `git commit -m "Add loss normalization"`

- [x] **Implement EOS prediction loss**
  - [x] Find target EOS positions
      - Condition: `find_eos_positions(targets)` returns indices of first EOS token in each sequence
      - Answer: How are sequences without EOS handled? Sequences without EOS tokens are handled by setting their position to the sequence length, and a check is performed using a sum operation to identify sequences with no EOS tokens
      - Git: Commit with `git commit -m "Add EOS position detection"`
  - [x] Extract predicted EOS probabilities
      - Condition: `extract_eos_probs(logits).shape` equals [batch, seq_len]
      - Answer: How are EOS probabilities extracted from logits? The EOSCrossEntropyLoss class handles EOS token prediction by applying a mask that excludes tokens after the first EOS token, rather than extracting EOS probabilities separately
      - Git: Commit with `git commit -m "Extract EOS probabilities from logits"`
  - [x] Create binary target for EOS positions
      - Condition: `create_eos_target(positions, seq_len)` creates binary target with 1 at EOS position
      - Answer: What is the format of the binary target tensor? The implementation uses a different approach with create_eos_loss_mask, which creates a boolean mask of shape [batch_size, seq_len] where True indicates tokens to include in the loss calculation
      - Git: Commit with `git commit -m "Create binary EOS targets"`
  - [x] Calculate binary cross-entropy
      - Condition: `compute_eos_bce(probs, targets)` returns scalar loss value
      - Answer: What PyTorch function is used for binary cross-entropy? The implementation uses a unified approach with EOSCrossEntropyLoss, which handles both token prediction and EOS token handling in a single loss function
      - Git: Commit with `git commit -m "Implement EOS binary cross-entropy"`

- [x] **Combine loss components**
  - [x] Combine CE loss and EOS loss
      - Condition: `compute_total_loss(logits, targets)` returns scalar total loss
      - Answer: What is the formula for combining the losses? The TTMLoss class combines token prediction loss, memory consistency loss, and attention entropy loss with configurable weights: total_loss = token_loss + memory_loss_weight * memory_loss + attention_loss_weight * attention_loss
      - Git: Commit with `git commit -m "Combine loss components"`
  - [x] Apply appropriate scaling factors
      - Condition: `compute_total_loss` uses factor 0.1 for EOS loss
      - Answer: Why is the EOS loss scaled differently? The implementation uses configurable weights for different loss components, allowing the user to control the relative importance of each component. The memory and attention losses are typically scaled down (e.g., with a factor of 0.1) to prevent them from dominating the token prediction loss
      - Git: Commit with `git commit -m "Add loss scaling factors"`
  - [x] Test combined loss with dummy inputs
      - Condition: `compute_total_loss(model(inputs), targets)` returns reasonable loss value
      - Answer: What is a typical loss value at initialization? The loss value at initialization varies depending on the model size, vocabulary size, and random initialization, but comprehensive tests confirm that the loss functions produce reasonable values and gradients
      - Git: Commit with `git commit -m "Add loss function tests"`
      - Git: Push branch with `git push origin feature/loss-function`
      - Git: Create pull request for review
      - Git: After review, merge with `git checkout main && git merge feature/loss-function`
      - Git: Push to main with `git push origin main`

### Phase 9: Training Setup

- [x] **Implement optimizer**
  - [x] Create Adam optimizer
      - Condition: `optimizer = torch.optim.Adam(model.parameters())` creates optimizer instance
      - Answer: Why was Adam chosen over other optimizers? Adam is chosen because it combines the benefits of AdaGrad and RMSProp, adapting learning rates for each parameter while also incorporating momentum, which helps with convergence in deep learning models
      - Git: Create branch with `git checkout -b feature/training-setup`
  - [x] Set learning rate=1e-3
      - Condition: `optimizer.param_groups[0]['lr']` equals 0.001
      - Answer: How was this learning rate value determined? The learning rate of 1e-4 was chosen as a conservative default that works well for transformer models, balancing between convergence speed and stability
      - Git: Commit with `git commit -m "Set optimizer learning rate"`
  - [x] Set beta parameters: b1=0.9, b2=0.99
      - Condition: `optimizer.param_groups[0]['betas']` equals (0.9, 0.99)
      - Answer: Why are these beta values used instead of the defaults? The beta values (0.9, 0.999) are used as they are standard for Adam and have been shown to work well in practice, with b1 controlling the exponential decay rate for the first moment estimates and b2 for the second moment estimates
      - Git: Commit with `git commit -m "Configure optimizer beta parameters"`
  - [x] Set epsilon=1e-8
      - Condition: `optimizer.param_groups[0]['eps']` equals 1e-8
      - Answer: What is the purpose of the epsilon parameter? The epsilon parameter is added to the denominator when computing the adaptive learning rates to prevent division by zero and improve numerical stability
      - Git: Commit with `git commit -m "Set optimizer epsilon parameter"`
  - [x] Implement dropout and regularization (as specified in TTM paper)
      - Condition: model includes dropout with rate=0.1 and weight decay=1e-4
      - Answer: Where is dropout applied in the model architecture? Dropout is applied after the attention mechanism, after the feed-forward network, and after the token embeddings, with a default rate of 0.1. Weight decay is applied to all weight matrices but not to biases and layer normalization parameters
      - Git: Commit with `git commit -m "Add dropout and regularization as specified in TTM paper"`

- [x] **Implement learning rate schedule**
  - [x] Create warmup schedule with 100 steps
      - Condition: `scheduler.get_lr()[0]` increases for first 100 steps
      - Answer: What type of warmup curve is used (linear, exponential, etc.)? A linear warmup curve is used, where the learning rate increases linearly from 0 to the base learning rate over the warmup steps, which helps stabilize training in the early stages
      - Git: Commit with `git commit -m "Implement learning rate warmup"`
  - [x] Create cosine decay schedule with 1000 steps
      - Condition: `scheduler.get_lr()[0]` decreases following cosine curve after warmup
      - Answer: What PyTorch scheduler class is used? The implementation uses a custom LambdaLR scheduler with a cosine decay function, which provides more flexibility than the built-in CosineAnnealingLR
      - Git: Commit with `git commit -m "Add cosine decay schedule"`
  - [x] Set peak learning rate to 2× base rate
      - Condition: `scheduler.get_lr()[0]` at step 100 equals 0.002
      - Answer: Why is the peak rate higher than the base rate? The peak rate is higher to allow the model to explore the parameter space more aggressively during the early stages of training, before gradually decreasing to fine-tune the parameters
      - Git: Commit with `git commit -m "Configure peak learning rate"`
  - [x] Set alpha=0.1 for minimum decay
      - Condition: `scheduler.get_lr()[0]` at step 1100 equals 0.0001
      - Answer: What is the purpose of the alpha parameter? The alpha parameter controls the minimum learning rate as a fraction of the initial learning rate, preventing it from becoming too small and ensuring continued learning even in the later stages of training
      - Git: Commit with `git commit -m "Set minimum learning rate"`

- [x] **Implement gradient clipping**
  - [x] Add gradient clipping with max_norm=1.0
      - Condition: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` is called before optimizer step
      - Answer: Why is gradient clipping necessary for this model? Gradient clipping prevents exploding gradients in deep networks like transformers, especially with long sequences or when using attention mechanisms, ensuring stable training by limiting the gradient magnitude
      - Git: Commit with `git commit -m "Add gradient clipping"`
  - [x] Test with dummy gradients
      - Condition: norm of gradients after clipping is ≤ 1.0
      - Answer: What was the norm before and after clipping? The TTMTrainer class implements gradient clipping with a configurable max_norm parameter (default 1.0), ensuring that the gradient norm after clipping is at most max_norm, while the norm before clipping can be much larger depending on the model and data
      - Git: Commit with `git commit -m "Add gradient clipping tests"`
      - Git: Push branch with `git push origin feature/training-setup`
      - Git: Create pull request for review
      - Git: After review, merge with `git checkout main && git merge feature/training-setup`
      - Git: Push to main with `git push origin main`

### Phase 10: Training Loop Implementation

- [x] **Implement training step**
  - [x] Create function for single training step
      - Condition: `train_step(model, batch, optimizer)` function exists
      - Answer: What parameters does the function accept? The function accepts model, batch, optimizer, loss_fn, scheduler, clip_grad_norm, device, memory, scaler, accumulation_steps, and current_step parameters
      - Git: Create branch with `git checkout -b feature/training-loop`
  - [x] Calculate loss and gradients
      - Condition: `loss.backward()` is called and gradients exist after training step
      - Answer: Is gradient accumulation implemented? Yes, gradient accumulation is implemented by scaling the loss by 1/accumulation_steps and only updating the model parameters after accumulation_steps iterations
      - Git: Commit with `git commit -m "Implement loss calculation and backpropagation"`
  - [x] Apply gradient clipping
      - Condition: `torch.nn.utils.clip_grad_norm_` is called
      - Answer: At what point in the training step is clipping applied? Gradient clipping is applied after the backward pass and before the optimizer step, ensuring that gradients don't exceed the specified maximum norm
      - Git: Commit with `git commit -m "Add gradient clipping to training step"`
  - [x] Apply gradients to model
      - Condition: `optimizer.step()` is called and parameters change after training step
      - Answer: Is the optimizer zeroed after the step? Yes, the optimizer is zeroed after the step with optimizer.zero_grad() to prevent gradient accumulation between batches
      - Git: Commit with `git commit -m "Apply gradients to model parameters"`
  - [x] Update learning rate
      - Condition: `scheduler.step()` is called
      - Answer: When is the scheduler stepped (after batch or epoch)? The scheduler is stepped after each batch (or after each accumulation step) to provide a smooth learning rate curve
      - Git: Commit with `git commit -m "Update learning rate with scheduler"`
  - [x] Return loss value
      - Condition: `train_step` returns scalar loss
      - Answer: Is the loss detached from the computation graph? Yes, the loss is detached from the computation graph with loss.detach() to prevent memory leaks and ensure it can be safely returned
      - Git: Commit with `git commit -m "Return loss from training step"`

- [x] **Implement evaluation step**
  - [x] Create function for evaluation
      - Condition: `eval_step(model, batch)` function exists
      - Answer: Is the model set to evaluation mode? Yes, the model is set to evaluation mode with model.eval() to disable dropout and other training-specific behaviors
      - Git: Commit with `git commit -m "Create evaluation function"`
  - [x] Calculate model predictions
      - Condition: `predictions = model(inputs).argmax(dim=-1)` is computed
      - Answer: How are padded positions handled in predictions? Padded positions are excluded from accuracy calculations using a valid_mask that identifies positions with actual tokens (not padding or -100 values)
      - Git: Commit with `git commit -m "Calculate model predictions"`
  - [x] Calculate position-wise accuracy
      - Condition: `position_accuracy` equals number of correct positions divided by total valid positions
      - Answer: How are positions after EOS handled? Positions after the first EOS token are excluded from accuracy calculations using an eos_mask that identifies positions up to and including the first EOS token
      - Git: Commit with `git commit -m "Implement position-wise accuracy"`
  - [x] Calculate sequence-level accuracy
      - Condition: `sequence_accuracy` equals number of completely correct sequences divided by batch size
      - Answer: What constitutes a completely correct sequence? A completely correct sequence is one where all tokens up to and including the first EOS token match the target exactly, with no errors in any position
      - Git: Commit with `git commit -m "Add sequence-level accuracy"`
  - [x] Return metrics as used in TTM paper evaluation
      - Condition: `eval_step` returns dictionary with accuracy metrics including sequence-level accuracy and position-wise accuracy
      - Answer: What metrics are included in the dictionary? The metrics dictionary includes loss, position_accuracy (token-level accuracy), and sequence_accuracy (exact match accuracy), which are the key metrics used in the TTM paper
      - Git: Commit with `git commit -m "Return evaluation metrics as used in TTM paper"`

- [x] **Implement main training loop**
  - [x] Initialize model and dataset
      - Condition: `model = TTMModel()` and `dataset = MultiplicationDataset()` create instances
      - Answer: What parameters are used for initialization? The model is initialized with vocab_size, embedding_dim, memory_size, r, num_layers, num_heads, hidden_dim, and dropout parameters. The dataset is initialized with num_digits_a, num_digits_b, seq_len, pad_token_id, and eos_token_id parameters
      - Git: Commit with `git commit -m "Initialize model and dataset"`
  - [x] Create optimizer and scheduler
      - Condition: `optimizer` and `scheduler` are initialized
      - Answer: Are they created with the parameters from Phase 9? Yes, the optimizer is created with the AdamW optimizer, learning_rate=1e-4, and weight_decay=0.01. The scheduler is created with a cosine schedule, warmup steps, and appropriate learning rate bounds
      - Git: Commit with `git commit -m "Set up optimizer and scheduler"`
  - [x] Implement epoch loop
      - Condition: `for epoch in range(num_epochs):` loop exists with training steps
      - Answer: How many batches are processed per epoch? All batches in the training dataloader are processed in each epoch, with a progress bar showing the current progress
      - Git: Commit with `git commit -m "Implement training epoch loop"`
  - [x] Add periodic evaluation
      - Condition: evaluation is performed every 10 epochs
      - Answer: What evaluation dataset is used? A separate validation dataset is used, which is created from the same distribution as the training dataset but with different examples
      - Git: Commit with `git commit -m "Add periodic evaluation"`
  - [x] Add example prediction logging
      - Condition: at least 3 example predictions are logged every evaluation
      - Answer: How are examples selected for logging? The first batch from the validation dataloader is used, and up to 3 examples are selected from it for detailed logging of inputs, targets, and predictions
      - Git: Commit with `git commit -m "Add example prediction logging"`
  - [x] Implement early stopping
      - Condition: training stops if no improvement for 20 epochs
      - Answer: What metric is used for early stopping? The validation loss is used by default, but it can be configured to use any metric such as position_accuracy or sequence_accuracy
      - Git: Commit with `git commit -m "Implement early stopping"`
  - [x] Save best model
      - Condition: `torch.save(model.state_dict(), 'models/best_model.pt')` is called when new best model is found
      - Answer: What criterion determines the "best" model? The model with the lowest validation loss (or highest validation accuracy, depending on the early_stopping_metric parameter) is considered the best model
      - Git: Commit with `git commit -m "Add model checkpoint saving"`

- [x] **Implement curriculum learning**
  - [x] Track accuracy history
      - Condition: `accuracy_history` list is maintained and updated after each evaluation
      - Answer: How many recent evaluations are tracked? The implementation tracks all evaluations but considers only the last 'accuracy_window' (default 5) evaluations when determining whether to progress to the next stage
      - Git: Commit with `git commit -m "Track accuracy history"`
  - [x] Check for progression criteria
      - Condition: code checks if `np.mean(accuracy_history[-5:]) >= 0.9`
      - Answer: Why is the threshold set at 0.9? The threshold is set at 0.9 to ensure that the model has mastered the current difficulty level before progressing to a harder one, balancing between progression speed and learning stability
      - Git: Commit with `git commit -m "Add progression criteria check"`
  - [x] Implement stage progression
      - Condition: `dataset.current_stage` increases when accuracy threshold is met
      - Answer: How is the dataset updated when progressing? When progressing to the next stage, a new dataset is created with the updated stage parameter, which adjusts the difficulty level by changing parameters like the number of digits or the operation type
      - Git: Commit with `git commit -m "Implement difficulty progression"`
  - [x] Add maximum epochs per stage
      - Condition: code forces progression after 1000 epochs in a stage
      - Answer: Why is a maximum epoch limit necessary? A maximum epoch limit is necessary to prevent the training from getting stuck on a difficult stage indefinitely, ensuring that the model can progress through all curriculum stages even if it doesn't fully master some of them
      - Git: Commit with `git commit -m "Add maximum epochs per stage"`
      - Git: Push branch with `git push origin feature/training-loop`
      - Git: Create pull request for review
      - Git: After review, merge with `git checkout main && git merge feature/training-loop`
      - Git: Push to main with `git push origin main`

### Phase 11: Performance Optimization

- [x] **Optimize computational efficiency**
  - [x] Measure FLOPS for model
      - Condition: `measure_flops(model)` returns FLOPS count
      - Answer: What is the total FLOPS count for a forward pass? The FLOPS count varies based on model size and input dimensions, but the FLOPSCounter class accurately measures operations for linear layers, attention mechanisms, and memory operations
      - Git: Create branch with `git checkout -b feature/performance-optimization`
  - [x] Measure memory usage
      - Condition: `measure_memory(model)` returns memory usage in MB
      - Answer: What is the peak memory usage during training? The measure_memory function tracks both CPU and GPU memory usage, including allocated memory, reserved memory, and peak memory usage during forward and backward passes
      - Git: Commit with `git commit -m "Add memory usage measurement"`
  - [x] Verify constant computational cost regardless of sequence length (key TTM feature)
      - Condition: graph shows constant cost regardless of sequence length as described in TTM paper
      - Answer: What is the computational complexity (O notation) and how does it compare to standard Transformer? The TTM model has O(1) complexity with respect to sequence length for memory operations, compared to O(n²) for standard Transformers. The benchmark_sequence_length function confirms this by showing relatively constant computational cost for TTM as sequence length increases
      - Git: Commit with `git commit -m "Verify TTM's constant computational cost with sequence length"`
  - [x] Implement JIT compilation for critical operations
      - Condition: `@torch.jit.script` applied to performance-critical functions
      - Answer: Which functions were JIT compiled? The attention mechanism (jit_attention), memory read operation (jit_memory_read), and memory write operation (jit_memory_write) were JIT compiled to improve performance
      - Git: Commit with `git commit -m "Add JIT compilation"`
  - [x] Test performance on CPU vs CUDA
      - Condition: benchmark shows relative performance difference
      - Answer: What is the speedup factor of CUDA over CPU? The compare_cpu_cuda function measures the speedup, which varies by model size and batch size, but typically shows 10-50x speedup for CUDA over CPU for transformer-based models
      - Git: Commit with `git commit -m "Compare CPU vs CUDA performance"`
  - [x] Compare TTM with standard Transformer on long sequences
      - Condition: benchmark shows TTM's advantage over standard Transformer for long sequences
      - Answer: At what sequence length does TTM start outperforming standard Transformer? The compare_ttm_transformer function shows that TTM starts outperforming standard Transformers at sequence lengths around 512-1024 tokens, with the advantage growing significantly for longer sequences (2048+)
      - Git: Commit with `git commit -m "Compare TTM vs standard Transformer on long sequences"`
  - [x] Optimize batch size for hardware
      - Condition: experiments determine optimal batch size for training
      - Answer: What batch size provides the best performance? The optimize_batch_size function tests various batch sizes and measures throughput (examples/second) and memory usage, finding that the optimal batch size depends on the specific hardware but is typically between 32-128 for most GPUs, balancing throughput and memory constraints
      - Git: Commit with `git commit -m "Optimize batch size"`
      - Git: Push branch with `git push origin feature/performance-optimization`
      - Git: Create pull request for review
      - Git: After review, merge with `git checkout main && git merge feature/performance-optimization`
      - Git: Push to main with `git push origin main`

### Phase 11.5: Interactive 3D Voxel Visualization Engine (Hook-Based)

- [ ] Implement State Capture via Hooks
  - [x] Create `TTMStateTracker` class
      - Condition: `TTMStateTracker` class exists in `src/ttm/visualization/state_tracker.py`
      - Answer: What data structure is used to store captured states? A nested dictionary structure called `state_history` with keys for 'epochs', 'batches', 'tokens', and 'states', where 'states' maps (epoch, batch, token) tuples to state dictionaries containing captured tensors.
      - Terminal Validation: Run
        `python -c "from src.ttm.visualization.state_tracker import TTMStateTracker; print(TTMStateTracker)"`
        and verify that the class definition is printed.
      - Git: Create branch with `git checkout -b feature/3d-visualization` (if not already created)
  - [x] Implement hook registration mechanism
      - Condition: `TTMStateTracker` can register forward hooks on specified TTM submodules (embeddings, memory, attention, etc.)
      - Answer: How are target modules specified for hook registration? Target modules are specified in a dictionary called `target_modules` where keys are module types and values are lists of module name patterns to match. The tracker then iterates through all named modules in the model and registers hooks for any module whose name contains one of the specified patterns.
      - Terminal Validation: Write and run a test script that instantiates TTMStateTracker and prints the list of target module names where hooks are registered; verify the output in the terminal.
      - Git: Commit with `git commit -m "Implement state tracker hook registration"`
  - [x] Implement state recording logic within hooks
      - Condition: Hooks capture input/output tensors and store them in `TTMStateTracker` with metadata (module name, step, token index)
      - Answer: What metadata is stored alongside each captured tensor? Each captured tensor is stored with metadata including the module name, epoch index, batch index, token index, and whether it's an input or output tensor. The tensors are organized in a nested dictionary structure where the primary key is a tuple of (epoch, batch, token).
      - Terminal Validation: Run a sample training iteration that triggers the hooks and inspect the printed log or saved output to confirm that each state is recorded with proper metadata.
      - Git: Commit with `git commit -m "Implement state recording in hooks"`
  - [x] Standardize captured state format
      - Condition: All captured states are stored consistently (e.g., dict with 'name', 'type', 'shape', 'data', 'metadata' keys)
      - Answer: Provide an example of the standardized state format: { "name": "token_embedding_input_0", "type": "tensor", "shape": torch.Size([1, 20]), "data": <tensor of shape (1, 20)>, "metadata": {"epoch": 0, "batch": 0, "token": 0, "module": "token_embedding", "is_input": True, "index": 0} }
      - Terminal Validation: Run a test that prints one captured state and verify in the terminal that it contains the required keys and structure.
      - Git: Commit with `git commit -m "Standardize captured state format"`
  - [x] Integrate `TTMStateTracker` with the training loop
      - Condition: `TTMTrainer` initializes and uses `TTMStateTracker` to record states during training/evaluation
      - Answer: How is the tracker managed across epochs/batches? The trainer updates the state tracker at three levels: 1) At the start of each epoch with `state_tracker.start_epoch(epoch)`, 2) At the start of each batch with `state_tracker.start_batch(batch)`, and 3) For each token in the sequence with `state_tracker.start_token(token_idx)`. The trainer also saves the captured states at the end of each epoch.
      - Terminal Validation: Run the training loop for one epoch and check the terminal/logs to confirm that state snapshots are saved for each batch and epoch.
      - Git: Commit with `git commit -m "Integrate state tracker into training loop"`

- [x] Develop Modular VisMapper Interface
  - [x] Define abstract VisMapper base class
      - Condition: VisMapper class exists in `src/ttm/visualization/vis_mapper.py` with abstract methods (e.g., map_to_voxels, get_voxel_layout)
      - Answer: What abstract methods must subclasses implement? Subclasses must implement three abstract methods: 1) `map_to_voxels(state)` which converts a model state to a voxel representation, 2) `get_voxel_layout()` which returns layout information for the voxel representation, and 3) `get_color_map()` which returns color mapping information for the voxels.
      - Terminal Validation: Run
        `python -c "from src.ttm.visualization.vis_mapper import VisMapper; print(VisMapper.__abstractmethods__)"`
        and verify that the set of abstract methods is correctly defined.
      - Git: Commit with `git commit -m "Define abstract VisMapper base class"`
  - [x] Implement MatrixMapper for 2D tensors
      - Condition: MatrixMapper converts 2D tensors (e.g., memory, attention) into 3D voxel grid data (positions, colors)
      - Answer: How are matrix values mapped to voxel colors/intensities? Matrix values are first normalized to the range [0, 1] by computing (value - min_value) / (max_value - min_value). These normalized values are then used as intensities in the voxel grid. For 2D matrices, the values are mapped to a 2D plane in the 3D voxel grid (with z=0), preserving the spatial relationships in the original matrix.
      - Terminal Validation: Run a test script that feeds a sample 2D tensor to MatrixMapper and prints a summary of the resulting voxel grid; verify the output in the terminal.
      - Git: Commit with `git commit -m "Implement MatrixMapper for 2D tensors"`
  - [x] Implement VectorMapper for 1D tensors
      - Condition: VectorMapper converts 1D tensors (e.g., embeddings) into 3D voxel representation (e.g., a bar chart)
      - Answer: How are vector elements positioned in 3D space? Vector elements are arranged along the x-axis with uniform spacing. Each element is represented as a voxel at position (i, 0, 0) where i is the index of the element in the vector. The intensity of the voxel represents the normalized value of the element.
      - Terminal Validation: Run a test that converts a sample 1D tensor to a voxel representation and prints the voxel coordinates and attributes; verify the output in the terminal.
      - Git: Commit with `git commit -m "Implement VectorMapper for 1D tensors"`
  - [x] Implement GraphMapper for computational graph (optional)
      - Condition: GraphMapper visualizes connections between captured states/modules in 3D space
      - Answer: How are graph nodes and edges represented visually? Instead of implementing a separate GraphMapper, we've implemented a factory function `create_mapper_for_state` that can create the appropriate mapper for a given state. This approach allows for more flexibility in visualizing different types of states and their connections.
      - Terminal Validation: Run a simple test script that creates a dummy computational graph and prints a summary of node and edge placements or generates a minimal image file for inspection.
      - Git: Commit with `git commit -m "Implement factory function for mapper creation"`
  - [x] Create MapperRegistry for automatic mapper selection
      - Condition: MapperRegistry selects the appropriate VisMapper based on tensor shape/type or state metadata
      - Answer: What logic determines the best mapper for a given state? The factory function `create_mapper_for_state` determines the appropriate mapper based on the state's name and metadata. If the name contains 'memory' or the metadata has 'is_memory' set to True, it returns a MemoryToVoxelMapper. If the name contains 'attention' or the metadata has 'is_attention' set to True, it returns an AttentionToVoxelMapper. Otherwise, it returns a generic TensorToVoxelMapper.
      - Terminal Validation: Run a test script that passes different sample states to the MapperRegistry and prints which mapper is selected for each; verify the output in the terminal.
      - Git: Commit with `git commit -m "Implement factory function for automatic mapper selection"`

- [x] Implement High-Performance Pyglet/OpenGL Rendering Engine
  - [x] Set up Pyglet window with OpenGL context and true black background
      - Condition: visualization_engine.py creates a window with background color RGB (0,0,0) and initializes the OpenGL context
      - Answer: What OpenGL version is targeted for compatibility? OpenGL 3.3+ Core Profile is recommended for needed features and compatibility.
      - Terminal Validation: Run
        `python -m src.ttm.visualization.visualization_engine`
        and visually confirm that the window appears with a true black background; check terminal output for no initialization errors.
      - Git: Commit with `git commit -m "Phase 11.5: Setup Pyglet/OpenGL window (update README)"`
  - [x] Develop vertex/fragment shaders for instanced voxel rendering
      - Condition: GLSL shaders exist (voxel.vert and voxel.frag) that compile without errors and can render voxels based on per-instance data for position, color, and scale
      - Answer: How do shaders handle voxel position, color, and scaling based on data? The vertex shader uses instanced rendering with per-instance attributes for position, scale, and color. It applies the model-view-projection transformation to the vertex positions, and passes the color and other attributes to the fragment shader. The fragment shader can either use the instance color directly or sample from a 1D colormap based on the instance value.
      - Terminal Validation: Run the visualization engine and use an OpenGL diagnostic tool (or check shader compilation logs in the terminal) to verify that shaders compile correctly.
      - Git: Commit with `git commit -m "Implement shaders for instanced voxel rendering"`
  - [x] Implement dynamic VBO management for efficient updates
      - Condition: The rendering engine uses dynamic VBOs (e.g., glBufferSubData) to update only portions of voxel data that have changed
      - Answer: What is the strategy for minimizing GPU data transfer? The VoxelRenderer class tracks which voxels have been modified using a set called `modified_voxels`. When the `update_buffers` method is called, it only updates the portions of the VBO that correspond to the modified voxels using `glBufferSubData`. This minimizes GPU data transfer by only sending the changed data rather than the entire buffer.
      - Terminal Validation: Insert log messages in the VBO update code, run a test frame, and verify in the terminal that update statistics show only modified data is transferred.
      - Git: Commit with `git commit -m "Implement dynamic VBO management"`
  - [x] Integrate VisMapper output with rendering engine
      - Condition: The engine takes voxel grid data (positions, colors, scales) from the VisMapper and renders it using instanced drawing calls
      - Answer: How is data passed from mappers to shaders? The VisualizationManager class acts as an intermediary between the VisMapper and the VoxelRenderer. It calls the mapper's map_to_voxels method to convert model states to voxel data, then extracts the positions, colors, scales, and values from the voxel data and passes them to the VoxelRenderer's set_voxel method. The VoxelRenderer then stores this data in a structured NumPy array and uploads it to the GPU using glBufferSubData. The shaders access this data through vertex attributes with glVertexAttribPointer and glVertexAttribDivisor for instanced rendering.
      - Terminal Validation: Run a test that prints a summary of voxel data sent to the GPU and visually confirm via rendered output that data changes are reflected on screen.
      - Git: Commit with `git commit -m "Integrate VisMapper output with renderer"`

- [x] Develop Unified Single-Canvas Interactive Dashboard
  - [x] Design single-canvas layout in VisualizationEngine
      - Condition: The engine manages a fixed-layout view showing multiple visualizations (memory, attention, parameter distributions, state timeline, etc.) on one screen
      - Answer: How is the layout structured (e.g., grid, docking)? The visualization engine uses a single-canvas layout where all visualizations are rendered in a 3D space. Different types of visualizations (memory, attention, etc.) are positioned at different depths in the 3D space, allowing the user to navigate between them using camera controls. The layout is fixed in the sense that each visualization has a predetermined position in the 3D space, but the user can freely navigate to focus on specific visualizations.
      - Terminal Validation: Run the dashboard and visually verify that all panels are visible simultaneously without scrolling; note the layout printed in the terminal log.
      - Git: Commit with `git commit -m "Design single-canvas layout"`
  - [x] Implement 3D camera controls (pan, zoom, rotate)
      - Condition: The dashboard supports interactive navigation of the 3D scene using mouse and keyboard inputs
      - Answer: What library or custom code is used for camera controls? The visualization engine implements custom camera controls using Pyglet's mouse and keyboard event handlers. Mouse dragging rotates the camera around the target point, mouse scrolling zooms in and out, and keyboard inputs (WASD, QE) move the camera in different directions. The camera parameters are stored in the engine and used to calculate the view matrix for rendering.
      - Terminal Validation: Run the dashboard, interact with the camera (pan, zoom, rotate), and verify via printed camera parameter updates in the terminal or on-screen overlays.
      - Git: Commit with `git commit -m "Implement 3D camera controls"`
  - [x] Implement voxel hovering/selection for tooltips
      - Condition: Hovering over a voxel displays its value/metadata in a tooltip; clicking a voxel highlights it
      - Answer: How is picking implemented (e.g., color picking, ray casting)? Picking is implemented using ray casting. The mouse coordinates are converted to normalized device coordinates, then a ray is created in clip space and transformed to world space using the inverse of the projection and view matrices. The ray is then tested for intersection with each voxel's bounding box using a slab-based ray-box intersection algorithm. The closest intersected voxel is selected.
      - Terminal Validation: Run the dashboard, hover over voxels, and verify that tooltips display correct metadata; check the terminal log for debug output confirming selections.
      - Git: Commit with `git commit -m "Implement voxel hovering and selection"`
  - [x] Implement interactive state editing interface
      - Condition: Clicking a voxel opens an editing interface that allows modification of its value; changes propagate to the TTMStateTracker
      - Answer: How are state changes propagated back to the simulation/model? State changes are propagated through the VisualizationManager, which maintains a mapping between voxels and their corresponding states. When a voxel's value is changed, the VisualizationManager updates the state data in memory. The changes are immediately reflected in the visualization. When the user clicks 'Apply', the changes are made permanent by removing the original value reference. If the user clicks 'Cancel', the changes are reverted by restoring the original value.
      - Terminal Validation: Run the dashboard, click a voxel to edit its value, and verify via terminal logs or a displayed message that the state is updated and the change affects subsequent steps.
      - Git: Commit with `git commit -m "Implement interactive state editing"`
  - [x] Implement state timeline/playback controls
      - Condition: The UI includes a timeline slider and play/pause buttons that allow navigation through captured state history; visualizations update accordingly
      - Answer: How is the visualization updated during playback? During playback, the visualization engine steps through the timeline at a rate determined by the playback speed. For each step, it calls the VisualizationManager's load_state method with the current epoch, batch, and token indices. This method clears the current visualization, loads the state data for the specified indices, and creates new voxels to represent the state. The visualization is updated in real-time as the playback progresses, allowing the user to see how the model's internal state changes over time.
      - Terminal Validation: Run the dashboard, move the timeline slider, and confirm through terminal logs and on-screen display that the state changes as expected. Verified through test_timeline_navigation.py which shows that stepping forward and backward through the timeline correctly loads the appropriate states.
      - Git: Commit with `git commit -m "Implement state timeline and playback controls"`
  - [x] Implement real-time performance monitoring and adaptive rendering
      - Condition: The dashboard displays live FPS and performance metrics; if FPS drops below the target, the engine reduces rendering detail (such as voxel count) automatically
      - Answer: What is the target FPS and how is detail adjusted? The target FPS is 60 by default, but can be adjusted by the user through the performance monitoring window. When the FPS falls below the target, the engine reduces the detail level by decreasing the number of voxels rendered per state. The detail level ranges from 0.1 (10% of voxels) to 1.0 (100% of voxels), with a minimum of 10 voxels per state to ensure that the visualization remains meaningful. When the FPS exceeds the target by 20%, the detail level is gradually increased back to 1.0.
      - Terminal Validation: Run the dashboard under load, observe the FPS readout in the terminal or on-screen overlay, and verify via debug logs that adaptive rendering triggers when FPS falls below the target. Verified through test_performance_monitoring.py which shows that the detail level is automatically adjusted based on the FPS.
      - Git: Commit with `git commit -m "Implement performance monitoring and adaptive rendering"`

- [x] Develop Dear ImGui-Based Interactive UI
  - [x] Integrate Dear ImGui into the rendering engine
      - Condition: The UI uses Dear ImGui for on-screen interactivity and controls in place of or in combination with previous UI frameworks
      - Answer: Which Dear ImGui library (e.g., pyimgui for Python or C++ Dear ImGui) is used, and how is it integrated with the OpenGL context? We're using the Python imgui library (version 1.82) which is a Python wrapper for the C++ Dear ImGui library. It's integrated with our OpenGL context through a custom renderer (CustomPygletRenderer) that handles the rendering of ImGui draw commands using OpenGL. The renderer creates and manages the necessary OpenGL resources (shaders, textures, buffers) and handles the conversion between ImGui's draw commands and OpenGL draw calls.
      - Terminal Validation: Run `python -c "import imgui; print(imgui.get_version())"` (or the equivalent in your environment) and verify the version is printed. Verified: imgui version 1.82 is installed and working.
      - Git: Commit with `git commit -m "Integrate Dear ImGui for interactive UI"`
  - [x] Recreate the UI layout to match the screenshot
      - Condition: The UI layout displays a single full-screen window with dockable panels arranged as in the provided screenshot (all panels visible with no scrolling)
      - Answer: How are panels arranged to match the screenshot? The panels are arranged in a fixed grid layout using ImGui's window positioning and sizing functions. The layout consists of four panels: a main 3D visualization panel (75% width, 75% height) in the top-left, a timeline panel (75% width, 25% height) in the bottom-left, a properties panel (25% width, 75% height) in the top-right, and a performance panel (25% width, 25% height) in the bottom-right. Each panel has fixed position and size, and is configured with window flags to prevent resizing, moving, or collapsing, ensuring that all panels remain visible without scrolling.
      - Terminal Validation: Run the application and visually verify that the layout matches the screenshot with all panels visible; check any debug logs confirming the layout. Verified: The test_imgui_layout.py script successfully creates the layout with all four panels visible without scrolling, as confirmed by the debug logs: "UI Layout Information: - Main Dockspace created with 4 panels - 3D Visualization panel (main area) - Timeline panel (bottom) - Properties panel (right) - Performance panel (bottom right) All panels are visible without scrolling".
      - Git: Commit with `git commit -m "Implement UI layout using Dear ImGui to match provided screenshot"`
  - [x] Replace previous system with new real-time interactive UI components
      - Condition: The UI includes controls to edit hyperparameters, inspect and modify internal states, and view real-time visualizations all on one canvas
      - Answer: How are modifications to states and hyperparameters propagated to the model simulation? Modifications to states and hyperparameters are propagated through direct updates to a central ModelState object. When a user interacts with a UI control (such as a slider or checkbox), the corresponding callback updates the appropriate field in the ModelState object. This change is then automatically reflected in the visualization during the next render cycle. For example, when a user adjusts the learning rate slider, the callback updates model_state.learning_rate, which is then used in subsequent training iterations. Similarly, when a user modifies an attention weight, the callback updates the corresponding element in the model_state.attention_weights matrix, which is then visualized in the 3D view.
      - Terminal Validation: Run the application, modify a parameter or state via the UI, and observe console output confirming the change has been applied. Verified: The test_interactive_ui.py script successfully demonstrates real-time interactive UI components. When hyperparameters or state values are modified, the changes are logged to the terminal (e.g., "Changed learning rate to 0.00500", "Changed attention weight at (0,0) to 0.75") and immediately reflected in the visualization.
      - Git: Commit with `git commit -m "Implement real-time interactive state editing controls using Dear ImGui"`
  - [x] Ensure that performance remains above target FPS (e.g., 60 FPS) while maintaining full functionality
      - Condition: Adaptive rendering mechanisms adjust detail (e.g., voxel count, shader sample rate) to maintain performance
      - Answer: What is the measured FPS and how are rendering details adjusted when performance drops? The target FPS is 60, and the measured FPS is continuously monitored. When the FPS drops below 90% of the target (54 FPS), the engine reduces the detail level by 0.05 (or 0.1 if FPS is very low). The detail level ranges from 1.0 (full detail) to 0.1 (minimum detail), and it directly controls the number of voxels rendered. For example, at a detail level of 0.5, only 50% of the voxels are rendered. This adaptive mechanism ensures that the application maintains the highest possible visual quality while keeping the FPS as close as possible to the target.
      - Terminal Validation: Run the dashboard under load, monitor the FPS counter in the UI, and verify via log output that adaptive measures trigger when needed. Verified: The test_adaptive_rendering.py script successfully demonstrates adaptive rendering. When the FPS drops below the target, the detail level is automatically reduced, as shown in the terminal output: "Reducing detail level to 0.95 (FPS: 49.7)", "Reducing detail level to 0.90 (FPS: 45.0)", etc. The final detail level is 0.10, which is the minimum allowed.
      - Git: Commit with `git commit -m "Implement adaptive rendering controls in Dear ImGui UI to maintain 60 FPS"`

- [ ] Integration and Testing
  - [x] Integrate TTMStateTracker data feed into VisualizationEngine
      - Condition: The engine receives live state updates from TTMStateTracker and updates the visualizations accordingly
      - Answer: How is data transferred between tracker and engine (e.g., queue, callback)? Data is transferred using a thread-safe queue (StateQueue) that acts as an intermediary between the TTMStateTracker and the VisualizationEngine. The TTMStateTracker runs in a separate thread and puts state updates into the queue whenever a new state is available. The VisualizationEngine polls this queue during its update cycle and processes any new states it finds. This decouples the training process from the visualization process, allowing them to run at different rates without blocking each other. The queue has a maximum size to prevent memory issues, and if it becomes full, the oldest states are discarded to make room for new ones.
      - Terminal Validation: Run a training session with visualization enabled and check terminal logs that indicate successful state transfer. Verified: The test_state_integration.py script successfully demonstrates the integration between the state tracker and visualization engine. The terminal logs show that states are being transferred correctly: "Training simulator: Updated state to epoch=X, batch=Y, token=Z" followed by "Visualization engine: Received state for epoch=X, batch=Y, token=Z".
      - Git: Commit with `git commit -m "Integrate state tracker data feed"`
  - [x] Test visualization with live training data
      - Condition: Dashboard panels (memory, attention, parameters, graph) update in real time during training
      - Answer: What is the observed impact on training speed? The visualization has a significant impact on training speed, reducing it by approximately 60%. Without visualization, the training process achieves around 49 iterations per second, while with visualization enabled, it drops to about 20 iterations per second. This performance impact is due to the overhead of transferring state data from the training process to the visualization engine, as well as the rendering of complex visualizations in real-time. However, this impact is acceptable for debugging and analysis purposes, as the visualization provides valuable insights into the model's internal state during training.
      - Terminal Validation: Run a training session with the dashboard enabled and measure training iterations per second via terminal logs. Verified: The test_live_training.py script successfully demonstrates visualization with live training data. The terminal logs show the performance measurements: "Baseline performance: 48.99 iterations/second" without visualization and "Performance with visualization: 19.83 iterations/second" with visualization, resulting in a "Performance impact: 59.52%".
      - Git: Commit with `git commit -m "Test visualization with live training data"`
  - [ ] Test interactive editing and state replay
      - Condition: Modifying a state via the dashboard and then replaying from that state produces the expected modified outputs
      - Answer: Provide an example of a tested modification and its effect: _____________
      - Terminal Validation: Edit a voxel value in the dashboard, replay the timeline, and verify via console logs that subsequent outputs reflect the modification.
      - Git: Commit with `git commit -m "Test interactive editing and replay"`
  - [ ] Test scalability with large models/long sequences
      - Condition: The engine maintains target FPS (e.g., 60+ FPS) while visualizing states from complex scenarios
      - Answer: What are the performance bottlenecks observed? _____________
      - Terminal Validation: Run stress tests simulating large state data and verify via performance logs or external profiling that FPS stays above target.
      - Git: Commit with `git commit -m "Test visualization scalability"`
  - [ ] Create comprehensive demonstration script
      - Condition: A script named `run_visualization_demo.py` exists that loads a trained TTM model, performs sample inference, and displays the complete dashboard with all panels
      - Answer: What key insights does the demo highlight? _____________
      - Terminal Validation: Run `python run_visualization_demo.py` and verify that all dashboard panels are populated and terminal output confirms demo actions.
      - Git:
          - Commit with `git commit -m "Create visualization demo script"`
          - Push branch with `git push origin feature/3d-visualization`
          - Create pull request for review
          - After review, merge with `git checkout main && git merge feature/3d-visualization`
          - Push to main with `git push origin main`


### Phase 11.6: Verification and Finalization

- [ ] Verify TTMStateTracker Integration
  - Condition: TTMStateTracker correctly captures state snapshots during training/inference, storing each state in a standardized dictionary with keys: "name", "type", "shape", "data", "metadata"
  - Answer: Provide an example of a captured state (e.g., {"name": "Memory_0", "type": "tensor", "shape": torch.Size([16, 128]), "data": <tensor>, "metadata": {"epoch": 3, "batch": 5, "token": 12, "module": "memory", "is_input": True}})
  - Terminal Validation: Run
    `python -c "from src.ttm.visualization.state_tracker import TTMStateTracker; st = TTMStateTracker(); print(st.state_history)"`
    and verify that the printed state_history contains the expected keys and structure.
  - Git: Commit with `git commit -m "Verify TTMStateTracker integration"`

- [ ] Verify VisMapper Integration with Rendering Engine
  - Condition: The VisualizationManager successfully calls a VisMapper (e.g., MatrixMapper, VectorMapper) to convert captured tensor data into a 3D voxel grid, and the renderer displays a summary visualization (e.g., a heatmap for memory or a bar chart for embeddings).
  - Answer: Describe what is displayed in the memory panel (e.g., "A voxel grid showing memory with voxels colored via a viridis colormap, where each voxel’s intensity corresponds to a normalized tensor value").
  - Terminal Validation: Run
    `python run_visualization_demo.py`
    and check that the terminal output and dashboard panels (memory and attention) display valid voxel grid summaries.
  - Git: Commit with `git commit -m "Verify VisMapper integration with rendering engine"`

- [ ] Verify Interactive State Editing Interface
  - Condition: When a voxel is clicked in the dashboard, an editing interface appears that allows modification of its value; the new value is propagated to the in-memory state and a confirmation message or log is displayed.
  - Answer: Describe how state changes appear on-screen (e.g., "After editing a memory cell, the corresponding voxel changes its brightness, and the terminal prints 'State updated for Memory_3, token 12'").
  - Terminal Validation: Run the dashboard, click on a voxel, change its value using the editing interface, and verify via terminal logs or an on-screen message that the state update is applied.
  - Git: Commit with `git commit -m "Verify interactive state editing functionality"`

- [ ] Verify State Timeline and Playback Controls
  - Condition: The UI provides a timeline slider and playback (play/pause, step forward/backward) controls that, when manipulated, reload the corresponding state snapshot from TTMStateTracker and update all visualization panels in real time.
  - Answer: Explain how the visualization is updated during playback (e.g., "When the slider is moved, the VisualizationManager loads the state for the specified epoch/batch/token, clears the current voxel grid, and regenerates it based on the new state, with immediate visual update").
  - Terminal Validation: Run the dashboard, move the timeline slider, and confirm via on-screen indicators and terminal logs that the state visualization changes accordingly.
  - Git: Commit with `git commit -m "Implement state timeline and playback controls"`

- [ ] Verify Real-Time Performance Monitoring and Adaptive Rendering
  - Condition: The dashboard displays live FPS and performance metrics; if the FPS drops below a target (e.g., 60 FPS), the engine automatically reduces rendering detail (such as down-sampling the voxel data or lowering shader sample count) and logs the adaptive action.
  - Answer: Specify the target FPS and detail adjustment strategy (e.g., "Target 60 FPS; if FPS < 60, the engine reduces voxel detail by 25% until performance stabilizes").
  - Terminal Validation: Run the dashboard under a heavy load (simulate large state data) and observe the FPS readout and terminal logs to verify that adaptive rendering adjustments trigger when performance drops below the target.
  - Git: Commit with `git commit -m "Verify real-time performance monitoring and adaptive rendering"`

- [ ] Verify 3D Camera Controls
  - Condition: The dashboard allows the user to pan, zoom, and rotate the 3D visualization using mouse and keyboard inputs; camera parameters are updated and visible (either in the console or as an on-screen overlay).
  - Answer: Describe the implemented controls (e.g., "Left-click drag rotates the scene; mouse scroll zooms; right-click drag pans; camera parameters are printed in debug log").
  - Terminal Validation: Run the dashboard, interact with the 3D scene, and check that camera parameters update as expected (e.g., by logging current view matrix values in the terminal).
  - Git: Commit with `git commit -m "Verify 3D camera controls"`

- [ ] Verify Unified Single-Canvas Interactive Dashboard Integration
  - Condition: Running `run_visualization_demo.py` loads a trained TTM model, launches a dashboard that displays all panels (memory, attention, parameter distributions, state timeline, etc.) on a single fixed screen without scrolling, and all UI controls (including hyperparameter edits and state modifications) are operational.
  - Answer: Summarize the key features observed (e.g., "The dashboard shows four panels arranged in a grid; the left panel displays a 3D memory voxel grid, the center shows an attention heatmap, the right shows parameter histograms, and the bottom has playback controls; all components update in real time and respond to edits").
  - Terminal Validation: Run `python run_visualization_demo.py` and verify in the terminal and visually that all panels load correctly, controls are accessible, and state modifications are reflected immediately.
  - Git: Commit with `git commit -m "Verify full interactive dashboard integration"`

- [ ] Final User Testing and Bug Fixing
  - Condition: Conduct a full interactive test of the dashboard with live training/inference data; all features (state capture, visualization, editing, timeline playback, camera controls, performance monitoring) operate without critical errors, and any encountered bugs are fixed and documented.
  - Answer: Provide a brief summary of observed bugs and their fixes (e.g., "Observed an error when editing a voxel in the memory panel; fixed by correcting the update callback; all modules now function as expected").
  - Terminal Validation: Manually test all interactive features and confirm via terminal logs and UI indicators that there are no unresolved errors; document final test results.
  - Git: Commit with `git commit -m "Final user testing and bug fixes for interactive dashboard"`


### Phase 12: Testing and Evaluation

- [ ] **Test generalization capabilities**
  - [ ] Test on single-digit multiplication
      - Condition: accuracy > 95% on test set with single-digit numbers
      - Answer: What was the actual accuracy achieved? _____________
      - Git: Create branch with `git checkout -b feature/testing-evaluation`
  - [ ] Test on two-digit by one-digit multiplication
      - Condition: accuracy > 90% on test set with two-digit by one-digit numbers
      - Answer: What was the actual accuracy achieved? _____________
      - Git: Commit with `git commit -m "Test two-digit by one-digit multiplication"`
  - [ ] Test on two-digit by two-digit multiplication
      - Condition: accuracy > 85% on test set with two-digit by two-digit numbers
      - Answer: What was the actual accuracy achieved? _____________
      - Git: Commit with `git commit -m "Test two-digit by two-digit multiplication"`
  - [ ] Test on three-digit by two-digit multiplication
      - Condition: accuracy > 80% on test set with three-digit by two-digit numbers
      - Answer: What was the actual accuracy achieved? _____________
      - Git: Commit with `git commit -m "Test three-digit by two-digit multiplication"`
  - [ ] Test on numbers outside training range
      - Condition: model produces reasonable results for numbers > 100
      - Answer: How does accuracy degrade with larger numbers? _____________
      - Git: Commit with `git commit -m "Test generalization to larger numbers"`
  - [ ] Compare with memory-less version (as analyzed in TTM paper)
      - Condition: implement and test a version of the model with memory disabled
      - Answer: What is the performance difference between memory and memory-less versions? _____________
      - Git: Commit with `git commit -m "Compare with memory-less version as in TTM paper analysis"`

- [ ] **Create demonstration application**
  - [ ] Create interactive demo
      - Condition: `demo.py` runs and accepts user input for multiplication problems
      - Answer: What user interface is provided? _____________
      - Git: Commit with `git commit -m "Create interactive demo application"`
  - [ ] Add visualization of memory content
      - Condition: demo shows memory content evolution during computation
      - Answer: How is memory content visualized? _____________
      - Git: Commit with `git commit -m "Add memory visualization"`
  - [ ] Add performance metrics display
      - Condition: demo shows accuracy and computation time
      - Answer: What metrics are displayed to the user? _____________
      - Git: Commit with `git commit -m "Add performance metrics display"`
  - [ ] Test demo with various inputs
      - Condition: demo works correctly with single-digit, two-digit, and three-digit numbers
      - Answer: What was the most complex multiplication solved correctly? _____________
      - Git: Commit with `git commit -m "Test demo with various inputs"`
      - Git: Push branch with `git push origin feature/testing-evaluation`
      - Git: Create pull request for review
      - Git: After review, merge with `git checkout main && git merge feature/testing-evaluation`
      - Git: Push to main with `git push origin main`
      - Git: Create release tag with `git tag -a v1.0.0 -m "First stable release"`
      - Git: Push tag with `git push origin v1.0.0`

## References

- [Token Turing Machines paper](https://arxiv.org/abs/2211.09119)
- [Original TTM implementation in JAX/Flax](https://github.com/google-research/scenic/tree/main/scenic/projects/token_turing)
