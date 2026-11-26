# Task 1 — Creating Embeddings

## Run
1. Create and activate a Python virtual environment.
2. Install dependencies: `pip install -r requirements.txt`
3. Set your OpenAI key in PowerShell: `$env:OPENAI_API_KEY="your-key"`
4. Open `notebook.ipynb` in VS Code or Jupyter and run all cells.

## Expected checks
- Local output shape: `(8, 384)`.
- OpenAI output shape: `(8, 1536)` for the default `text-embedding-3-small` dimensions.
- The 100-sentence timing values depend on your computer, model download/cache, network, and API service. They cannot be guaranteed to equal the illustrative example in the assignment.
- API cost is calculated from returned token usage. Confirm the model's current price and update `PRICE_PER_MILLION_TOKENS_USD` if needed.
