# Deploy & Scripts TurboQuant Mac

```bash
# Package install
pip install -e .

# Launch compressed OpenAI-compatible Server 
python3 scripts/run_server.py --model mlx-community/Meta-Llama-3-8B-Instruct-4bit

# Run generation tests across 5 models
PYTHONPATH=. python3 scripts/run_needle_test.py
```
