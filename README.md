# memoryweave

**Agent memory that learns — persistent, searchable, and self-organizing memory for AI agents**

![Build](https://img.shields.io/badge/build-passing-brightgreen) ![License](https://img.shields.io/badge/license-proprietary-red)

## Install
```bash
pip install -e ".[dev]"
```

## Quick Start
```python
from src.core import Memoryweave
 instance = Memoryweave()
r = instance.store(input="test")
```

## CLI
```bash
python -m src status
python -m src run --input "data"
```

## API
| Method | Description |
|--------|-------------|
| `store()` | Store |
| `retrieve()` | Retrieve |
| `consolidate()` | Consolidate |
| `link_memories()` | Link memories |
| `score_importance()` | Score importance |
| `prune_old()` | Prune old |
| `get_stats()` | Get stats |
| `reset()` | Reset |

## Test
```bash
pytest tests/ -v
```

## License
(c) 2026 Officethree Technologies. All Rights Reserved.
