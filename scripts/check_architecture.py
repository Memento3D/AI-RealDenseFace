"""Check internal dependencies without importing optional model SDKs."""
import ast
from pathlib import Path

root = Path(__file__).resolve().parents[1] / 'src' / 'memento3d_head'
errors = []
for layer in ('domain','application','pipelines'):
    for file in (root/layer).rglob('*.py'):
        tree = ast.parse(file.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            names = []
            if isinstance(node,ast.Import):
                names = [item.name for item in node.names]
            elif isinstance(node,ast.ImportFrom):
                names = [node.module or '']
            for name in names:
                if name.startswith('memento3d_head') and not name.startswith('memento3d_head.domain'):
                    errors.append(f'{file.relative_to(root)}:{node.lineno}: forbidden {name}')
                if name.split('.')[0] in {'numpy','PIL','mediapipe','torch','requests','httpx'}:
                    errors.append(f'{file.relative_to(root)}:{node.lineno}: SDK in {layer}')
if errors:
    raise SystemExit('\n'.join(errors))
print('Architecture boundaries: PASS')
