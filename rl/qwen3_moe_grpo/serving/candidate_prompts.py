"""Load verbatim business prompt snapshots as test-model input, not runner instructions."""
import argparse
import hashlib
import json
from pathlib import Path

INDUSTRIES = ('homecare', 'general-worker')
COMMON = ('IDENTITY.md', 'AGENTS.md', 'SOUL.md', 'MEMORY.md', 'TOOLS.md')
SKILLS = ('job-service', 'group-delivery')
FILES = ([f'common/{name}' for name in COMMON]
         + [f'common/skills/{name}/SKILL.md' for name in SKILLS]
         + [f'industries/{industry}/{name}' for industry in INDUSTRIES
            for name in ('IDENTITY.md', 'AGENTS.md', 'SOUL.md')]
         + [f'industries/{industry}/skills/{name}.md' for industry in INDUSTRIES for name in SKILLS])


def digest(data):
    return hashlib.sha256(data).hexdigest()


def snapshot(source, destination):
    # Explicit allowlist: no recursive collection of business data or credentials.
    source, destination = Path(source), Path(destination)
    payload = {name: (source / name).read_bytes() for name in FILES}
    destination.mkdir(parents=True, exist_ok=False)
    rows = []
    for name, data in payload.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        rows.append(dict(path=name, bytes=len(data), sha256=digest(data)))
    manifest = dict(version=1, source_kind='local-business-prompt-snapshot', files=rows)
    (destination / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    return manifest


class PromptBundle:
    def __init__(self, root):
        self.root = Path(root)
        raw = (self.root / 'manifest.json').read_bytes()
        manifest = json.loads(raw)
        rows = manifest['files']
        if len(rows) != len(FILES) or {r['path'] for r in rows} != set(FILES):
            raise ValueError('Prompt manifest must contain exactly the approved prompt files')
        self.text = {}
        for row in rows:
            data = (self.root / row['path']).read_bytes()
            if digest(data) != row['sha256'] or len(data) != row['bytes']:
                raise ValueError('Prompt snapshot changed: ' + row['path'])
            self.text[row['path']] = data.decode('utf-8')
        self.sha256 = digest(raw)

    def skill(self, name, industry):
        if name not in SKILLS or industry not in INDUSTRIES:
            raise ValueError('Unknown skill or industry')
        return (self.text[f'common/skills/{name}/SKILL.md'] + '\n\n'
                + self.text[f'industries/{industry}/skills/{name}.md'])

    def system(self, context):
        industry = context['industryCode']
        if industry not in INDUSTRIES:
            raise ValueError('Unknown industry')
        # Same declared order as conversation-demo application.yml components.
        sections = []
        for name in COMMON:
            sections.append(self.text[f'common/{name}'])
            if name in ('IDENTITY.md', 'AGENTS.md', 'SOUL.md'):
                sections.append(self.text[f'industries/{industry}/{name}'])
        sections.append('可用业务策略：job-service（岗位服务）、group-delivery（拉群交付）。'
                        '内容未加载时通过 read_skill 读取；本次是普通私聊，不是促活事件。')
        sections.append('当前会话上下文（应用每次请求刷新）：\n' + json.dumps(context, ensure_ascii=False))
        return '\n\n'.join(sections)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True, help='aihire-candidate-agent workspace root')
    parser.add_argument('--output', type=Path, required=True, help='Use an ignored runs/ directory')
    args = parser.parse_args()
    manifest = snapshot(args.source, args.output)
    print(json.dumps(dict(files=len(manifest['files']), bytes=sum(r['bytes'] for r in manifest['files']),
                          manifest_sha256=PromptBundle(args.output).sha256)))


if __name__ == '__main__':
    main()
