# -*- coding: utf-8 -*-

"""
Skill discovery for QualCoder AI agents.

Skills are stored as SKILL.md files in three scopes:
- system: src/qualcoder/ai_skills
- user:   <confighome>/ai_skills
- project:<project>/ai_data/ai_skills

Conflicts are resolved by scope priority: project > user > system.
The conflict key is relative_path + name (case-insensitive).
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Dict, List, Optional, Tuple

import yaml


@dataclass
class SkillRecord:
    scope: str
    root_path: str
    relative_path: str
    name: str
    description: str
    skill_id: str
    skill_md_path: str
    content: str
    metadata: Dict[str, Any]


class AiSkillsCatalog:
    """Discover and resolve QualCoder skills across system/user/project scopes."""

    _scope_priority = {
        "system": 0,
        "user": 1,
        "project": 2,
    }

    def __init__(self, app):
        self.app = app
        self._system_root = os.path.join(os.path.dirname(__file__), "ai_skills")

    def list_skills(self) -> List[SkillRecord]:
        """Return resolved skills with scope override handling."""

        selected: Dict[str, SkillRecord] = {}
        for scope, root in self._skill_roots():
            for skill in self._discover_scope(scope, root):
                conflict_key = self._conflict_key(skill.relative_path, skill.name)
                prev = selected.get(conflict_key)
                if prev is None:
                    selected[conflict_key] = skill
                    continue
                prev_rank = self._scope_priority.get(prev.scope, -1)
                new_rank = self._scope_priority.get(skill.scope, -1)
                if new_rank > prev_rank:
                    selected[conflict_key] = skill

        skills = list(selected.values())
        skills.sort(key=lambda item: item.skill_id.casefold())
        return skills

    def get_skill(self, identifier: str) -> Optional[SkillRecord]:
        """Resolve one skill by skill_id, or by unique plain name fallback."""

        query = str(identifier if identifier is not None else "").strip()
        if query == "":
            return None
        query_cf = query.casefold()
        skills = self.list_skills()
        for skill in skills:
            if skill.skill_id.casefold() == query_cf:
                return skill

        by_name = [skill for skill in skills if skill.name.casefold() == query_cf]
        if len(by_name) == 1:
            return by_name[0]
        return None

    def _skill_roots(self) -> List[Tuple[str, str]]:
        roots: List[Tuple[str, str]] = []
        roots.append(("system", self._system_root))

        confighome = ""
        if hasattr(self.app, "confighome"):
            confighome = str(getattr(self.app, "confighome", "")).strip()
        if confighome != "":
            roots.append(("user", os.path.join(confighome, "ai_skills")))

        project_path = ""
        if hasattr(self.app, "project_path"):
            project_path = str(getattr(self.app, "project_path", "")).strip()
        if project_path != "":
            roots.append(("project", os.path.join(project_path, "ai_data", "ai_skills")))
        return roots

    def _discover_scope(self, scope: str, root: str) -> List[SkillRecord]:
        if root is None or str(root).strip() == "" or not os.path.isdir(root):
            return []

        result: List[SkillRecord] = []
        for dirpath, _dirnames, filenames in os.walk(root):
            _dirnames.sort(key=lambda item: item.casefold())
            filenames.sort(key=lambda item: item.casefold())
            if "SKILL.md" not in filenames:
                continue
            skill_path = os.path.join(dirpath, "SKILL.md")
            raw = self._read_text(skill_path)
            if raw is None:
                continue
            metadata, body = self._split_frontmatter(raw)

            rel_dir = os.path.relpath(dirpath, root)
            if rel_dir == ".":
                rel_dir = ""
            rel_path = rel_dir.replace("\\", "/").strip("/")

            name = str(metadata.get("name", "")).strip()
            if name == "":
                name = os.path.basename(dirpath).strip()
            if name == "":
                name = "Unnamed skill"

            description = str(metadata.get("description", "")).strip()
            if description == "":
                description = self._infer_description(body)

            skill_id = name if rel_path == "" else f"{rel_path}/{name}"

            result.append(
                SkillRecord(
                    scope=scope,
                    root_path=root,
                    relative_path=rel_path,
                    name=name,
                    description=description,
                    skill_id=skill_id,
                    skill_md_path=skill_path,
                    content=body.strip(),
                    metadata=metadata,
                )
            )
        return result

    def _conflict_key(self, relative_path: str, name: str) -> str:
        rel = str(relative_path if relative_path is not None else "").strip().strip("/")
        nm = str(name if name is not None else "").strip()
        key = nm if rel == "" else f"{rel}/{nm}"
        return key.casefold()

    def _read_text(self, path: str) -> Optional[str]:
        try:
            with open(path, "r", encoding="utf-8-sig") as handle:
                return handle.read()
        except OSError:
            return None

    def _split_frontmatter(self, raw: str) -> Tuple[Dict[str, Any], str]:
        text = str(raw if raw is not None else "")
        lines = text.splitlines()
        if len(lines) < 3 or lines[0].strip() != "---":
            return {}, text

        end_idx = -1
        for idx in range(1, len(lines)):
            if lines[idx].strip() == "---":
                end_idx = idx
                break
        if end_idx <= 1:
            return {}, text

        frontmatter = "\n".join(lines[1:end_idx]).strip()
        body = "\n".join(lines[end_idx + 1 :]).strip()

        metadata: Dict[str, Any] = {}
        if frontmatter != "":
            try:
                loaded = yaml.safe_load(frontmatter)
                if isinstance(loaded, dict):
                    metadata = loaded
            except Exception:
                metadata = {}
        return metadata, body

    def _infer_description(self, body: str) -> str:
        text = str(body if body is not None else "")
        lines = [line.strip() for line in text.splitlines()]
        paragraph: List[str] = []
        in_paragraph = False
        for line in lines:
            if line == "":
                if in_paragraph:
                    break
                continue
            paragraph.append(line)
            in_paragraph = True
        if len(paragraph) == 0:
            return ""
        desc = " ".join(paragraph).strip()
        if len(desc) > 240:
            desc = desc[:237].rstrip() + "..."
        return desc
