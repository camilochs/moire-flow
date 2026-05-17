"""LammpsExecutor: thin Docker subprocess wrapper for the LAMMPS runtime.

Support B. Does NOT execute LAMMPS itself — it shells out to:

    docker run --platform linux/amd64 --rm \\
        -v {host_work_dir}:/work -w /work \\
        {image} lmp -in {script.name} -log {log.name}

The image is built separately (see `runtime/` directory). linux/amd64 only
— ANALYSIS.md §10 documents that arm64 cannot ship QUIP+MACE.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

DEFAULT_IMAGE = "ghcr.io/camilochs/moire-flow-runtime:latest"


@dataclass(frozen=True)
class LammpsRun:
    """Outcome of a single LAMMPS invocation."""

    returncode: int
    script: Path
    log: Path
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class LammpsExecutor:
    """Run LAMMPS inside the moire-flow Docker runtime."""

    def __init__(
        self,
        image: str = DEFAULT_IMAGE,
        docker_bin: str | None = None,
        platform: str = "linux/amd64",
        extra_run_args: list[str] | None = None,
    ):
        self.image = image
        self.docker_bin = docker_bin or shutil.which("docker") or "docker"
        self.platform = platform
        self.extra_run_args = list(extra_run_args or [])

    def capabilities(self, timeout: float = 30.0) -> dict[str, str | bool]:
        """Probe the runtime: does docker exist, is the image pullable, what LAMMPS version?"""
        if not shutil.which(self.docker_bin):
            return {"docker_available": False, "image": self.image, "lammps_version": ""}
        try:
            res = subprocess.run(
                [self.docker_bin, "run", "--platform", self.platform, "--rm",
                 self.image, "lmp", "-help"],
                capture_output=True, text=True, timeout=timeout,
            )
        except (subprocess.SubprocessError, FileNotFoundError) as exc:
            return {"docker_available": True, "image": self.image,
                    "lammps_version": "", "probe_error": str(exc)}
        first_line = (res.stdout or "").splitlines()[:1]
        version_line = next((line for line in (res.stdout or "").splitlines()
                             if line.startswith("LAMMPS")), "")
        return {
            "docker_available": True,
            "image": self.image,
            "lammps_version": version_line or (first_line[0] if first_line else ""),
            "probe_returncode": res.returncode,
        }

    def run(
        self,
        script_path: str | Path,
        log_path: str | Path | None = None,
        work_dir: str | Path | None = None,
        timeout: float | None = None,
    ) -> LammpsRun:
        script_path = Path(script_path).resolve()
        if not script_path.exists():
            raise FileNotFoundError(script_path)
        work_dir = Path(work_dir or script_path.parent).resolve()
        log_path = Path(log_path or work_dir / f"{script_path.stem}.log").resolve()
        # Force the lmp binary regardless of the image's ENTRYPOINT (the
        # upstream lammps/lammps image uses ENTRYPOINT ["lmp"] which would
        # otherwise concatenate "lmp lmp -in …" and fail).
        cmd = [
            self.docker_bin, "run", "--platform", self.platform, "--rm",
            *self.extra_run_args,
            "-v", f"{work_dir}:/work", "-w", "/work",
            "--entrypoint", "lmp",
            self.image,
            "-in", script_path.name, "-log", log_path.name,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return LammpsRun(
            returncode=res.returncode,
            script=script_path,
            log=log_path,
            stdout=res.stdout or "",
            stderr=res.stderr or "",
        )


__all__ = ["LammpsExecutor", "LammpsRun", "DEFAULT_IMAGE"]
