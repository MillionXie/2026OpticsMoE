"""Git-only isolated server sync; password is prompted, never persisted.

Requires paramiko on the local management host, not on the training server.
No GPU job is started. SSH host keys must already be trusted in known_hosts.
"""
from __future__ import annotations

import argparse
import getpass
import shlex


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("inspect", "sync", "test", "publish-bundle"), required=True)
    parser.add_argument("--bundle", help="Git bundle transport only; never raw source overwrite")
    parser.add_argument("--base", help="GitHub prerequisite SHA for a small Git bundle")
    parser.add_argument("--host", default="202.120.62.181")
    parser.add_argument("--port", type=int, default=24096)
    parser.add_argument("--user", default="guest3")
    parser.add_argument("--repo", default="/DATA/DATA1/guest3/2026OpticsMoE")
    parser.add_argument("--worktree", default="/DATA/DATA1/guest3/2026OpticsMoE/.worktrees/t13_temporal_robust_20260927")
    parser.add_argument("--branch", default="codex/t13-temporal-robust-20260927")
    parser.add_argument("--commit")
    parser.add_argument("--python", default="/home/guest3/miniconda3/envs/xml/bin/python")
    args = parser.parse_args()
    if not args.repo.startswith("/") or not args.worktree.startswith(args.repo.rstrip("/") + "/.worktrees/"):
        parser.error("Require an absolute repository path and an isolated child .worktrees target")
    import paramiko
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.connect(args.host, port=args.port, username=args.user,
                   password=getpass.getpass("SSH password: "), allow_agent=False, look_for_keys=False, timeout=20)

    def execute(command):
        stdin, stdout, stderr = client.exec_command(command, timeout=120)
        out, err = stdout.read().decode(), stderr.read().decode()
        print(out, end="")
        if err:
            print(err, end="")
        status = stdout.channel.recv_exit_status()
        if status:
            raise RuntimeError(f"Remote command failed (exit {status})")
        return out.strip()

    repo, worktree, branch = map(shlex.quote, (args.repo, args.worktree, args.branch))
    # The laboratory network blocks GitHub SSH port 22; retain strict host checking.
    ssh_command = "ssh -oBatchMode=yes -oConnectTimeout=10 -oHostName=ssh.github.com -oHostKeyAlias=github.com -oStrictHostKeyChecking=yes -p443"
    network_git = f"git -C {repo} -c core.sshCommand={shlex.quote(ssh_command)}"
    try:
        if args.phase == "inspect":
            execute(f"hostname; git -C {repo} rev-parse HEAD; git -C {repo} status --short --untracked-files=no")
            execute("command -v python3; ls /opt/conda/bin/python /opt/conda/envs/*/bin/python /home/guest3/anaconda3/bin/python /home/guest3/anaconda3/envs/*/bin/python /home/guest3/miniconda3/envs/*/bin/python /home/guest3/.conda/envs/*/bin/python /DATA/DATA1/guest3/miniconda3/envs/*/bin/python 2>/dev/null || true")
            execute(f"find {repo}/cache {repo}/LightGenV2/tasks/t06_video_quality_assessment/runs -maxdepth 4 -type f \\( -name '*49x1024*' -o -name '*temporal_prompt*' \\) -print 2>/dev/null || true")
            return
        if not args.commit or len(args.commit) != 40 or any(c not in "0123456789abcdef" for c in args.commit):
            parser.error("sync/test requires an exact 40-character commit SHA")
        commit = shlex.quote(args.commit)
        if args.phase == "publish-bundle":
            from pathlib import Path
            if not args.bundle or not args.base or len(args.base) != 40 or any(c not in "0123456789abcdef" for c in args.base):
                parser.error("publish-bundle requires --bundle and exact --base SHA")
            bundle = Path(args.bundle)
            if not bundle.is_file():
                parser.error("Git bundle is missing")
            stage = args.repo.rstrip("/") + "/.codex_tmp/t13_git_transport_20260927"
            remote_bundle = stage + "/" + args.commit + ".bundle"
            execute(f"mkdir -p {shlex.quote(stage)}; test ! -e {shlex.quote(remote_bundle)}")
            sftp = client.open_sftp()
            try:
                sftp.put(str(bundle), remote_bundle)
            finally:
                sftp.close()
            # Bundle transport carries only Git objects; shared working files stay untouched.
            execute(f"git -C {repo} cat-file -e {shlex.quote(args.base)}^{{commit}} || {network_git} fetch origin {shlex.quote(args.base)}")
            execute(f"git -C {repo} bundle verify {shlex.quote(remote_bundle)}")
            execute(f"git -C {repo} fetch {shlex.quote(remote_bundle)} HEAD")
            actual = execute(f"git -C {repo} rev-parse FETCH_HEAD")
            if actual != args.commit:
                raise RuntimeError("Transported Git commit mismatch")
            execute(f"{network_git} push origin {shlex.quote(args.commit + ':refs/heads/' + args.branch)}")
            return
        if args.phase == "sync":
            execute(f"{network_git} fetch origin {branch}")
            remote = execute(f"git -C {repo} rev-parse FETCH_HEAD")
            if remote != args.commit:
                raise RuntimeError("GitHub branch tip differs from requested commit")
            # Never pull/checkout the shared dirty server tree.
            execute(f"if test -d {worktree}; then test -z \"$(git -C {worktree} status --porcelain)\" && git -C {worktree} checkout --detach {commit}; else git -C {repo} worktree add --detach {worktree} {commit}; fi")
            execute(f"git -C {worktree} rev-parse HEAD")
        else:
            actual = execute(f"git -C {worktree} rev-parse HEAD")
            if actual != args.commit:
                raise RuntimeError("Server checkout is not the requested commit")
            python = shlex.quote(args.python)
            task = f"{worktree}/LightGenV2/tasks/t13_temporal_robust_training"
            execute(f"cd {worktree} && {python} {task}/verify_source.py && {python} -m pytest {task}/tests -q")
    finally:
        client.close()


if __name__ == "__main__":
    main()
