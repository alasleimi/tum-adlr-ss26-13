# Research archive and recovery

The clean `main` branch is a curated submission view, not a deletion of the
research record.

- Branch: `research/exploratory-archive-20260802`
- Archive branch point: commit `835ddaa9d4b7b9b09ba394425508b137714e4981`
- Pre-operation Git bundle: `Project 15 PRE-OP 20260802-134528.bundle`
- Full filesystem snapshot: `Project 15 BACKUP 20260802-134528`

The bundle and filesystem snapshot are intentionally stored outside the Git
worktree. The snapshot was compared to the source by per-file size and SHA-256
for 40,926 high-value files (24,212,890,626 bytes) with zero mismatches. The
Git bundle passed `git bundle verify`, and the source repository passed strict
`git fsck` before restructuring.

Use the research branch for exploratory scripts, intermediate reports, old
poster versions, broad experiment sweeps, logs, and data-mining. Use `main` for
grading and for reproducing only claims made in the final report.

The archive captures the committed history. The external filesystem snapshot
also preserves untracked, ignored, and unstaged research artifacts that cannot
be represented by a branch alone.
