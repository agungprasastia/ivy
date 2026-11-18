## Summary
<!-- Concise description of what this PR changes and why -->

## Related Issue
<!-- Closes #... or Refs #... -->

## Type of Change
- [ ] Bug fix (non-breaking fix for an existing issue)
- [ ] New feature / tool / slash command
- [ ] Refactor / modularization / cleanup
- [ ] Documentation / CI / repo hygiene

## Verification
Run the full test suite before opening this PR:
```powershell
& ".\.venv\Scripts\pytest.exe" -p no:cacheprovider -v
```

- [ ] All existing and new `pytest` tests pass without skipped/relaxed assertions
- [ ] `ruff check ivy tests` passes cleanly

## Ivy Safety & Hygiene Checklist
- [ ] No new runtime dependencies added to `pyproject.toml` without justification
- [ ] Tool outputs remain bounded (`truncate_head_tail`) and secret-redacted (`redact_secrets`)
- [ ] File mutations use `FileCheckpointStore` and shell execution uses `SafetyApprovalPolicy`
- [ ] No API keys, `.env`, `.db`, or local artifacts committed
