# Security Policy

## Supported versions

This is a research project; only the `main` branch receives fixes.

## Reporting a vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, use GitHub's private vulnerability reporting on this repository
(Security → Report a vulnerability), or email the maintainer at
<hengma@uchicago.edu>.

Please include as much of the following as you can:

- Type of issue (e.g. path traversal, deserialization of untrusted data, code
  injection via a config or checkpoint file)
- Full path of the source file(s) involved
- The affected commit or tag
- Any configuration needed to reproduce
- Step-by-step reproduction instructions, and a proof of concept if you have one
- What an attacker could achieve with it

You can expect an initial response within a week. Since this project is maintained
on a best-effort basis alongside research work, please allow reasonable time for a
fix before public disclosure.

## Scope notes

Two things worth knowing about this project's threat model:

- **Checkpoints and caches are trusted input.** `torch.load` is used with
  `weights_only=False` to read training checkpoints and featurization caches,
  which means loading one is equivalent to executing its author's code. Only load
  checkpoints and `cache/` directories you produced or otherwise trust.
- **Model weights come from a third party.** ESMFold2 weights are downloaded from
  the Hugging Face Hub on first use; their integrity is whatever the Hub provides.
