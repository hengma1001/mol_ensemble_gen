# Support

## How to file issues and get help

This project uses [GitHub Issues](https://github.com/hengma1001/mol_ensemble_gen/issues)
to track bugs and feature requests. Please search existing issues before filing a
new one to avoid duplicates.

When reporting a problem, it helps a lot to include:

- The command you ran and the config YAML (redact any paths you'd rather not share)
- The full traceback, not just the last line
- Which environment you were in, and whether `esm` / `transformers` / a GPU were
  involved — many issues are specific to the Biohub `esm` fork
- For training issues, the `out_dir` layout and the step the run reached

For questions about using the package, open a Discussion or an Issue with the
`question` label.

## Support policy

This is a research project maintained on a best-effort basis alongside other work.
There is no service-level agreement: expect a reply within about a week. Bug
reports with a minimal reproduction get looked at first.

`@pytest.mark.gpu` tests and anything touching mdCATH need hardware and data that
may not be available to the maintainer for a given report, so reproductions that
run offline (the default `pytest -m "not integration"` suite) are far easier to act
on.
