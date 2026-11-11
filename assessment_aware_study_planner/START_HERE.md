# Start here

1. Open README.md and follow its Python 3.12 environment setup.
2. Run `python -m unittest discover -s tests -v` (27 tests).
3. Run `python run_demo.py --trace --rejection-demo`.
4. Open `outputs/demo/cooperation.png` and `outputs/demo/result.json`.
5. Read `docs/discussion_guide.md` before your project discussion.
6. Use `docs/project_presentation.pptx` for the ten-minute talk, followed by
   `docs/live_demo.md` for the five-minute live demo.
7. Submit the nine-page `docs/project_report.pdf` with source and results.
8. Complete `docs/contributions_template.md` with truthful names and work.

The trained model, processed real data and measured experiments are already
included. No data download is needed to run the demo. Original raw OULAD files
are excluded to keep the package compact; `python data/make_data.py` recreates
them from a checked public archive.

The system is a course demonstration and observational what-if tool. Its
five-seed mean absolute prediction error is 21.67 mark points, versus 28.57 for
the specified average baseline. Predicted target attainment is not a guarantee.

The source_history.bundle preserves the actual initial git history. To continue
with that history in a separate folder, run:

```bash
git clone source_history.bundle ../study_planner_repo
```

Otherwise, edit the extracted source directly and create your own repository.
The supplied commits are explicitly labelled as AI-assisted work.
