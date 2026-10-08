# PID velocity control

A simple vehicle simulation for the Formula Slug Autonomous onboarding. I added a few tools to make the controller easier to tune and its response easier to inspect:

- Air drag calculated from velocity and plotted in newtons alongside velocity and error. It is a separate measurement; the vehicle model still uses the template's constant friction.
- Plot markers for the 10% and 90% rise thresholds, peak overshoot, and settling within 1% of the target. Rise time, settling time, and overshoot are displayed below the graph.
- PID gain controls in the plot window. Change `K_P`, `K_I`, or `K_D` and press Enter to refresh the simulation without restarting the program.

As well as, of course, implementing the PID explained in [./project_description.md](https://github.com/GalexY727/fs-5-auto-onboarding-project/blob/main/auto/project_description.md)

Check out my [drawing board](https://www.tldraw.com/f/2jMMKuLLauSb1jM2Kv22k?d=v-17.488.2410.1427.page) (poster) for more information!

Run `python run_template.py` with NumPy and Matplotlib installed. (please make a venv for our sake)