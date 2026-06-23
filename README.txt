=====================================================================
AgTech Drought Model -- Getting Started (for the team)
=====================================================================

WHAT THIS IS
  A model that forecasts the U.S. Drought Monitor (USDM) drought
  severity for Kerr County, TX, 4 weeks ahead, from weather +
  vegetation data. There are 6 severity levels:
      no_drought, D0, D1, D2, D3, D4   (D4 = most severe)

  You can run experiments by editing ONE config file and running ONE
  command -- no heavy coding needed.

---------------------------------------------------------------------
1. ONE-TIME SETUP
---------------------------------------------------------------------
(a) Install these once:
      - Git:       https://git-scm.com/downloads
      - Miniconda: https://docs.conda.io/en/latest/miniconda.html

(b) Download (clone) the repo. Open a terminal -- on Windows use
    "Git Bash", on Mac use "Terminal" -- and run:

      git clone https://github.com/akhan994/agtech_drought_model.git
      cd agtech_drought_model

(c) Create the Python environment (installs everything the model needs):

      conda env create -f environment.yml
      conda activate drought

    You only create it once. After that, each time you open a new
    terminal just run:  conda activate drought

---------------------------------------------------------------------
2. MAKE YOUR OWN BRANCH (so your experiments don't disturb others)
---------------------------------------------------------------------
    git checkout main
    git pull                        # get everyone's latest
    git checkout -b my-experiment   # make + switch to YOUR branch
                                    # (give it a meaningful name)

---------------------------------------------------------------------
3. RUN THE MODEL
---------------------------------------------------------------------
  From the top "agtech_drought_model" folder, with the env active:

      python -m scripts.run_pipeline @configs/training.txt

  This trains the model, evaluates it, and saves everything to a new
  timestamped folder:  results/training_<date-time>/  containing:
      - metrics.txt            (QWK, macro-F1, MAE, accuracy)
      - confusion_matrix.png   (predicted vs. true severity)
      - severity_timeseries.png
      - loss_curve.png
  Each run makes its OWN folder, so runs never overwrite each other.

---------------------------------------------------------------------
4. RUN YOUR OWN EXPERIMENT
---------------------------------------------------------------------
  Open  configs/training.txt  in any text editor, change a setting,
  then re-run the command in step 3. Worth trying:

      --input_window=8     how many past weeks the model sees (try 4/8/12)
      --neurons=64         model size
      --num_layers=2       model depth
      --dropout=0.3        regularization (helps avoid overfitting)
      --epochs=500         max training passes
      --feature_cols       which inputs to use (one per line)

  RULES for the config file:
      - one setting per line
      - for list settings (like --feature_cols), put each value on its
        own line below it
      - do NOT add comment lines (#...) -- every line is read as a setting

  Tip: read your run's results/.../metrics.txt -- QWK is our main score
  (higher = better; 1.0 is perfect).

---------------------------------------------------------------------
5. SAVE & SHARE YOUR WORK
---------------------------------------------------------------------
    git add configs/training.txt          # the file(s) you changed
    git commit -m "tried input_window=8"  # short note on what you did
    git push -u origin my-experiment      # upload YOUR branch to GitHub

  Then on GitHub, open a "Pull Request" from your branch to share your
  results. Your branch never touches `main` until it's reviewed.

---------------------------------------------------------------------
REMINDERS
---------------------------------------------------------------------
  - Always run from the top "agtech_drought_model" folder.
  - Always have the env active first:  conda activate drought
  - If a run errors, copy the full message and send it to Ayesha.
=====================================================================
