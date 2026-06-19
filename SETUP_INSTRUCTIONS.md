# Setup Instructions — VS Code + GitHub

## Part 1: Get the project onto your machine

1. Download the project folder `bmw-volatility-risk` (shared at the end of this chat) and unzip it somewhere you can find it, e.g. `Desktop/bmw-volatility-risk`.

## Part 2: Open in VS Code

1. Open VS Code.
2. `File → Open Folder` → select `bmw-volatility-risk`.
3. Open a terminal inside VS Code: `Terminal → New Terminal` (or Ctrl+` / Cmd+`).

## Part 3: Set up a Python virtual environment

In the VS Code terminal:

```bash
python -m venv venv
```

Activate it:

- **Windows (PowerShell):** `venv\Scripts\Activate.ps1`
  (If you get a script-disabled error, run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` then try again)
- **Windows (cmd):** `venv\Scripts\activate.bat`
- **macOS/Linux:** `source venv/bin/activate`

You should now see `(venv)` at the start of your terminal prompt. In VS Code, also select this interpreter: `Ctrl+Shift+P` → "Python: Select Interpreter" → choose the one inside `venv`.

## Part 4: Install dependencies

```bash
pip install -r requirements.txt
```

This installs pandas, numpy, arch (GARCH models), statsmodels, scipy, matplotlib, seaborn, streamlit, and plotly.

## Part 5: Run the pipeline

```bash
python main.py
```

This will:
- Load and validate `data/bmw.csv`
- Run EDA and save plots to `outputs/figures/`
- Fit GARCH(1,1), GJR-GARCH, and EGARCH and compare them
- Forecast volatility 30 days ahead
- Compute VaR / Expected Shortfall
- Backtest the model and save everything to `outputs/results/`

You'll see progress printed step by step in the terminal, ending with a summary of results.

## Part 6: Launch the dashboard

```bash
streamlit run dashboard/app.py
```

This opens an interactive dashboard in your browser (usually `http://localhost:8501`) where you can switch between models, adjust the VaR confidence level, and change the forecast horizon live.

To stop it, go back to the terminal and press `Ctrl+C`.

## Part 7: Push to GitHub

### 7a. Create the repo on GitHub
1. Go to [github.com](https://github.com) → click **+** → **New repository**.
2. Name it `bmw-volatility-risk-analysis` (or your preference).
3. Leave it empty (no README/.gitignore/license — you already have these locally).
4. Click **Create repository**. Keep the page open — you'll need the URL it shows you.

### 7b. Push from VS Code terminal

Make sure you're in the project folder, then:

```bash
git init
git add .
git commit -m "Initial commit: BMW volatility & risk modeling project"
git branch -M main
git remote add origin https://github.com/shiv-speccc/bmw-volatility-risk-analysis.git
git push -u origin main
```

(Replace the URL with whatever GitHub gave you if you named the repo differently.)

If this is your first time pushing from this machine, Git may prompt you to sign in via browser — follow the prompt and authorize.

### 7c. Verify

Refresh your GitHub repo page — you should see all your files, and the `outputs/figures/*.png` images should render directly in the GitHub file browser.

## Troubleshooting

- **`git: command not found`** → install Git from [git-scm.com](https://git-scm.com/downloads), restart VS Code.
- **`pip: command not found`** → make sure your virtual environment is activated (you should see `(venv)` in the prompt).
- **Push rejected / authentication failed** → GitHub no longer accepts password auth for git push. Use a [Personal Access Token](https://github.com/settings/tokens) as your password when prompted, or set up the GitHub CLI (`gh auth login`) for easier authentication.
- **`ModuleNotFoundError`** → run `pip install -r requirements.txt` again, and confirm the VS Code interpreter is set to your `venv`.
