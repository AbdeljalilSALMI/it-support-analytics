# GitHub publishing checklist

## Portfolio screenshots

Capture the dashboard from the local Streamlit app after connecting to the
project database. For prediction, use example inputs from the form. Keep the
password prompt and any personal information outside screenshots.

| Capture | What to show |
|---|---|
| EDA | Category volume and delay. The README already links the generated figure. |
| Overview | Four KPIs and the monthly ticket trend. |
| Ticket Analytics | An active category or date filter and the updated breakdown. |
| Resolution Prediction | Inputs, predicted class, probability, and contributing factors. |
| SHAP explanation | The individual waterfall. The README already links the generated figure. |
| Model comparison | The Model Insights page with validation and held-out tables. |

Save dashboard captures under reports/figures and link them from the README
once captured.

## Publish from PowerShell

Create an empty repository on GitHub. Run the following from the project root.
This workspace already has a local Git repository. Use git init -b main only
when initializing a workspace that does not already have one.

```powershell
git branch --show-current
git status --short
git add .
git diff --cached --name-only
git commit -m "Build IT support analytics and resolution prediction project"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/it-support-analytics.git
git push -u origin main
```

Replace YOUR-USERNAME with your GitHub username. If origin is already
configured, use git remote -v to verify its destination before pushing.
