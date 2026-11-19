# Dataset selection

Use the existing local tickets.csv to develop and evaluate the project as a
synthetic support analytics case study. The file has 100,000 rows, 38 columns,
and unique ticket IDs from 1 to 100,000. No exact duplicate rows were observed.

The schema matches the public humans_only configuration of Talgat Kozahmetov's
[Support Tickets with an AI-Automation Counterfactual](https://huggingface.co/datasets/s2pidape/support-ticket-dataset).
Several non-personal fields for ticket 100 also match the public preview.
This supports the attribution. The exact export history remains unverified.
The publisher describes the public dataset as synthetic and lists CC BY 4.0.
The local CSV stays outside Git.

## Suitability and limitations

- Creation timestamps span January 2022 to December 2024.
- There are 11 categories, 4 priorities, 5 channels, and 6 products. The file
  also contains issue types, descriptions, previous ticket counts, and
  satisfaction scores. There is no explicit requester department field.
  The support team is useful for historical analysis.
- Both resolution duration fields are populated for 89,932 tickets, all with
  Closed or Resolved status. The remaining 10,068 lack these outcomes and
  cannot receive a duration-based label from the recorded durations alone.
- The resolution_time_hours field has 111 negative values. Timestamp checks
  found 124 resolutions before creation and 284 closures before resolution.
  These counts may overlap.
- At 24 hours, the wall-clock field gives 41,614 fast and 48,318 slow tickets.
  The other duration field gives 76,911 fast and 12,910 slow after excluding
  negative values for this comparison. The final target requires validation
  of duration semantics and inconsistent timestamps.
- Candidate opening-time inputs include creation time, issue category, product,
  and channel. Priority and previous ticket count require confirmation that
  the recorded values represent what was known at opening. Resolution fields,
  escalation, and satisfaction are collected after opening and would leak
  information into the model.
- Model scores will describe performance on simulated tickets. Generalization
  to an operational support desk remains untested.

## Reproduce the aggregate audit

Run from the project root with the source CSV in the parent workspace.

```powershell
.\.venv\Scripts\python.exe src\dataset_audit.py
```
