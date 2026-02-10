# HI-Algosoc-ADM-Framework

## Running

Run all models:
```powershell
python main.py --k 10
```

Run a single model:
```powershell
python main.py --model model_3 --k 10
```

This outputs a JSON object with the top-k rankings.

## Notes

- Data source: `data/applicant_cvs.json`
- You can change `k` to any positive integer.
- Programmatic use:
```python
import adms

adms.run_model("model_3", k=10)
adms.run_all_models(k=5)
```
