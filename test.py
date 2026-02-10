import adms
from pathlib import Path

adms.run_model("model_3", k=10)
adms.run_all_models(k=5)

# Custom input file
print(adms.run_model("model_1", k=10, path=Path("data/applicant_cvs.json")))
print(adms.run_all_models(k=5, path=Path("data/applicant_cvs.json")))