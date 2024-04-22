# The Minimal Example of Detecting Addresses from Clinical Notes Using LLM

Author: *Huayu Zhang*

## Directory Structure and description

```text
├── README.md
├── example_input.csv
├── ner_config.py
├── output
│   ├── 0_entity.json
│   ├── 0_marked.md
│   ├── 0_output.json
│   ├── 1_entity.json
│   ├── 1_marked.md
│   ├── 1_output.json
│   ├── 2_entity.json
│   ├── 2_marked.md
│   └── 2_output.json
├── requirements.txt 
└── run.py
```

- `example_input.csv` Input csv file with row index and `note`
- `ner_config.py` Functions and configurations
- `requirements.txt` Packages to be used for `venv`: `pip install -f requirements.txt`
- `run.py` To run all notes in the `example_input.csv`
- `output` Outputs generated

## How to use

- Build the python virtual environment using `requirements.txt`
- In `run.py` change the `PATH` variable to your folder containing the files
- Run by `python3 run.py`

## Example output

There are three outputs generated:

### Output

Output and parsed output from different sections

```json
[
    {
        "section_name": "full_text",
        "section_start": 0,
        "section_end": 1326,
        "section_text": "Discharge Summary #1\nPatient Name: Jan...<The full input>",
        "generated_text": " Here is the output in .yaml format:\n```yaml\n     - \"The University of Edinburgh, 5-7, 3 Little France Rd, Edinburgh EH16 4UX\"\n     - \"5 Little France Dr, Edinburgh EH16 4UU\"\n```</s>",
        "generated_parsed": [
            "The University of Edinburgh, 5-7, 3 Little France Rd, Edinburgh EH16 4UX",
            "5 Little France Dr, Edinburgh EH16 4UU"
        ]
    }
]
```

### Entities

List of entities and the corresponding lcoations

```json
[
    {
        "type": "entity",
        "section_name": "full_text",
        "string": "The University of Edinburgh, 5-7, 3 Little France Rd, Edinburgh EH16 4UX",
        "description": "AddressInfo",
        "start": 117,
        "end": 189
    },
    {
        "type": "entity",
        "section_name": "full_text",
        "string": "5 Little France Dr, Edinburgh EH16 4UU",
        "description": "AddressInfo",
        "start": 1266,
        "end": 1304
    }
]
```

### Marked text

*Below is the marked text for visualization:*

```markdown
Discharge Summary #1
Patient Name: Jane Doe
Date of Birth: April 12, 1980
Medical Record Number: 123456789

Address: <mark>The University of Edinburgh, 5-7, 3 Little France Rd, Edinburgh EH16 4UX<sup>AddressInfo</sup></mark>

Admission Date: April 1, 2024
Discharge Date: April 7, 2024

Admitting Diagnosis: Community-acquired Pneumonia
Principal Discharge Diagnosis: Resolved pneumonia

History of Present Illness:
Ms. Doe presented to the emergency department with a 3-day history of fever, productive cough, and pleuritic chest pain. Chest X-ray confirmed the diagnosis of pneumonia, and she was admitted for IV antibiotics and supportive care.

Hospital Course:
During her stay, Ms. Doe responded well to treatment with IV levofloxacin. Her fever subsided, and respiratory symptoms improved. She tolerated oral intake and ambulation independently.

Discharge Medications:

Levofloxacin 500 mg orally once daily for 7 days
Acetaminophen 500 mg every 6 hours as needed for pain or fever
Discharge Instructions:
Ms. Doe was discharged in stable condition. She was advised to complete her course of antibiotics and follow up with her primary care physician for repeat chest X-ray in 4 weeks to ensure resolution of pneumonia.

Follow-Up Plan:
Primary Care Follow-Up:
Dr. John Smith, MD
Anytown Clinic
<mark>5 Little France Dr, Edinburgh EH16 4UU<sup>AddressInfo</sup></mark>
Phone: (555) 123-4567
```
