# Atomic 3D + Functional Ontology + 50 AI Agents -- ULTIMATE DRUG DISCOVERY ENGINE

**THE ONE THAT WORKS EVERYWHERE**

Generates **low-energy 3D conformers**, computes **Gasteiger charges**,
**dipole**, **PMI**, **asphericity**, embeds **per-atom 3D coordinates +
charges** into a **full OWL ontology**, and launches **50+ atomic-level
LLM agents** via Ollama for deep SAR insights.

------------------------------------------------------------------------

## Media (new additions)

This repository now includes:\
1) a short demo video\
2) an audio description file for the 50 AI agents\
3) a high-resolution map/flowchart PNG illustrating the pipeline and
agent interactions.

**Files added**: - `media/video/The_AI_Drug_Discovery_Engine.mp4` -
`media/audio/50_AI_Agents_description.mp3` -
`media/audio/50_AI_Agents_description.txt` - `images/map_flowchart.png`

### Embeds

``` html
<video controls width="720">
  <source src="The_AI_Drug_Discovery_Engine.mp4" type="video/mp4">
</video>
```

``` html
<audio controls>
  <source src="50_AI_Agents_description.mp3" type="audio/mpeg">
</audio>
```

![Pipeline map and flowchart](images/map.png)

------------------------------------------------------------------------

## Uses

1.  Drug discovery & design\
2.  SAR analysis via 50+ agents\
3.  OWL ontology building\
4.  Visualization & reporting\
5.  Research & education

------------------------------------------------------------------------

## Features

-   ETKDGv3 → MMFF → UFF optimization\
-   50+ specialized agents\
-   Full OWL/RDF ontology\
-   Gasteiger charges, PMI, dipole, asphericity\
-   Works in Jupyter, terminal, Docker\
-   Auto-generates sample data

------------------------------------------------------------------------

## Quick Start

``` bash
pip install -r requirements.txt
python atomic_3d_ontology_50_agents.py --ollama --model llama3.2
```

------------------------------------------------------------------------

## Repository Structure

    atomic-3d-ontology-50-agents/
    ├── README.md
    ├── requirements.txt
    ├── data.csv
    ├── atomic_3d_ontology_50_agents.py
    ├── agents/
    │   └── __init__.py
    ├── ontology/
    │   └── builder.py
    ├── conformers/
    │   └── generator.py
    ├── utils/
    │   └── ollama_client.py
    ├── config/
    │   └── agents.json
    ├── media/
    │   ├── video/
    │   │   └── The_AI_Drug_Discovery_Engine.mp4
    │   └── audio/
    │       ├── 50_AI_Agents_description.mp3
    │       └── 50_AI_Agents_description.txt
    ├── images/
    │   └── map_flowchart.png
    └── .gitignore

------------------------------------------------------------------------

## Contact

Open an issue or contact the maintainer.
