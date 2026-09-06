# AGH Analytica NLP Predictor

### **Automated financial news analysis system based on language models (LLM) using Dual Path Architecture.**


<div align="center">

<div align="center">

![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LLM Provider](https://img.shields.io/badge/LLM_Provider-OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)

![uv](https://img.shields.io/badge/uv-Package_Manager-DE5FE9?style=for-the-badge)
![Sphinx](https://img.shields.io/badge/Sphinx-Docs-000000?style=for-the-badge&logo=readthedocs&logoColor=white)

</div>

</div>

## Project developed within the [AGH Analytica Student Research Group](https://github.com/AGHAnalytica) at [AGH University of Krakow](https://www.agh.edu.pl/) (2026).

## 🛠️ Setup & Installation

The NLP Predictor pipeline uses `pyproject.toml` for dependency management and the ultra-fast `uv` package manager.

### Prerequisites
- **Python** (v3.14+)
- **uv** - follow instructions at [official Astral page](https://docs.astral.sh/uv/getting-started/)
- API key for [OpenAI API](https://platform.openai.com/api-keys)

### Installation Steps

1. **Clone the repository:**
```bash
   cd <your/working/directory>
   git clone https://github.com/AGHAnalytica/NLP_predictor.git
   cd NLP_predictor
```
2. Create `.env` file
In repo's directory create a file named `.env` and fill it with the following content
```
OPENAI_API_KEY=your_api_key_goes_here
```

3. Install all needed dependencies and create virtual environment using `uv`

```bash
   uv sync
```

4. **Run the main pipeline:**

```bash
   uv run main
```

---

## 📚 Documentation Generation

Technical documentation for the NLP engine is generated automatically directly from the Python docstrings using **Sphinx**.

Here is the step-by-step guide on how to build the latest HTML documentation:

### Step 1: Activate the environment

Ensure your virtual environment is active (see steps 3 and 4 in the Setup section). All required documentation packages (like Sphinx) are already installed via `uv sync`.

### Step 2: Navigate to the docs directory

Move to the folder containing the Sphinx `Makefile`:

```bash
cd docs

```

### Step 3: Update source files (if code structure changed)

If you have added new `.py` files or changed the directory structure in `src/nlp_predictor`, rebuild the `.rst` source files:

```bash
sphinx-apidoc -f -o source ../src/nlp_predictor
```

### Step 4: Build the HTML documentation

Generate the web pages using the `make` command:

* **Linux / macOS:**

```bash
  make html
```

* **Windows:**

```cmd
  make.bat html
```

### Step 5: View the documentation

Once the build succeeds, the generated files will be located in:
`docs/build/html/`

Open the **`index.html`** file from that directory in any web browser to view the latest documentation.

```
    file:///$YOUR_PATH$/NLP_predictor/docs/build/html/index.html
```

## 🏗️ Architecture (TODO)

[TODO: Describe the data flow: Scraping -> Topic Extraction -> Upper Path / Lower Path -> Signal Fusion]

## 🧮 DCIM Model & Scoring (TODO)
[TODO]

## 🧑‍💻 Contributors
| Name | GitHub | Role |
| :--- | :--- | :--- |
| **Norbert Jarzmik** | [![GitHub](https://img.shields.io/badge/-njarzmik-181717?style=flat&logo=github)](https://github.com/njarzmik) [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://pl.linkedin.com/in/norbert-jarzmik-38802431b) | Team Leader / Researcher / Python Developer|
| **Konrad Zatorski** | [![GitHub](https://img.shields.io/badge/-K0D1Z-181717?style=flat&logo=github)](https://github.com/K0D1Z) [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/konrad-zatorski/) | Python Developer / Researcher / Upper & Lower Path Developer|
| **Rafał Mazur** | [![GitHub](https://img.shields.io/badge/-Rafek731-181717?style=flat&logo=github)](https://github.com/Rafek731) [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)]()| Python Developer / Researcher|
| **Filip Wojtczak** | [![GitHub](https://img.shields.io/badge/-imnotbruhwe-181717?style=flat&logo=github)](https://github.com/imnotbruhwe) [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/filip-wojtczak/)| Python Developer / Researcher / Scrapper Developer|
| **Wojciech Kocemba** | [![GitHub](https://img.shields.io/badge/-Operative88-181717?style=flat&logo=github)](https://github.com/Operative88) [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)]()| Python Developer / Researcher / Scrapper Developer|
| **Filip Kontek** | [![GitHub](https://img.shields.io/badge/-napewnoniefixu-181717?style=flat&logo=github)](https://github.com/napewnoniefixu) [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)]()| Python Developer / Researcher / Upper Path Developer|
| **Olga Ehrenfeld** | [![GitHub](https://img.shields.io/badge/-TODO-181717?style=flat&logo=github)]() [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)]()| Python Developer / Researcher|
| **Bartek Tasak** | [![GitHub](https://img.shields.io/badge/-TODO-181717?style=flat&logo=github)]() [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)]()| Python Developer / Researcher|
| **Andrii-Dmytro Kilnytskyi** | [![GitHub](https://img.shields.io/badge/-TODO-181717?style=flat&logo=github)]() [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)]()| Python Developer / Researcher|
