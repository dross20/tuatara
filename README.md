<p align="center" style="margin-bottom: 0">
  <picture style="display: block; height: auto;">
    <source media="(prefers-color-scheme: dark)" srcset="https://i.imgur.com/001BRhf.png">
    <source media="(prefers-color-scheme: light)" srcset="https://i.imgur.com/X0Qq560.png">
    <img src="https://i.imgur.com/X0Qq560.png" width="200" style="height: auto;" alt="Tuatara logo"></img>
  </picture>
</p>

<div align="center">

  <a href="https://www.python.org/">![Static Badge](https://img.shields.io/badge/python-3.9+-green)</a>
  <a href="https://github.com/dross20/tuatara/blob/main/LICENSE">![GitHub license](https://img.shields.io/badge/license-MIT-brown.svg)</a>
  <a href="https://github.com/astral-sh/ruff">![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)</a>

</div>

---

>
> "Artificial intelligence is only as good as the data it learns from." - Unknown
>

## 🦎 What is Tuatara?

Tuatara is a library for generating fine-tuning pairs for large language model (LLM) post training.

## 🤔 Why Tuatara?

Fine-tuning large language models requires high-quality training data pairs that are well grounded in their source documents. Creating these pairs manually is laborious and error-prone, and existing tools often lack flexibility or fail to scale across different document types and domains. Tuatara addresses these challenges directly.

## 📦 Installation
Run the following command to install Tuatara:

```sh
pip install git+https://github.com/dross20/tuatara
```

## 🚀 Quickstart
The following example demonstrates how to use Tuatara's preconfigured pipeline for creating fine tuning pairs from multiple documents. By default, `default_pipeline` will use the OpenAI API for LLM inference and search for your OpenAI API key in the environment variables.

```python
from tuatara import default_pipeline

documents = [
  "./document1.pdf",
  "./document2.pdf",
  "./document3.txt"
]

pipeline = default_pipeline(model="gpt-4o")
pairs, history = pipeline(documents)
```

## 📜 License
This project is licensed under the [MIT license](https://github.com/dross20/tuatara/blob/2ab8b458f0d6d3109d7e5381c58961c9df992449/LICENSE).
