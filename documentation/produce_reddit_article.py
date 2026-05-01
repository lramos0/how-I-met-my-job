#!/usr/bin/env python3
"""
Analyze job listings with OpenAI, choose topics, and generate a Reddit-ready markdown article.

Setup:
  pip install openai pandas tabulate pyperclip

Usage:
  export OPENAI_API_KEY="your_key"
  python jobs_to_reddit.py --company Disney --csv disney_jobs.csv

Copy output:
  cat out.txt | pbcopy      # macOS
  cat out.txt | xclip -sel clip   # Linux
  powershell Get-Content out.txt | Set-Clipboard   # Windows
"""

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
from openai import OpenAI


def read_jobs(csv_path: str, max_rows: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if max_rows:
        df = df.head(max_rows)
    return df.fillna("")


def dataframe_summary(df: pd.DataFrame) -> str:
    preview = df.to_markdown(index=False)
    stats = {
        "row_count": len(df),
        "columns": list(df.columns),
    }
    return f"DATASET_STATS:\n{json.dumps(stats, indent=2)}\n\nDATASET:\n{preview}"


def ask_openai(client: OpenAI, model: str, prompt: str) -> str:
    response = client.responses.create(
        model=model,
        input=prompt,
    )
    return response.output_text


def copy_instructions() -> str:
    return (
        "\n---\n\n"
        "## How to copy this to Reddit\n\n"
        "macOS:\n"
        "cat out.txt | pbcopy\n\n"
        "Linux:\n"
        "cat out.txt | xclip -selection clipboard\n\n"
        "Windows PowerShell:\n"
        "Get-Content out.txt | Set-Clipboard\n\n"
        "Then paste into Reddit’s Markdown editor.\n"
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--company", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--model", default="gpt-4.1")
    parser.add_argument("--out", default="out.txt")
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()
    today = dt.date.today().isoformat()
    client = OpenAI()

    df = read_jobs(args.csv, args.max_rows)
    dataset = dataframe_summary(df)

    topic_prompt = f"""
        Here are some job listings from {args.company}. Given that today is {today}, and this is a complete set of their available positions, what can you tell me about their listings? Only return a numbered list of interesting article topics.Make the topics specific, data-driven, and suitable for Reddit discussion.
    {dataset}
    """
    print("\nQuerying OpenAI for possible topics...\n")
    topics = ask_openai(client, args.model, topic_prompt)
    print(topics)

    selected = input(
        "\nEnter the topic number(s) or paste your selected topic text: "
    ).strip()

    article_prompt = f"""
        Write me an article about this selected topic: {selected}

Use the dataset I already provided below.

Requirements:

Reddit-ready Markdown
Include a strong title
Include clear sections
Include the data itself in Markdown tables
Include complex visuals using text/Markdown where possible:
ASCII charts
distribution tables
grouped counts
comparisons
Be transparent that the dataset is a snapshot as of {today}
Do not invent data not present in the dataset
Make it interesting and analytical

Company: {args.company}
Date: {today}

{dataset}
    """
    print("\nGenerating article...\n")
    article = ask_openai(client, args.model, article_prompt)

    final_text = article.rstrip() + copy_instructions()
    Path(args.out).write_text(final_text, encoding="utf-8")

    print(f"\nWrote Reddit-ready Markdown to {args.out}")
