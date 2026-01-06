import os
import re
from dotenv import load_dotenv
from llama_index.core import Document, SummaryIndex
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
)

from db_utils import get_vector_index

def main():
    load_dotenv()
    
    # Initialize Vector Store via utility
    vector_index, chroma_collection = get_vector_index()
    
    print(chroma_collection.count())

    # We need a SummaryIndex for the list query as well, but simply creating it empty 
    # and not persisting it might differ from original intent if it was meant to index docs.
    summary_index = SummaryIndex([])

    # LOAD RESUMES (1 FILE = 1 CANDIDATE)
    documents = []
    data_dir = "data"

    # check if exists?
    # For this practical task, we'll keep the behavior of re-inserting or we'll assume the user runs it once.
    # However, to be safe and match original behavior, we re-run insertion.
    
    # To avoid duplicates in a real app we'd check existence.
    # Let's assume for this task we just run it. 
    
    for idx, filename in enumerate(sorted(os.listdir(data_dir))):
        if not filename.endswith(".txt"):
            continue

        path = os.path.join(data_dir, filename)

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        name_match = re.search(r"name\s*[-:]\s*(.+)", text, re.IGNORECASE)
        name = name_match.group(1).strip() if name_match else "unknown"
        print(f"{name} idx - {idx}")

        doc = Document(
            text=text,
            metadata={
                "candidate_id": idx,
                "candidate_name": name,
                "file_name": filename,
                "type": "resume",
            },
        )

        # In a persistent setup, we might want to skip insertion if already present, 
        # but the original code had no check. We will keep it as is.
        vector_index.insert(doc)
        summary_index.insert(doc)

    # LIST ALL CANDIDATES
    list_prompt = """
    You are given a collection of resumes.
    Each document represents ONE DISTINCT candidate.

    Task:
    Generate a list of ALL candidates in the index.
    Do NOT merge information from different candidates.

    For each candidate, output:
    - Full Name (from resume, or "unknown")
    - Current Job Title (or "unknown")
    - Years of commercial experience (estimate if possible, otherwise "unknown")

    Rules:
    - Include EVERY candidate exactly once
    - Use one line per candidate

    Output format:
    1. Full Name — Job Title — X years of experience
    """

    list_qe = summary_index.as_query_engine(response_mode="tree_summarize")
    list_response = list_qe.query(list_prompt)

    print("\n===== CANDIDATE LIST =====")
    print(list_response)


    # SUMMARY FOR ONE CANDIDATE (by candidate_id)
    candidate_id = 0  # change this id

    summary_prompt = """
    You are analyzing a single candidate resume.

    Task:
    Create a concise professional summary of this candidate.

    Include:
    1. Short professional summary (3–4 sentences)
    2. Key skills (bullet list)
    3. Work experience overview:
       - total years of experience
       - main roles and industries

    Rules:
    - Do NOT invent information
    - Use only what is present in the resume
    - Be concise and factual
    """

    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="candidate_id", operator=FilterOperator.EQ, value=candidate_id
            ),
        ]
    )

    summary_qe = vector_index.as_query_engine(similarity_top_k=1, filters=filters)

    summary_response = summary_qe.query(summary_prompt)

    print(f"\n===== SUMMARY FOR CANDIDATE {candidate_id} =====")
    print(summary_response)

if __name__ == "__main__":
    main()
