import json

from api.agents.graph import compiled_graph


def run_test():
    print("Starting local pipeline test...\n")

    # Define the initial state (same as what main.py creates)
    initial_state = {
        "document_name": "test_contract.pdf",
        "file_bytes": None,
        "file_path": None,
        "page_count": 0,
        "raw_text": "",
        "clauses": [],
        "analyzed_risks": [],
        "validation_passed": False,
        "retry_count": 0,
        "judge_feedback": "",
        "final_report": None,
    }

    # Run the pipeline
    final_state = compiled_graph.invoke(initial_state)

    # Print the final report nicely
    report = final_state.get("final_report")
    if report:
        print("\nPipeline Finished Successfully!\n")
        print("Final Report Output:")
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print("\nPipeline failed to generate a report.")


if __name__ == "__main__":
    run_test()
