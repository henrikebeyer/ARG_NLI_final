import pandas as pd

import pandas as pd

def find_marker_conflicts(file_paths):
    """
    Crosscheck multiple CSV lists of discourse markers
    and report which markers appear in more than one list.

    Parameters
    ----------
    file_paths : dict
        A dictionary mapping list names to CSV file paths.
        Example:
        {
            "deduction": "deduction_markers.csv",
            "refutation": "refutation_markers.csv",
            "condition": "condition_markers.csv"
        }

    Returns
    -------
    pd.DataFrame
        A DataFrame showing markers that appear in multiple lists,
        and which lists they belong to.
    """

    # Read and label each list
    dataframes = []
    for list_name, path in file_paths.items():
        df = pd.read_csv(path)
        df["Source_List"] = list_name.lower()
        df["Marker"] = df["Marker"].str.strip().str.lower()  # normalize
        dataframes.append(df[["Marker", "Source_List"]])

    # Combine all markers
    combined = pd.concat(dataframes, ignore_index=True)

    # Group by marker and collect all lists where each appears
    grouped = combined.groupby("Marker")["Source_List"].unique().reset_index()

    # Keep only those appearing in more than one list
    duplicates = grouped[grouped["Source_List"].apply(lambda x: len(x) > 1)]

    # Sort for readability
    duplicates = duplicates.sort_values("Marker").reset_index(drop=True)

    return duplicates

file_paths = {
    "deduction": "/home/oenni/Dokumente/NLI-Argumentation-project/feature_analysis/reference_lists/deduction_markers.csv",
    "refutation": "/home/oenni/Dokumente/NLI-Argumentation-project/feature_analysis/reference_lists/refutation_markers.csv",
    "condition": "/home/oenni/Dokumente/NLI-Argumentation-project/feature_analysis/reference_lists/condition_markers.csv",
    "explanation": "/home/oenni/Dokumente/NLI-Argumentation-project/feature_analysis/reference_lists/explanation_markers.csv",
    "adverbs_of_emphasis": "/home/oenni/Dokumente/NLI-Argumentation-project/feature_analysis/reference_lists/emphasis_adverbs.csv",
    "justification": "/home/oenni/Dokumente/NLI-Argumentation-project/feature_analysis/reference_lists/justification_markers.csv"
}

conflicts = find_marker_conflicts(file_paths)
print(conflicts)