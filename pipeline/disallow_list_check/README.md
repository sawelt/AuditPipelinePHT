# Regex Disallowlist Matcher

This script is designed to check if each regex in the `pipeline/regex_disallowlist.txt` file matches any content in the files located in the `train/` directory. If a match is found, the script will print the regex and the corresponding file name.

## Prerequisites

Before running the script, ensure that you have the following:

1. The `pipeline/regex_disallowlist.txt` file containing the list of regular expressions to check against the files in the `train/` directory.
2. Files in the `train/` directory that need to be checked for matches.

## How to Use

1. Make sure the script file is executable. If not, set the appropriate permissions using the following command:

   ```bash
   chmod +x script.sh
   ```

2. Execute the script by passing the path to the `train/` directory as an argument:

   ```bash
   ./script.sh train/
   ```

## Script Behavior

1. The script reads the contents of `pipeline/regex_disallowlist.txt` and stores it in the `disallowlist` variable.
2. If the `disallowlist` is empty, the script will exit early, as there are no regex patterns to check.
3. The script initializes an empty JSON array called `matches`, which will store the matched patterns and corresponding file names.
4. For each regex in the `disallowlist`, the script iterates over all files in the `train/` directory.
5. It uses `grep` to check if the current regex is found in each file.
6. If a match is found, the regex and the file name are added to the `matches` array in JSON format.
7. The script prints a message for each matching file found, displaying the matched pattern and the file name.
8. After processing all the regex patterns and files, the `matches` JSON array is written to `regex_matches.json` in the current directory.

**Note:** The script assumes that the `jq` command-line tool is installed in the system to manipulate JSON data.

## Output

The script will create a file named `regex_matches.json` in the current directory. This file will contain an array of objects, each representing a match between a regex pattern and a file in the `train/` directory. The JSON structure will be as follows:

```json
[
  {
    "pattern": "<regex_pattern_1>",
    "file": "<matching_file_path_1>"
  },
  {
    "pattern": "<regex_pattern_2>",
    "file": "<matching_file_path_2>"
  },
  ...
]
```

**Note:** The `<regex_pattern_X>` and `<matching_file_path_X>` placeholders will be replaced with the actual regex pattern and the corresponding file path resulting from the script's execution. If no matches are found, the `regex_matches.json` file will contain an empty array `[]`.

Ensure that the script is run from the appropriate working directory and that the necessary files and directories are correctly set up to obtain accurate results.

---