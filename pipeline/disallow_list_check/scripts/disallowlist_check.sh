# for each regex in pipeline/regex_disallowlist.txt, check if it fulfills for all files in directory train/
# if it does, print the regex and the file name

# read disallowlist
backup_path=$(dirname "$0")
disallowlist=$(cat "$backup_path/regex_disallowlist.txt")

# if disallowlist is empty, exit 
if [ -z "$disallowlist" ]
then
    echo "Disallowlist Regex List is empty"
    exit
fi

# init matches JSON array with elements { "pattern": "", "file": "" }
matches='[]'

# for each regex in disallowlist
for regex in $disallowlist
do
    # for each file in train/
    for file in $1/*
    do
        # check if regex is in file
        if grep -q $regex $file
        then
            # if it is, add it to matches
            matches=$(echo $matches | jq --arg regex "$regex" --arg file "$file" '. += [{ "pattern": $regex, "file": $file }]')
            echo "Found file content matching disallowlisted pattern in Train:"
            echo "Pattern: $regex"
            echo "File: $file"
        fi
    done
done

# write regex_matches to file
echo $matches > regex_matches.json
