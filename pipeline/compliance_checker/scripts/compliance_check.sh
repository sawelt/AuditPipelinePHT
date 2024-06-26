# for each *.py file in train/ directory:

ls $1/*.py > train_files.txt

code_to_inject_line_0="import sys"
code_to_inject_line_1="try:"
code_to_inject_line_2="\    if not \"padme_conductor\" in sys.modules:"
code_to_inject_line_3="\        print(0)"
code_to_inject_line_4="\    else:"
code_to_inject_line_5="\        locals = vars().keys()"
# code_to_inject_line_6="\        if not any(name in locals for name in dir(padme_conductor)):"
code_to_inject_line_7="\            print(1)"
code_to_inject_line_8="\        else:"
code_to_inject_line_9="\            print(2)"
code_to_inject_line_10="except Exception:"
code_to_inject_line_11="\    print(3)"
code_to_inject_line_12="\    pass"
code_to_inject_line_13="exit()"

was_imported_and_used=0

while read python_file; do
    autopep8 --in-place --aggressive "$python_file"
    # Find the line number of the last import statement
    line_num=$(grep -n "^import" "$python_file" | tail -1 | awk -F: '{print $1}')
    if [ -z "$line_num" ]; then
        echo "$python_file: No import statement found"
        continue
    fi
    line_num=$((line_num+1))

    package_name="padme_conductor"
    package_name=$(grep -Po "(?<=import $package_name as )\w+" $python_file)
    if [ -z "$package_name" ]; then
        package_name="padme_conductor"
    fi

    code_to_inject_line_6="\        if not any(name in locals for name in dir($package_name)):"

    # Inject the code after the last import statement
    sed -i "${line_num}a ${code_to_inject_line_0}" "$python_file"
    line_num=$((line_num+1))
    sed -i "${line_num}a ${code_to_inject_line_1}" "$python_file"
    line_num=$((line_num+1))
    sed -i "${line_num}a ${code_to_inject_line_2}" "$python_file"
    line_num=$((line_num+1))
    sed -i "${line_num}a ${code_to_inject_line_3}" "$python_file"
    line_num=$((line_num+1))
    sed -i "${line_num}a ${code_to_inject_line_4}" "$python_file"
    line_num=$((line_num+1))
    sed -i "${line_num}a ${code_to_inject_line_5}" "$python_file"
    line_num=$((line_num+1))
    sed -i "${line_num}a ${code_to_inject_line_6}" "$python_file"
    line_num=$((line_num+1))
    sed -i "${line_num}a ${code_to_inject_line_7}" "$python_file"
    line_num=$((line_num+1))
    sed -i "${line_num}a ${code_to_inject_line_8}" "$python_file"
    line_num=$((line_num+1))
    sed -i "${line_num}a ${code_to_inject_line_9}" "$python_file"
    line_num=$((line_num+1))
    sed -i "${line_num}a ${code_to_inject_line_10}" "$python_file"
    line_num=$((line_num+1))
    sed -i "${line_num}a ${code_to_inject_line_11}" "$python_file"
    line_num=$((line_num+1))
    sed -i "${line_num}a ${code_to_inject_line_12}" "$python_file"
    line_num=$((line_num+1))
    sed -i "${line_num}a ${code_to_inject_line_13}" "$python_file"

    autopep8 --in-place --aggressive "$python_file"

    # exit_code=$(sudo python3 "$python_file")
    exit_code=$(python3 "$python_file")

    if [ $exit_code -eq 0 ]; then
        echo "$python_file: No padme_conductor module imported"
    elif [ $exit_code -eq 1 ]; then
        echo "$python_file: padme_conductor module was imported, but not used"
    elif [ $exit_code -eq 2 ]; then
        echo "$python_file: padme_conductor module was imported and used"
        was_imported_and_used=$((was_imported_and_used+1))
    else
        echo "$python_file: Runtime error"
    fi
done < train_files.txt

if [ $was_imported_and_used -eq 0 ]; then
    echo "No train/*.py file imported and used padme_conductor module"
    # write the exit code to file compliance_report.json
    echo "{\"standards_compliance\": false}" > compliance_report.json
    exit 0
fi

echo "At least one train/*.py file imported and used padme_conductor module"
echo "{\"standards_compliance\": true}" > compliance_report.json
exit 0
