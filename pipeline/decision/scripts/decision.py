import json
import math
import os
from typing import Union

vulnerability_per_line = -1
static_score = -1
sast_score = -1
dependency_score = -1
secret_score = -1
dynamic_score = -1
image_score = -1
compliance_score = -1

forbidden_pattern_found = False

remarks = []

# read static_report.json
try:
    with open("static_report.json", "r") as f:
        static_report = json.load(f)
        static_score = 0
        sast_score = 0
        dependency_score = 0
        secret_score = 0
        forbidden_pattern_matches = static_report["forbidden_pattern_matches"]
        total_critical = static_report["total"]["critical"]
        if len(forbidden_pattern_matches) > 0:
            remarks.append(
                "Found forbidden patterns in the code: {}.".format(forbidden_pattern_matches))
            forbidden_pattern_found = True
        if total_critical > 0:
            remarks.append(
                f"Found {total_critical} critical {'vulnerability' if total_critical == 1 else 'vulnerabilities'} in the static analysis."
            )
        number_of_lines = static_report["n_lines"]
        if len(forbidden_pattern_matches) == 0 and total_critical == 0:
            total_low = static_report["total"]["low"]
            total_medium = static_report["total"]["medium"]
            total_high = static_report["total"]["high"]
            vulnerability_per_line = (total_critical * 8 +
                                      total_high * 4 + total_medium * 2 + total_low) / number_of_lines
            # TODO: find a suitable formula
            static_score = 1 / math.exp(vulnerability_per_line)
        sast_critical = static_report["sast"]["critical"]
        sast_high = static_report["sast"]["high"]
        sast_medium = static_report["sast"]["medium"]
        sast_low = static_report["sast"]["low"]
        sast_score = 1 / math.exp(
            (sast_critical * 8 + sast_high * 4 + sast_medium * 2 + sast_low) / number_of_lines)
        
        if "dependency_scanning" in static_report:
            dependency_critical = static_report["dependency_scanning"]["critical"]
            dependency_high = static_report["dependency_scanning"]["high"]
            dependency_medium = static_report["dependency_scanning"]["medium"]
            dependency_low = static_report["dependency_scanning"]["low"]
            dependency_score = 1 / math.exp(
                (dependency_critical * 8 + dependency_high * 4 + dependency_medium * 2 + dependency_low) / number_of_lines)
        secret_critical = static_report["secret_detection"]["critical"]
        secret_high = static_report["secret_detection"]["high"]
        secret_medium = static_report["secret_detection"]["medium"]
        secret_low = static_report["secret_detection"]["low"]
        secret_score = 1 / math.exp(
            (secret_critical * 8 + secret_high * 4 + secret_medium * 2 + secret_low) / number_of_lines)


except FileNotFoundError:
    print("No static_report.json found. Skipping...")

# read dynamic_report.json
try:
    with open("dynamic_report.json", "r") as f:
        dynamic_report = json.load(f)
        dynamic_score = 0
        tx_bytes = dynamic_report["tx_bytes"]
        if tx_bytes == 0:
            station_size_differences: dict[str,
                                           int] = dynamic_report["station_size_differences"]
            max_station_size_difference = max(
                station_size_differences.values())
            max_cpu_usage = dynamic_report["max_cpu_usage"]
            max_memory_usage = dynamic_report["max_memory_usage"]
            max_n_pids = dynamic_report["max_n_pids"]
            rx_bytes = dynamic_report["rx_bytes"]
            # TODO: find a suitable formula
            dynamic_score = 1 / ((max_station_size_difference + max_cpu_usage + max_memory_usage
                                 + max_n_pids + rx_bytes + tx_bytes + 1) / (1024 * 1024 * 64))
            # dynamic_score = 0.4
        else:
            remarks.append(
                "Identified network traffic in dynamic analysis.")
except FileNotFoundError:
    print("No dynamic_report.json found. Skipping...")

# read image_report.json
try:
    with open("image_report.json", "r") as f:
        image_report = json.load(f)
        n_critical_vulnerabilities = 0
        n_high_vulnerabilities = 0
        n_medium_vulnerabilities = 0
        n_low_vulnerabilities = 0
        image_score = 0
        for vulnerability in image_report["vulnerabilities"]:
            severity = vulnerability["severity"]
            if severity == "critical":
                n_critical_vulnerabilities += 1
            elif severity == "high":
                n_high_vulnerabilities += 1
            elif severity == "medium":
                n_medium_vulnerabilities += 1
            elif severity == "low":
                n_low_vulnerabilities += 1
        if n_critical_vulnerabilities > 0:
            remarks.append(
                f"Found {n_critical_vulnerabilities} critical {'vulnerability' if n_critical_vulnerabilities == 1 else 'vulnerabilities'} in the image.")
        else:
            image_score = 1 / math.log(n_high_vulnerabilities * 4 +
                                       n_medium_vulnerabilities * 2 + n_low_vulnerabilities)
except FileNotFoundError:
    print("No image_report.json found. Skipping...")

# read compliance_report.json
try:
    with open("compliance_report.json", "r") as f:
        compliance_report = json.load(f)
        compliance_score = 0
        if compliance_report["standards_compliance"]:
            compliance_score = 1
        else:
            remarks.append(
                "Couldn't detect compliance with standards (usage of the python module padme_conductor [https://pypi.org/project/padme-conductor]).")
except FileNotFoundError:
    print("No compliance_report.json found. Skipping...")


if static_score == -1 and dynamic_score == -1 and image_score == -1 and compliance_score == -1:
    print("No reports found. Exiting...")
    exit(1)


type tree = dict[Union[str, bool], Union[str, dict, bool]]

class DecisionModel:
    thresholds: dict[str, float]

    decision_tree: tree

    def __init__(self, decision_model: dict[str, Union[tree, float]]):
        required_thresholds = [
            "sast_threshold",
            "static_threshold",
            "dependency_threshold",
            "secret_threshold",
            "standard_threshold",
            "blacklist_threshold",
            "image_threshold",
            "dast_threshold"
        ]
        # validate thresholds
        for key in required_thresholds:
            if key not in decision_model["thresholds"]:
                raise KeyError(f"Missing threshold for {key}")
            if not isinstance(decision_model["thresholds"][key], float):
                raise ValueError(f"Threshold for {key} is not a float")
            if decision_model["thresholds"][key] < 0 or decision_model["thresholds"][key] > 1:
                raise ValueError(f"Threshold for {key} is not between 0 and 1")
            
        self.thresholds = decision_model["thresholds"]
        
        # validate tree
        # example structure:
        # subtree = {
        #     "threshold": {
        #         "operator": ">=",
        #         True: another_tree,
        #         False: False
        #     }
        # }
        # True accepts train, false rejects train
        def validate_subtree(subtree: tree):
            for key in subtree:
                # check for valid key
                if not key.replace("score", "threshold") in required_thresholds:
                    raise ValueError(f"Unknown key found in subtree: {key}")
                # explore subtree
                if isinstance(subtree[key], dict):
                    # check for valid operator
                    if not "operator" in subtree[key]:
                        raise KeyError(f"Subtree {subtree} does not have operator key")
                    op = subtree[key]["operator"]
                    valid_operators = [">=", "<=", "<", ">"]
                    if not op in valid_operators:
                        raise ValueError(f"Operator {op} is not valid. Should be one of {valid_operators}")
                    
                    # check subtrees for positive and negative outcome
                    for outcome in ["true", "false"]:
                        if not outcome in subtree[key]:
                            raise KeyError(f"{outcome} key missing: No instructions for this case found")
                        if isinstance(subtree[key][outcome], dict):
                            validate_subtree(subtree[key][outcome])
                        elif not isinstance(subtree[key][outcome], bool):
                            raise ValueError(f"Neither bool nor dict found for case {outcome} in {subtree[key]}")
                
        validate_subtree(decision_model["decision_tree"])

        self.decision_tree = decision_model["decision_tree"]


    def eval(self, sast_score: float, static_score: float, dependency_score: float, secret_score: float, 
             blacklist_score: float, image_score: float, dast_score: float, compliance_score: int, 
             forbidden_pattern_found: bool, remarks: list) -> dict[str, any]:
        lookup = {
            "sast_score": sast_score,
            "static_score": static_score,
            "dependency_score": dependency_score,
            "secret_score": secret_score,
            "standard_score": compliance_score,
            "blacklist_score": blacklist_score,
            "image_score": image_score,
            "dast_score": dast_score
        }

        def eval_subtree(subtree: tree) -> bool:
            for key in subtree:
                score = lookup[key]
                # instant reject trains when necessary report is missing
                if score == -1:
                    remarks.append("Rejecting train because report is missing to calculate {}.".format(key))
                    return False
                threshold = self.thresholds[key.replace("score", "threshold")]
                op = subtree[key]["operator"]
                if op == ">=":
                    bool_key = score >= threshold
                elif op == "<=":
                    bool_key = score <= threshold
                elif op == "<":
                    bool_key = score < threshold
                else:
                    bool_key = score > threshold

                bool_key = "true" if bool_key else "false"
                next = subtree[key][bool_key]
                if isinstance(next, bool):
                    return next
                else:
                    return eval_subtree(next)
        passed = eval_subtree(self.decision_tree)

        decision: dict[str, any] = {}
        decision["remarks"] = remarks
        for key in lookup:
            decision[key] = lookup[key] > self.thresholds[key.replace("score", "threshold")]
        decision["passed"] = passed
        decision["standards_compliance"] = compliance_score == 1
        decision["forbidden_pattern_found"] = forbidden_pattern_found

        return decision

decision_model = None
try:
    with open("decision_model.json", "r") as f:
        decision_model = json.load(f)
        decision_model = DecisionModel(decision_model)
except FileNotFoundError:
    print("No decision_model.json found. Using standard model instead...")

    threshold = float(os.environ.get(
    "SECURITY_AUDIT_ACCEPTANCE_THRESHOLD", 0.5))

    decision_model = DecisionModel({
        "thresholds": {
            "sast_threshold": threshold,
            "static_threshold": threshold,
            "dependency_threshold": threshold,
            "secret_threshold": threshold,
            "standard_threshold": threshold,
            "blacklist_threshold": threshold,
            "image_threshold": threshold,
            "dast_threshold": threshold
        },
        "decision_tree": {
            "sast_score": {
                "operator": ">=",
                "true": {
                    "dast_score": {
                        "operator": ">=",
                        "true": True,
                        "false": False
                    }
                },
                "false": False
            }
        }
    })
finally:
    missing_reports = []

    if dynamic_score == -1:
        missing_reports.append("dynamic_report.json")
    if image_score == -1:
        missing_reports.append("image_report.json")
    if static_score == -1:
        missing_reports.append("static_report.json")
    if compliance_score == -1:
        missing_reports.append("compliance_report.json")

    if len(missing_reports) > 0:
        remarks.append(
            "The reports {} are missing.".format("".join([", ".join(missing_reports[:-1]), " and " if len(missing_reports) > 1 else "", missing_reports[-1]])))

    decision = decision_model.eval(
        sast_score, static_score, dependency_score, secret_score, forbidden_pattern_found, image_score, dynamic_score, compliance_score, forbidden_pattern_found, remarks)

    with open("decision.json", "w") as f:
        json.dump(decision, f)
