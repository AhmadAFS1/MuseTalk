#!/usr/bin/env python3
"""Bind an authored license assessment to frozen hashes and retained notices.

Offline evidence assembly only. Does not create a deployable release manifest,
approve an image, modify a model, or change the runtime release gate.
"""
import argparse
import json
from pathlib import Path

import release


def assemble(dossier, rules_path):
    rules = json.loads(rules_path.read_text())
    index_path = dossier / "notice-reference-inventory.json"
    binding_path = dossier / "3090-model-byte-binding.json"
    index = json.loads(index_path.read_text())
    binding = json.loads(binding_path.read_text())
    supplemental = json.loads((dossier / "supplemental/sources.json").read_text())
    release.require(rules.get("schema") == "musetalk_public_payload_assessment_rules_v1", "Wrong assessment rules")
    release.require(binding.get("reference_inventory_sha256") == release.sha256(index_path), "Binding/reference checksum differs")
    reference = {x["target_path"]: x for x in index["models"]}
    captured = {x["target_path"]: x for x in binding["models"]}
    release.require(len(reference) == len(index["models"]) and len(captured) == len(binding["models"]), "Duplicate model path")
    notices = dict(index["notices"])
    notices.update({"supplemental/" + item["path"]: item for item in supplemental["sources"].values()})
    output = []
    for model_id, group in rules["model_groups"].items():
        release.require(group["assessment"] == "ELIGIBLE_WITH_LISTED_NOTICES", "Unexpected model disposition")
        retained = {}
        for name in group["notices"]:
            entry = notices[name]
            release.check_file(dossier, name, entry)
            retained[name] = {k: entry[k] for k in ("sha256", "size_bytes")}
            retained[name]["source_url"] = entry.get("source_url", entry.get("url"))
        for name in group["expected_paths"]:
            release.relative(name)
            expected, actual = reference[name], captured[name]
            release.require(expected["model_id"] == model_id == actual["model_id"], "Model group differs")
            release.require(actual["byte_binding"] == "MATCH", "Model bytes are not matched")
            for key in ("sha256", "size_bytes"):
                release.require(expected[key] == actual[key] == actual["actual"][key], "Captured/reference model bytes differ")
            for key in ("repository", "revision", "upstream_path"):
                release.require(actual[key] == expected[key], "Captured/reference source identity differs")
            output.append({"path": name, "sha256": expected["sha256"], "size_bytes": expected["size_bytes"],
                           "model_id": model_id, "upstream_repository": expected["repository"],
                           "upstream_revision": expected["revision"], "license_basis": group["license_basis"],
                           "disposition": group["assessment"], "upstream_redistribution_grant_identified": True,
                           "retained_notices": retained, "conditions": group["conditions"]})
    release.require(len(output) == 10 and len({x["path"] for x in output}) == 10, "Expected exact ten-file scope")
    release.require(not {x["path"] for x in output} & set(rules["excluded_model_payloads"]["paths"]), "HOLD model included")
    return {"schema": "musetalk_public_model_file_license_map_v1", "review_date_utc": rules["review_date_utc"],
            "scope": rules["scope"], "rules_sha256": release.sha256(rules_path),
            "byte_binding_sha256": release.sha256(binding_path),
            "captured_inventory_sha256": binding["captured_inventory_sha256"],
            "runtime_package_metadata_sha256": release.sha256(dossier / "runtime-package-metadata.json"),
            "additional_primary_runtime_sources": rules["additional_primary_runtime_sources"],
            "final_image_publication_decision": "PENDING_SPECIFIC_RUNTIME_GATES_NOT_A_BLANKET_MODEL_HOLD",
            "historical_review_note": "Supersedes the notice/byte-binding-pending disposition only for these ten unchanged files in the earlier model review; does not override HOLD weights or certify a built image.",
            "model_files": output, "excluded_model_payloads": rules["excluded_model_payloads"],
            "kokoro": rules["kokoro"], "runtime_components": rules["runtime_components"],
            "remaining_release_gates": rules["remaining_release_gates"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dossier", type=Path, required=True)
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = assemble(args.dossier, args.rules)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "eligible_unchanged_model_files": len(result["model_files"]),
                      "final_image_publication_decision": result["final_image_publication_decision"]}))
