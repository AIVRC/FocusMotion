import json

input_json_path = "/home/yanghaotian/server_data/yanghaotian/test/Musepose_copy1/meta2/ubc_train_meta.json"      # your original json file
output_json_path = "/home/yanghaotian/server_data/yanghaotian/test/Musepose_copy1/meta2/ubc_meta.json"    # file to save updated json

# Load JSON
with open(input_json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Add "label": 0 to each object
if isinstance(data, list):
    for item in data:
        if isinstance(item, dict):
            item["label"] = 0
else:
    raise ValueError("JSON root must be a list of objects.")

# Save updated JSON
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4)

print("Done. Updated file saved to", output_json_path)
