import math
import json

frame_path = "../viewer/frame_data_20241220_153819.json"

def calculate_distance(cone1, cone2):
    return math.sqrt((cone1["x_ground_m"] - cone2["x_ground_m"]) ** 2 + (cone1["y_ground_m"] - cone2["y_ground_m"]) ** 2)

def find_closest_cone(reference, detected_cones, threshold):
    closest_cone = None
    min_distance = float('inf')
    for detected_cone in detected_cones:
        distance = calculate_distance(reference, detected_cone)
        if distance < min_distance and distance <= threshold:
            min_distance = distance
            closest_cone = detected_cone
    return closest_cone, min_distance

def match_cones(reference_cones, detected_cones, threshold):
    matches = []
    unmatched_references = []
    unmatched_detected = detected_cones.copy()

    for reference in reference_cones:
        closest_cone, error = find_closest_cone(reference, unmatched_detected, threshold)
        if closest_cone:
            matches.append({
                "reference": reference,
                "detected": closest_cone,
                "error": error,
                "class_match": reference["class_id"] == closest_cone["class_id"]
            })
            unmatched_detected.remove(closest_cone)
        else:
            unmatched_references.append(reference)

    return matches, unmatched_references, unmatched_detected

# Load data from your JSON file
with open(frame_path, "r") as file:
    data = json.load(file)

reference_cones = data["reference_cones"]
detected_cones = data["detected_cones"]

# Define matching threshold (e.g., 2 meters)
matching_threshold = 2.0

matches, unmatched_references, unmatched_detected = match_cones(reference_cones, detected_cones, matching_threshold)

error_sum = 0

# Output results
print("Matched Cones:")
for match in matches:
    error_sum += match["error"]
    print(f"Reference: {match['reference']['name']}, Detected: {match['detected']['name']}, "
          f"Distance: {match['error']:.3f}m, Class Match: {match['class_match']}")

print(f"\nAverage Error: {error_sum / len(matches):.3f}m")

print("\nUnmatched Reference Cones:")
for ref in unmatched_references:
    print(ref)

print("\nUnmatched Detected Cones:")
for det in unmatched_detected:
    print(det)
