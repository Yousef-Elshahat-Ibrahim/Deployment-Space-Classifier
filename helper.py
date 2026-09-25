from pipeline import classify_object

def classify_and_report(object_id, model, broker, transient_map):
    """Runs classify_object and prints a readable summary of the result."""
    probs = classify_object(object_id, model, broker=broker)[0]  # unwrap the (1, N) array

    # invert the map: {0: "SN", 1: "AGN", 2: "TDE"}
    idx_to_label = {v: k for k, v in transient_map.items()}

    # pair each label with its probability, sorted highest first
    ranked = sorted(
        ((idx_to_label[i], p) for i, p in enumerate(probs)),
        key=lambda x: x[1],
        reverse=True
    )

    top_label, top_prob = ranked[0]
    print(f"\nObject {object_id}  (broker: {broker})")
    print(f"  Prediction: {top_label}  ({top_prob:.2%})")
    for label, p in ranked:
        bar = "█" * int(p * 30)
        print(f"    {label:>4}: {p:6.2%}  {bar}")

    return top_label, dict(ranked)