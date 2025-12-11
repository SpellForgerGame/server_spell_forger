
import matplotlib.pyplot as plt


# Define your fields and max values per index field (for later validation/constraints if needed)
SPELL_FIELDS = [
    "type", "damage", "area", "speed",
    "rColor", "gColor", "bColor",
    "cost", "castSpeed", "cooldown", "areaChanged"
]

# Example ranges: damage is categorical (1-4), colors are [0–255], others can be adapted
NUM_CLASSES_PER_FIELD = {
    "type": 2,
    "damage": 5,
    "area": 10,
    "speed": 10,
    "rColor": 256,
    "gColor": 256,
    "bColor": 256,
    "cost": 10,
    "castSpeed": 10,
    "cooldown": 10,
    "areaChanged": 4
}



def get_spell_field_max(field_name):
  return NUM_CLASSES_PER_FIELD[field_name] - 1


def print_spell(spell_features):
    print("Predicted Spell Features:")
    # If spell_features is 2D, take the first row; otherwise, use as is
    features = spell_features[0] if len(spell_features.shape) > 1 else spell_features
    for i, field in enumerate(SPELL_FIELDS):
        predicted_value = features[i]
        if isinstance(predicted_value, float):
            predicted_value = round(predicted_value, 3)  # round to 3 decimals
        print(f"  - {field:12}: {predicted_value}/{get_spell_field_max(field)}")



def visualize_spell(labels):
    # If labels is 2D, take first row
    values = labels[0] if len(labels.shape) > 1 else labels

    if len(values) != len(SPELL_FIELDS):
        raise ValueError(f"Labels must match the SPELL_FIELDS length. Expected {len(SPELL_FIELDS)}, Found {len(values)}")

    # Extract color (rColor, gColor, bColor)
    r, g, b = values[4], values[5], values[6]
    r,g,b = int(r), int(g), int(b)
    if r == 0:
      r = 1
    if g == 0:
      g = 1
    if b == 0:
      b = 1

    color_preview = (r / 255, g / 255, b / 255)  # Normalize RGB for color display

    # Extract other stats: excluding rColor, gColor, bColor (index 4, 5, 6)
    stats_labels = SPELL_FIELDS[:4] + SPELL_FIELDS[7:]  # Removing color fields
    stats_values = list(values[:4]) + list(values[7:])  # Removing color values from stats

    # Create figure with 2 subplots (bar chart and color preview)
    fig, (ax_bar, ax_color) = plt.subplots(2, 1, figsize=(6, 6), gridspec_kw={'height_ratios': [4, 1]})
    fig.suptitle(f"Spell Visualization - Type: {values[0]}", fontsize=14)

    # Bar plot for stats (all fields except color)
    ax_bar.bar(stats_labels, stats_values, color='skyblue')
    ax_bar.set_ylabel("Index Value")
    ax_bar.set_ylim(0, max(10, max(stats_values) + 1))  # Adjust the y-axis range
    ax_bar.grid(axis='y', linestyle='--', alpha=0.7)
    ax_bar.set_xticklabels(stats_labels, rotation=45, ha='right')

    # Color preview box (for r, g, b colors)
    ax_color.set_facecolor(color_preview)
    ax_color.set_xticks([])  # Hide ticks
    ax_color.set_yticks([])  # Hide ticks
    ax_color.set_title(f"RGB Color: ({int(r)}, {int(g)}, {int(b)})")

    plt.tight_layout()  # Adjust layout for better fitting
    plt.show()
