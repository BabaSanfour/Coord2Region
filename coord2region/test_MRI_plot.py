import matplotlib.pyplot as plt
from nilearn import plotting, datasets

# Load the standard MNI152 template
mni_template = datasets.load_mni152_template()

# Define the MNI coordinate to highlight
coord = (30, 22, -8)

# Set marker properties
marker_color = 'orange'
marker_size = 100

# ---------------- Axial Slice ----------------
# Plot an axial slice at z = -8
display_axial = plotting.plot_anat(
    mni_template,
    display_mode='z',
    cut_coords=[coord[2]],
    annotate=False,
    draw_cross=False,
    title='Axial View'
)
# Add the marker at the coordinate location
display_axial.add_markers([coord], marker_color=marker_color, marker_size=marker_size)
plt.show()

# ---------------- Coronal Slice ----------------
# Plot a coronal slice at y = 22
display_coronal = plotting.plot_anat(
    mni_template,
    display_mode='y',
    cut_coords=[coord[1]],
    annotate=False,
    draw_cross=False,
    title='Coronal View'
)
display_coronal.add_markers([coord], marker_color=marker_color, marker_size=marker_size)
plt.show()

# ---------------- Sagittal Slice ----------------
# Plot a sagittal slice at x = 30
display_sagittal = plotting.plot_anat(
    mni_template,
    display_mode='x',
    cut_coords=[coord[0]],
    annotate=False,
    draw_cross=False,
    title='Sagittal View'
)
display_sagittal.add_markers([coord], marker_color=marker_color, marker_size=marker_size)
plt.show()
