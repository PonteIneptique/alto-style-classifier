import matplotlib.image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from stylalto.datasets.extractor import (
    get_image_locally,
    get_alto_bboxes,
    extract_styles,
    NS
)


def vizualise_from_file(xml: "lxml.etree.ElementBase", xml_filepath: str):
    img_path = get_image_locally(xml, path_xml=xml_filepath, temp_fix=False)
    styles = extract_styles(xml)
    bboxes = get_alto_bboxes(xml)
    cmap = plt.get_cmap("Dark2").colors

    colors = {
        style: color
        for style, color in zip(styles, cmap)
    }

    figure = plt.figure(1, figsize=(10, 20))
    ax = figure.add_subplot(111)
    img = matplotlib.image.imread(img_path)
    ax.imshow(img)

    rects = {}
    for bbox in bboxes:
        rect = patches.Rectangle(
            xy=(bbox.x1, bbox.y1),
            width=bbox.x2 - bbox.x1, height=bbox.y2 - bbox.y1,
            alpha=0.4,
            fill=True,
            color=colors[bbox.style]
        )
        ax.add_patch(rect)
        if styles[bbox.style] not in rects:
            rects[styles[bbox.style].replace("_", "Normal")] = rect

    ax.legend(handles=[
        patches.Rectangle(
            xy=(5, 5), width=5, height=5,
            color=color, label=styles[style].replace("_", "Normal")
        )
        for style, color in colors.items()
    ])
    return figure
