from unicodedata import category


def hello_world():
    environment = "outdoor"
    categories = ["0.675 -> patio", "0.064 -> porch", "0.043 -> restaurant_patio", "0.027 -> courtyard", "0.014 -> zen_garden"]
    scene_attributes = "man-made, no horizon, natural light, foliage, wood, open area, vegetation, leaves, reading"
    img_CAM = "static/CogModal.png"
    time = 2.0
    return environment, categories, scene_attributes, img_CAM, time