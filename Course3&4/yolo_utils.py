import colorsys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Load class names from .txt file
def read_classes(classes_path):
    with open(classes_path, 'r') as f:
        class_names = f.read().splitlines()
    return class_names


# Load YOLO anchors from .txt file
def read_anchors(anchors_path):
    with open(anchors_path, 'r') as f:
        anchors = f.read().split(',')
        anchors = [float(x) for x in anchors]
        anchors = np.array(anchors).reshape(-1, 2)
    return anchors


# Preprocess input image → fit network input size (608x608 usually)
def preprocess_image(image_path, model_image_size):
    image = Image.open(image_path)
    resized_image = image.resize(tuple(model_image_size), Image.BICUBIC)
    image_data = np.array(resized_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # shape: (1,h,w,3)
    return image, image_data


# Generate distinct colors for boxes
def generate_colors(class_names):
    hsv = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    colors = [(int(r * 255), int(g * 255), int(b * 255)) for (r, g, b) in colors]
    np.random.seed(10101)
    np.random.shuffle(colors)
    np.random.seed(None)
    return colors


# Box scaling for original image
def scale_boxes(boxes, image_shape):
    height, width = image_shape
    image_dims = np.array([height, width, height, width])
    return boxes * image_dims


# Draw YOLO boxes
# def draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors):
#     font = ImageFont.load_default()
#     thickness = (image.size[0] + image.size[1]) // 300

#     for i, c in enumerate(out_classes):
#         predicted_class = class_names[c]
#         box = out_boxes[i]
#         score = out_scores[i]

#         label = '{} {:.2f}'.format(predicted_class, score)
#         top, left, bottom, right = box
#         top = max(0, np.floor(top).astype('int32'))
#         left = max(0, np.floor(left).astype('int32'))
#         bottom = min(image.size[1], np.floor(bottom).astype('int32'))
#         right = min(image.size[0], np.floor(right).astype('int32'))

#         draw = ImageDraw.Draw(image)
#         label_size = draw.textsize(label, font)

#         if top - label_size[1] >= 0:
#             text_origin = (left, top - label_size[1])
#         else:
#             text_origin = (left, top + 1)

#         for t in range(thickness):
#             draw.rectangle(
#                 [left + t, top + t, right - t, bottom - t],
#                 outline=colors[c]
#             )
#         draw.rectangle(
#             [tuple(text_origin), (text_origin[0] + label_size[0], text_origin[1] + label_size[1])],
#             fill=colors[c]
#         )
#         draw.text(text_origin, label, fill=(0, 0, 0), font=font)
#         del draw

def draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors):
    # Load font (Try loading the specific font used in the assignment, fallback to default)
    try:
        font = ImageFont.truetype("font/FiraMono-Medium.otf", size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    except IOError:
        font = ImageFont.load_default()
    
    thickness = (image.size[0] + image.size[1]) // 300

    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = class_names[c]
        box = out_boxes[i]
        score = out_scores[i]

        label = '{} {:.2f}'.format(predicted_class, score)

        draw = ImageDraw.Draw(image)
        
        # --- FIX FOR PILLOW 10+ ---
        # Old: label_size = draw.textsize(label, font)
        # New: Use textbbox to calculate width and height
        bbox = draw.textbbox((0, 0), label, font)
        label_size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
        # --------------------------

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        print(label, (left, top), (right, bottom))

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        # Draw the bounding box
        for i in range(thickness):
            draw.rectangle(
                [left + i, top + i, right - i, bottom - i],
                outline=colors[c])
        
        # Draw the label background
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=colors[c])
        
        # Draw the text
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw

    return image
