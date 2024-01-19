import io
import csv
import os
import base64
import json
import streamlit as st
import pandas as pd
import torch
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Patch
import numpy as np
import easyocr
from transformers import AutoModelForObjectDetection, AutoImageProcessor
from torchvision import transforms
from pdf2image import convert_from_path
from tempfile import TemporaryDirectory
import shutil
import fitz
from pdf2image import convert_from_bytes

device = "cuda" if torch.cuda.is_available() else "cpu"

class MaxResize(object):
    def __init__(self, max_size=800):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.resize((int(round(scale * width)), int(round(scale * height))))
        return resized_image

detection_transform = transforms.Compose([
    MaxResize(800),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

structure_transform = transforms.Compose([
    MaxResize(1000),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-detection", revision="no_timm").to(device)
structure_model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition-v1.1-all").to(device)
reader = easyocr.Reader(['en'])

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    width, height = size
    boxes = box_cxcywh_to_xyxy(out_bbox)
    boxes = boxes * torch.tensor([width, height, width, height], dtype=torch.float32)
    return boxes

def outputs_to_objects(outputs, img_size, id2label):
    m = outputs.logits.softmax(-1).max(-1)
    pred_labels = list(m.indices.detach().cpu().numpy())[0]
    pred_scores = list(m.values.detach().cpu().numpy())[0]
    pred_bboxes = outputs['pred_boxes'].detach().cpu()[0]
    pred_bboxes = [elem.tolist() for elem in rescale_bboxes(pred_bboxes, img_size)]

    objects = []
    for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
        class_label = id2label[int(label)]
        if not class_label == 'no object':
            objects.append({'label': class_label, 'score': float(score),
                            'bbox': [float(elem) for elem in bbox]})

    return objects

def visualize_detected_tables(img, det_tables):
    plt.imshow(img, interpolation="lanczos")
    fig = plt.gcf()
    fig.set_size_inches(20, 20)
    ax = plt.gca()

    for det_table in det_tables:
        bbox = det_table['bbox']

        if det_table['label'] == 'table':
            facecolor = (1, 0, 0.45)
            edgecolor = (1, 0, 0.45)
            alpha = 0.3
            linewidth = 2
            hatch = '//////'
        elif det_table['label'] == 'table rotated':
            facecolor = (0.95, 0.6, 0.1)
            edgecolor = (0.95, 0.6, 0.1)
            alpha = 0.3
            linewidth = 2
            hatch = '//////'
        else:
            continue

        rect = patches.Rectangle(bbox[:2], bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=linewidth,
                                 edgecolor='none', facecolor=facecolor, alpha=0.1)
        ax.add_patch(rect)
        rect = patches.Rectangle(bbox[:2], bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=linewidth,
                                 edgecolor=edgecolor, facecolor='none', linestyle='-', alpha=alpha)
        ax.add_patch(rect)
        rect = patches.Rectangle(bbox[:2], bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=0,
                                 edgecolor=edgecolor, facecolor='none', linestyle='-', hatch=hatch, alpha=0.2)
        ax.add_patch(rect)

    plt.xticks([], [])
    plt.yticks([], [])

    legend_elements = [Patch(facecolor=(1, 0, 0.45), edgecolor=(1, 0, 0.45),
                             label='Table', hatch='//////', alpha=0.3),
                       Patch(facecolor=(0.95, 0.6, 0.1), edgecolor=(0.95, 0.6, 0.1),
                             label='Table (rotated)', hatch='//////', alpha=0.3)]
    plt.legend(handles=legend_elements, bbox_to_anchor=(0.5, -0.02), loc='upper center', borderaxespad=0,
               fontsize=10, ncol=2)
    plt.gcf().set_size_inches(10, 10)
    plt.axis('off')

    return fig

def detect_and_crop_table(image, margin=20):
    pixel_values = detection_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(pixel_values)

    id2label = model.config.id2label
    id2label[len(model.config.id2label)] = "no object"
    detected_tables = outputs_to_objects(outputs, image.size, id2label)

    bbox = detected_tables[0]["bbox"]
    expanded_bbox = [
        max(0, bbox[0] - margin),
        max(0, bbox[1] - margin),
        min(image.width, bbox[2] + margin),
        min(image.height, bbox[3] + margin)
    ]

    cropped_table = image.crop(expanded_bbox)
    return cropped_table

def recognize_table(image):
    pixel_values = structure_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = structure_model(pixel_values)

    id2label = structure_model.config.id2label
    id2label[len(structure_model.config.id2label)] = "no object"
    cells = outputs_to_objects(outputs, image.size, id2label)

    draw = ImageDraw.Draw(image)

    for cell in cells:
        draw.rectangle(cell["bbox"], outline="red")

    return image, cells

def get_cell_coordinates_by_row(table_data):
    rows = [entry for entry in table_data if entry['label'] == 'table row']
    columns = [entry for entry in table_data if entry['label'] == 'table column']

    rows.sort(key=lambda x: x['bbox'][1])
    columns.sort(key=lambda x: x['bbox'][0])

    def find_cell_coordinates(row, column):
        cell_bbox = [column['bbox'][0], row['bbox'][1], column['bbox'][2], row['bbox'][3]]
        return cell_bbox

    cell_coordinates = []

    for row in rows:
        row_cells = []
        for column in columns:
            cell_bbox = find_cell_coordinates(row, column)
            row_cells.append({'column': column['bbox'], 'cell': cell_bbox})

        row_cells.sort(key=lambda x: x['column'][0])
        cell_coordinates.append({'row': row['bbox'], 'cells': row_cells, 'cell_count': len(row_cells)})

    cell_coordinates.sort(key=lambda x: x['row'][1])
    return cell_coordinates

def apply_ocr(cell_coordinates, cropped_table):
    data = dict()
    max_num_columns = 0
    for idx, row in enumerate(cell_coordinates):
        row_text = []
        for cell in row["cells"]:
            cell_image = np.array(cropped_table.crop(cell["cell"]))
            result = reader.readtext(np.array(cell_image))
            if len(result) > 0:
                text = " ".join([x[1] for x in result])
                row_text.append(text)

        if len(row_text) > max_num_columns:
            max_num_columns = len(row_text)

        data[str(idx)] = row_text

    for idx, row_data in data.copy().items():
        if len(row_data) != max_num_columns:
            row_data = row_data + ["" for _ in range(max_num_columns - len(row_data))]
        data[str(idx)] = row_data

    df = pd.DataFrame.from_dict(data, orient='index')
    return df, data
def download_csv(df, filename="output.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    st.markdown(f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV file</a>', unsafe_allow_html=True)

def download_json(data, filename="output.json"):
    json_str = json.dumps(data, indent=4)
    b64 = base64.b64encode(json_str.encode()).decode()
    st.markdown(f'<a href="data:file/json;base64,{b64}" download="{filename}">Download JSON file</a>', unsafe_allow_html=True)
def process_pdf(image, page_prefix):
    cropped_table = detect_and_crop_table(image)

    image, cells = recognize_table(cropped_table)

    cell_coordinates = get_cell_coordinates_by_row(cells)

    df, data = apply_ocr(cell_coordinates, image)

    # Save the modified image with bounding boxes
    image_with_boxes_path = f'detected_table_{page_prefix}.png'
    image.save(image_with_boxes_path)
    print(f"Detected table saved at: {image_with_boxes_path}")

    # Save the CSV file
    csv_path = f'output_{page_prefix}.csv'
    df.to_csv(csv_path, index=False)
    print(f"CSV file saved at: {csv_path}")

    # Save the JSON file
    json_path = f'output_{page_prefix}.json'
    with open(json_path, 'w') as json_file:
        json.dump(data, json_file)
    print(f"JSON file saved at: {json_path}")

    # Automatically download files
    st.markdown(get_image_download_link(image_with_boxes_path, f'Download Detected Image - Page {page_prefix}'), unsafe_allow_html=True)
    st.markdown(get_file_download_link(csv_path, f'Download CSV - Page {page_prefix}'), unsafe_allow_html=True)
    st.markdown(get_file_download_link(json_path, f'Download JSON - Page {page_prefix}'), unsafe_allow_html=True)


def get_image_download_link(file_path, text):
    with open(file_path, 'rb') as file:
        data = file.read()
        b64 = base64.b64encode(data).decode()
        return f'<a href="data:image/png;base64,{b64}" download="{file_path}">{text}</a>'

def get_file_download_link(file_path, text):
    with open(file_path, 'rb') as file:
        data = file.read()
        b64 = base64.b64encode(data).decode()
        return f'<a href="data:file/txt;base64,{b64}" download="{file_path}">{text}</a>'

def pdf_to_images(uploaded_file):
    # Read the PDF file from the uploaded file
    pdf_content = uploaded_file.read()

    # Convert the PDF to a list of PIL Images using pdf2image
    images = convert_from_bytes(pdf_content)

    return images
def main():
    st.set_page_config(page_title="Shareholder List Parser")

    st.title("SignalX - Shareholder List Parser")

    uploaded_file = st.file_uploader("Upload the shareholders list file...", type=["pdf"])
    if uploaded_file is not None:
        pdf_images = pdf_to_images(uploaded_file)

        for idx, pdf_image in enumerate(pdf_images):
            # Generate unique file names for each page
            page_prefix = f'page_{idx + 1}'

            # Process the current PDF page
            process_pdf(pdf_image, page_prefix)

            # Display the CSV content for the current page
            st.subheader(f"CSV Data - Page {idx + 1}")
            df = pd.read_csv(f'output_{page_prefix}.csv')
            st.write(df)

            # Display the JSON content for the current page
            st.subheader(f"JSON Data - Page {idx + 1}")
            with open(f'output_{page_prefix}.json') as json_file:
                json_data = json.load(json_file)
                st.json(json_data)

            # Display the image with bounding boxes for the current page
            st.subheader(f"Detected table - Page {idx + 1}")
            st.image(f'detected_table_{page_prefix}.png', use_column_width=True)

            # Clear the output files for the current page
            os.remove(f'output_{page_prefix}.csv')
            os.remove(f'output_{page_prefix}.json')
            os.remove(f'detected_table_{page_prefix}.png')

        st.header('Designed and developed by Mithun Shivakoti', divider='rainbow')

if __name__ == "__main__":
    main()
