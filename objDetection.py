# pip install gradio transformers pillow torch timm
import gradio as gr
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import ImageDraw

# Load the pre-trained model and processor
model_name = "facebook/detr-resnet-50"
processor = DetrImageProcessor.from_pretrained(model_name)
model = DetrForObjectDetection.from_pretrained(model_name)

def detect_objects(image):
    # Preprocess the image
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # Extract detected objects
    target_sizes = [image.size[::-1]]  # (height, width)
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    # Annotate the image with detected objects
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)
    for box, label, score in zip(results["boxes"], results["labels"], results["scores"]):
        box = [round(i, 2) for i in box.tolist()]
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0], box[1]), f"{model.config.id2label[label.item()]}: {score:.2f}", fill="red")

    return annotated_image

# Gradio interface
interface = gr.Interface(
    fn=detect_objects,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil"),
    title="Object Detection App",
    description="Upload an image to detect objects using a pre-trained Hugging Face model."
)

if __name__ == "__main__":
    interface.launch()