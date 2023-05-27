import torch
from transformers import SamModel, SamProcessor
from segmentation_test.utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

model = SamModel.from_pretrained('facebook/sam-vit-huge').to(device)
processor = SamProcessor.from_pretrained('facebook/sam-vit-huge')


if __name__ == '__main__':
    from PIL import Image
    import requests
    from matplotlib import pyplot as plt

    image_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
    image = Image.open(requests.get(image_url, stream=True).raw).convert('RGB')

    plt.imshow(image)

    # STEP 1: Retrieve the image embeddings
    inputs = processor(image, return_tensors="pt").to(device)
    image_embeddings = model.get_image_embeddings(inputs['pixel_values'])

    input_points = [[[450, 600]]]
    show_points_on_image(image, input_points[0])

    inputs = processor(image, input_points=input_points, return_tensors="pt").to(device)
    # pop the pixel_values as they are not neded
    inputs.pop("pixel_values", None)
    inputs.update({"image_embeddings": image_embeddings})

    with torch.no_grad():
        outputs = model(**inputs)

    masks = processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(),
                                                         inputs["reshaped_input_sizes"].cpu())
    scores = outputs.iou_scores

    show_masks_on_image(image, masks[0], scores)
