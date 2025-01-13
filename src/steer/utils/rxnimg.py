import base64
from io import BytesIO

import cairosvg
import requests
from PIL import Image


def get_rxn_img(smiles) -> Image.Image | None:

    # The URL for the GET request
    url = "https://www.simolecule.com/cdkdepict/depict/cot/svg"

    # The parameters for the request
    params = {
        "smi": smiles,
        "w": "-1",
        "h": "-1",
        "abbr": "off",
        "hdisp": "S",
        "zoom": "1.3",
        "annotate": "none",
        "r": "0",
    }

    # Desired final image size
    final_size = (1456, 819)

    # Make the GET request
    response = requests.get(url, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        # Get the SVG content
        svg_content = response.content

        # Convert SVG to PNG
        png_data = cairosvg.svg2png(bytestring=svg_content, dpi=300)

        # Open the PNG image using PIL
        img = Image.open(BytesIO(png_data))

        # Calculate the scaling factor to fit the image within the final size
        img_ratio = img.width / img.height
        final_ratio = final_size[0] / final_size[1]

        if img_ratio > final_ratio:
            # Image is wider than the final aspect ratio
            new_width = final_size[0]
            new_height = int(final_size[0] / img_ratio)
        else:
            # Image is taller than the final aspect ratio
            new_height = final_size[1]
            new_width = int(final_size[1] * img_ratio)

        # Resize the image while maintaining aspect ratio
        img = img.resize((new_width, new_height))  # , Image.LANCZOS)

        # Create a new image with a white background and the desired final size
        final_img = Image.new("RGB", final_size, (255, 255, 255))

        # Calculate position to center the original image
        x_offset = (final_size[0] - img.size[0]) // 2
        y_offset = (final_size[1] - img.size[1]) // 2

        # Paste the original image onto the final image
        final_img.paste(
            img, (x_offset, y_offset), mask=img.split()[3]
        )  # Use the alpha channel as mask

        # Save the image to a BytesIO object in PNG format
        buffered = BytesIO()
        final_img.save(buffered, format="PNG")
        return final_img

        # Get the byte data and encode it to base64
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # # Save the image to a BytesIO object in PNG format
        # buffered = BytesIO()
        # img.save(buffered, format="PNG")

        # # Get the byte data and encode it to base64
        # img_str = base64.b64encode(buffered.getvalue()).decode()

        return img_str
    else:
        print(
            f"Failed to retrieve the SVG. Status code: {response.status_code}"
        )
        return None
