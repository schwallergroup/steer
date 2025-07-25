"""
Some utilities to get reaction images from SMILES strings.
"""

import base64
import os
from io import BytesIO

import cairosvg
import requests
from PIL import Image


def get_manual_rxn_img(smiles) -> Image.Image:
    """
    Get a reaction image from a SMILES string.
    """

    reactants, _, products = smiles.split(">")

    reac_img = get_rxn_img(reactants)
    prod_img = get_rxn_img(products)

    # Resize the images to 668x375
    reac_img = reac_img.resize((668, 375))
    prod_img = prod_img.resize((668, 375))

    # Create a white background image of size 1456x819 with the arrow in the middle
    final_img_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "data",
        "images",
        "1456_819_arrow.png",
    )

    # Collate reactant at (0, 222) and product at (788, 222)
    final_img = Image.open(final_img_path)
    final_img.paste(reac_img, (0, 222))
    final_img.paste(prod_img, (787, 222))

    # Save the image to a BytesIO object in PNG format
    buffered = BytesIO()
    final_img.save(buffered, format="PNG")

    return final_img


def get_rxn_img(smiles, final_size: tuple = (1456, 819)) -> Image.Image:
    """
    Get a reaction image from a SMILES string using the Simolecule CDK Depict service.
    """

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
        "annotate": "colmap",  # "none"
        "r": "0",
    }

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


if __name__ == "__main__":
    # Test the function
    smiles = "[H]C([H])=C([H])C([H])([H])Br.[H]OC(C1=C([H])C([H])=C([H])C(N2C3=NC(N([H])C4=C([H])C([H])=C(Br)C([H])=C4[H])=NC([H])=C3C(=O)[N:1]2[H:1])=N1)(C([H])([H])[H])C([H])([H])[H]>>[H+:1].[H]C([H])=C([H])C([H])([H])Br.[H]OC(C1=C([H])C([H])=C([H])C(N2C3=NC(N([H])C4=C([H])C([H])=C(Br)C([H])=C4[H])=NC([H])=C3C(=O)[N-:1]2)=N1)(C([H])([H])[H])C([H])([H])[H]"
    smiles = "[H]O[C:1]([H])([H])[C:1]([H])([H])[H]>>[H]O[C+:1]([H])[H].[H][C-:1]([H])[H]"
    img = get_manual_rxn_img(smiles)
    img.show()
