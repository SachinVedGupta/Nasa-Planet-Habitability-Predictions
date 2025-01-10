import requests
from PIL import Image
from io import BytesIO
from predictions.make_predictions import predict_habitability

# Replace these with your actual API key and Search Engine ID
API_KEY = 'AIzaSyBNlRWSgkSbwhSulCMZmG-rYq_l5YJA72E'
SEARCH_ENGINE_ID = 'b7ab1235d87284bd3'


def get_first_image(query):
    # URL for Google Custom Search API
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'q': query,  # Search query
        'cx': SEARCH_ENGINE_ID,  # Custom Search Engine ID
        'key': API_KEY,  # API key
        'searchType': 'image',  # Search type is image
        'num': 1  # Number of results to return
    }

    response = requests.get(url, params=params)
    response.raise_for_status()  # Raise HTTPError for bad requests (4xx and 5xx)

    # Parse the JSON response
    results = response.json()
    if 'items' in results:
        # Extract the first image link
        image_url = results['items'][0]['link']
        return image_url
    else:
        print("No image results found.")
        return None


def display_image(query, width=300):
    image_url = get_first_image("An ultra-high-definition image of the planet " + query.title() + " as seen from space")

    # Define a User-Agent header
    headers = {
        'User-Agent': 'YourAppName/1.0 (https://yourwebsite.com; your_email@example.com)'
    }

    # Make the request with headers
    response = requests.get(image_url)

    if response.status_code != 200:
      print(f"URL for {query.title()} photo not found")
      return

    # Display the image
    image = Image.open(BytesIO(response.content))
    image.show()


def predict_habitability_with_visual(planet_info, name):
  prediction_text = predict_habitability(planet_info, name)

  print("\n\nSHOWING IMAGE OF " + name.title() + "\n\n")
  display_image(name)
  return prediction_text