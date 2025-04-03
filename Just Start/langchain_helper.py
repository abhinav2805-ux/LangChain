import ollama

# Define the model name
model_name = "gemma3:1b"

# Function to generate restaurant name
def generate_restaurant_name(cuisine):
    prompt = f"Generate a unique restaurant name for a {cuisine} cuisine."
    response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

# Function to generate menu
def generate_menu(restaurant_name):
    prompt = f"Generate a menu with 5 dishes for the restaurant {restaurant_name}."
    response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

# Main function
def generate_restaurant_and_menu(cuisine):
    restaurant_name = generate_restaurant_name(cuisine)
    menu = generate_menu(restaurant_name)
    return {"restaurant_name": restaurant_name, "menu": menu}



