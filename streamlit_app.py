import streamlit as st
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

st.set_page_config(page_title="AI Recipe Generator", layout="wide")

# ----------------------------------------
# Load model
# ----------------------------------------
@st.cache_resource
def load_model():
    base_model = "distilgpt2"
    adapter_model = "recipe_model_finetuned"    # folder you unzipped

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    st.write("🔄 Loading model…")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    st.write("🔄 Loading LoRA adapter…")
    model = PeftModel.from_pretrained(model, adapter_model)
    model = model.merge_and_unload()  # merges LoRA into base weights for faster inference
    model.eval()

    return tokenizer, model


tokenizer, model = load_model()


# ----------------------------------------
# Generate recipe
# ----------------------------------------
def generate_recipe(ingredients, diet, calories, cuisine):
    prompt = "### Recipe Request ###\n"
    prompt += f"Ingredients: {', '.join(ingredients)}\n"
    if diet:
        prompt += f"Dietary: {', '.join(diet)}\n"
    if calories:
        prompt += f"Max Calories: {calories}\n"
    if cuisine:
        prompt += f"Cuisine: {cuisine}\n"
    prompt += "\n### Recipe ###\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=400,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Extract JSON part
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        data = json.loads(text[start:end])
        return data
    except:
        return None


# ----------------------------------------
# UI
# ----------------------------------------
st.title("🍽️ AI Recipe Generator")
st.subheader("Create personalized recipes using your fine-tuned Food.com model")

st.write("---")

col1, col2 = st.columns(2)

with col1:
    ingredients = st.text_area(
        "Available Ingredients (comma-separated)",
        "chicken, garlic, onion, tomato"
    )
    ingredients = [x.strip() for x in ingredients.split(",") if x.strip()]

with col2:
    diet = st.multiselect(
        "Dietary Preferences",
        ["vegan", "vegetarian", "keto", "gluten-free", "dairy-free", "low-fat", "high-protein"]
    )
    calories = st.number_input("Max Calories (optional)", min_value=50, max_value=2000, value=None)
    cuisine = st.text_input("Cuisine (optional)", "")

st.write("---")

if st.button("Generate Recipe 🍳"):
    with st.spinner("Cooking something delicious…"):
        result = generate_recipe(ingredients, diet, calories, cuisine)

    if result:
        st.success("Recipe Generated!")

        st.subheader(result.get("recipe_name", "Recipe"))

        st.write("### 📝 Description")
        st.write(result.get("description", ""))

        st.write("### 🥦 Ingredients")
        for i in result.get("ingredients", []):
            st.write("• " + i)

        st.write("### 🍳 Instructions")
        for step in result.get("instructions", []):
            st.write("➡️ " + step)

        st.write("### 🔢 Nutrition")
        st.json(result.get("nutrition_estimate", {}))

        st.write("### 🕒 Prep & Cook Time")
        st.write(result.get("prep_time", ""))
        st.write(result.get("cook_time", ""))

    else:
        st.error("The model did not return a proper recipe JSON.")
